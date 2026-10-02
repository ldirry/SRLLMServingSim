import bisect
import json
import random
import ast
import math
import os
from .logger import get_logger
from .request import Request

_SR_EPS = 1e-12


def _sr_add(a, b):
    return a + b


def _sr_sub(a, b):
    return a - b


def _sr_mul(a, b):
    return a * b


def _sr_pdiv(a, b):
    """Protected division."""
    return a / b if abs(b) > _SR_EPS else 1.0


def _sr_sqrtabs(x):
    """Protected sqrt."""
    return math.sqrt(abs(x))


def _sr_logabs(x):
    """Protected log."""
    return math.log1p(abs(x))


def _sr_square(x):
    return x * x


def _sr_absval(x):
    return abs(x)


_SR_FUNCS = {
    "add": _sr_add,
    "sub": _sr_sub,
    "mul": _sr_mul,
    "pdiv": _sr_pdiv,
    "sqrtabs": _sr_sqrtabs,
    "logabs": _sr_logabs,
    "square": _sr_square,
    "absval": _sr_absval,

    # Convenience aliases for hand-written equations:
    "sqrt": _sr_sqrtabs,
    "log": _sr_logabs,
    "abs": _sr_absval,
}

class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "LMETRIC":
            self._select_instance = self._lmetric
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

        self._custom_expr_text = None
        self._custom_expr_code = None

        if self.routing_policy == "CUSTOM":
            self._init_custom_expression()

        # ------------------------------------------------------------------
        # Prefix popularity x_p
        # ------------------------------------------------------------------

        # EWMA update strength.
        # Roughly: effective history ~ 1 / alpha arrivals.
        self._xp_alpha = float(os.environ.get("XP_ALPHA", "0.05"))

        if not (0.0 < self._xp_alpha <= 1.0):
            raise ValueError("XP_ALPHA must be in (0, 1]")

        # Fallback prefix-class definition when the workload has no explicit
        # prefix_id. This is number of leading token IDs used as the class key.
        self._xp_prefix_tokens = int(
            os.environ.get("XP_PREFIX_TOKENS", "256")
        )

        if self._xp_prefix_tokens <= 0:
            raise ValueError("XP_PREFIX_TOKENS must be > 0")

        # EWMA mass associated with each prefix class.
        self._xp_mass = {}

        # Sum of all EWMA masses. Used so XP stays normalized to [0, 1].
        self._xp_total_mass = 0.0

        self._xp_num_observations = 0

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role, req_data=None):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role, req_data=None):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role, req_data=None):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.waiting)
            running = len(sched.running)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, role, req_data=None):
        """
        Symbolic routing.

        For every candidate instance i:

            P_i, BS_i = live simulator state
            score_i = f(P_i, BS_i)

        Route to:

            argmin_i score_i
        """

        if role == "decode" or req_data is None:
            return self._least_load_select(
                schedulers,
                role,
                req_data,
            )

        num_instances = len(schedulers)

        if num_instances == 0:
            raise RuntimeError(
                "CUSTOM called with no candidate instances"
            )

        best_idx = None
        best_score = float("inf")

        # Deterministic fair tie-breaking.
        start = self._get_counter(role) % num_instances

        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]

            feat = self._routing_features(
                sched,
                req_data,
            )

            score = self._eval_custom_expression(
                feat["P"],
                feat["BS"],
                feat["XP"],
            )

            self.logger.debug(
                "CUSTOM expr=%s req=%d inst=%d "
                "P=%d BS=%d hit=%d "
                "queued_P=%d incoming_P=%d score=%g",
                self._custom_expr_text,
                req_data["index"],
                sched.instance_id,
                feat["P"],
                feat["BS"],
                feat["XP"],
                feat["prefix_hit"],
                feat["queued_p"],
                feat["incoming_p"],
                score,
            )

            if score < best_score:
                best_score = score
                best_idx = idx

        # A pathological expression might produce inf everywhere.
        if best_idx is None:
            self.logger.warning(
                "CUSTOM expression produced no valid score; "
                "falling back to LOAD"
            )

            return self._least_load_select(
                schedulers,
                role,
                req_data,
            )

        self._set_counter(
            role,
            (best_idx + 1) % num_instances,
        )

        return best_idx

    def _lmetric(self, schedulers, role, req_data=None):
        """LMETRIC: minimize P * BS."""

        if role == "decode" or req_data is None:
            return self._least_load_select(
                schedulers,
                role,
                req_data,
            )

        num_instances = len(schedulers)

        if num_instances == 0:
            raise RuntimeError(
                "LMETRIC called with no candidate instances"
            )

        best_idx = None
        best_score = float("inf")

        start = self._get_counter(role) % num_instances

        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]

            feat = self._routing_features(
                sched,
                req_data,
            )

            score = feat["P"] * feat["BS"]

            self.logger.debug(
                "LMETRIC req=%d inst=%d "
                "P=%d BS=%d hit=%d score=%d",
                req_data["index"],
                sched.instance_id,
                feat["P"],
                feat["BS"],
                feat["prefix_hit"],
                score,
            )

            if score < best_score:
                best_score = score
                best_idx = idx

        self._set_counter(
            role,
            (best_idx + 1) % num_instances,
        )

        return best_idx

    def _xp_prefix_key(self, req_data):
        """
        Return the prefix class p for this request.

        Preferred:
            explicit prefix_id supplied by the workload.

        Fallback:
            first XP_PREFIX_TOKENS input token IDs.
        """

        # Best option for controlled hotspot experiments.
        explicit = req_data.get("prefix_id")

        if explicit is not None:
            return ("prefix_id", str(explicit))

        # Existing LLMServingSim traces.
        token_ids = req_data.get("input_hash_ids", [])

        if not token_ids:
            return None

        n = min(
            len(token_ids),
            self._xp_prefix_tokens,
        )

        # Tuple is hashable and collision-free with respect to these token IDs.
        return ("tokens", tuple(token_ids[:n]))


    def _observe_prefix_popularity(self, req_data):
        """
        Observe one newly arrived request and return x_p:

            x_p = recent fraction of traffic belonging to prefix p

        using an exponentially weighted moving average over arrivals.

        Returns a normalized value in [0, 1].
        """

        key = self._xp_prefix_key(req_data)

        if key is None:
            return 0.0

        alpha = self._xp_alpha
        decay = 1.0 - alpha

        # Decay old observations.
        dead_keys = []

        for old_key, mass in self._xp_mass.items():
            new_mass = mass * decay

            if new_mass < 1e-12:
                dead_keys.append(old_key)
            else:
                self._xp_mass[old_key] = new_mass

        for old_key in dead_keys:
            del self._xp_mass[old_key]

        self._xp_total_mass *= decay

        # Current observation: I_p(t) = 1.
        self._xp_mass[key] = (
            self._xp_mass.get(key, 0.0)
            + alpha
        )

        self._xp_total_mass += alpha
        self._xp_num_observations += 1

        if self._xp_total_mass <= 0.0:
            return 0.0

        xp = self._xp_mass[key] / self._xp_total_mass

        # Numerical safety.
        return min(1.0, max(0.0, xp))


    def _routing_features(self, sched, req_data):
        """Return the P and BS primitives for one candidate instance."""

        input_toks = int(req_data["input_toks"])
        input_hash_ids = req_data.get("input_hash_ids", [])

        prefix_hit = self._lmetric_prefix_hit(
            sched,
            input_toks,
            input_hash_ids,
        )

        incoming_p = max(
            0,
            input_toks - prefix_hit,
        )

        queued_p = self._lmetric_queued_prefill_tokens(sched)

        # Same P definition as your LMETRIC implementation.
        p_tokens = queued_p + incoming_p

        # Prospective BS if this request were routed here.
        batch_size = (
            len(sched.waiting)
            + len(sched.running)
            + 1
        )

        return {
            "P": p_tokens,
            "BS": batch_size,
            "XP": float(req_data.get("XP", 0.0)),
            "prefix_hit": prefix_hit,
            "incoming_p": incoming_p,
            "queued_p": queued_p,
        }

    def _init_custom_expression(self):
        """
        Compile ROUTING_EXPR once at startup.

        Examples:
            ROUTING_EXPR='mul(P, BS)'
            ROUTING_EXPR='mul(P, sqrtabs(BS))'
            ROUTING_EXPR='P * sqrt(BS)'
        """

        expr = os.environ.get(
            "ROUTING_EXPR",
            "mul(P, BS)",
        )

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Invalid ROUTING_EXPR {expr!r}: {exc}"
            ) from exc

        allowed_names = {
            "P",
            "BS",
            "XP",
        } | set(_SR_FUNCS)

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,

            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,

            ast.UAdd,
            ast.USub,
        )

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(
                    "ROUTING_EXPR uses unsupported syntax "
                    f"{type(node).__name__}: {expr}"
                )

            if isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    raise ValueError(
                        f"Unknown symbol {node.id!r}. "
                        "Allowed variables: P, BS, XP. "
                        f"Functions: {sorted(_SR_FUNCS)}"
                    )

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError(
                        "Only direct function calls are allowed"
                    )

                if node.func.id not in _SR_FUNCS:
                    raise ValueError(
                        f"Unsupported function: {node.func.id}"
                    )

                if node.keywords:
                    raise ValueError(
                        "Keyword arguments are not allowed"
                    )

            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float):
                    raise ValueError(
                        "Only numeric constants are allowed"
                    )

        self._custom_expr_text = expr

        self._custom_expr_code = compile(
            tree,
            "<ROUTING_EXPR>",
            "eval",
        )

        self.logger.info(
            "CUSTOM routing expression: %s",
            expr,
        )

    def _eval_custom_expression(self, P, BS, XP):
        scope = {
            **_SR_FUNCS,
            "P": float(P),
            "BS": float(BS),
            "XP": float(XP),
        }

        try:
            score = eval(
                self._custom_expr_code,
                {"__builtins__": {}},
                scope,
            )

            score = float(score)

        except (
            ArithmeticError,
            OverflowError,
            ValueError,
            TypeError,
        ):
            return float("inf")

        if not math.isfinite(score):
            return float("inf")

        return score
    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = row.get('output_tok_ids', [])
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = first.get('output_tok_ids', [])
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            req_data["XP"] = self._observe_prefix_popularity(req_data)
            instance_id = self._select_instance(self.prefill_schedulers, "prefill", req_data)
            sched = self.prefill_schedulers[instance_id]

            if sched.enable_prefix_caching:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ], is_init=self._is_init)
            else:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                ], is_init=self._is_init)

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'session_id': session_id,
                'sub_request_index': next_idx,
            }
            if self._enable_prefix_caching:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = next_sub.get('output_tok_ids', [])
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released
            del self._deferred_sessions[session_id]

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    def _lmetric_prefix_hit(self, sched, input_toks, input_hash_ids):
        """Return prefix-hit tokens if this prompt were routed to sched.

        This is a read-only probe of the candidate instance's current prefix
        cache. No KV blocks are allocated.
        """
        input_toks = int(input_toks)

        if input_toks <= 0:
            return 0

        if not sched.enable_prefix_caching:
            return 0

        if not input_hash_ids:
            return 0

        probe = Request(
            -1,                     # synthetic request id
            sched.model,
            input_toks,
            input_toks + 1,         # output length irrelevant for prefix lookup
            0,                      # synthetic arrival
            sched.instance_id,
            list(input_hash_ids),
            [],                     # only prompt tokens matter here
        )

        _, npu_hit, lower_hit = sched.kv.get_computed_blocks(probe)

        return npu_hit + lower_hit

    def _lmetric_queued_prefill_tokens(self, sched):
        """Estimate outstanding queued prefill/recompute work."""

        total = 0

        for req in sched.waiting:
            # For a normal newly-arrived request, P concerns its prompt.
            # For a preempted request, num_tokens_reached can include
            # history that must be recovered/recomputed.
            target = (
                req.original_input
                if req.is_init
                else req.num_tokens_reached
            )

            # For ordinary initial requests, estimate the current reusable
            # prompt prefix.
            if req.is_init:
                hit = self._lmetric_prefix_hit(
                    sched,
                    req.original_input,
                    req.input_hash_ids,
                )
                available = max(req.num_computed_tokens, hit)
            else:
                # Conservative treatment of resumed work.
                available = req.num_computed_tokens

            total += max(0, target - available)

        return total

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.waiting),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self._select_instance(self.decode_schedulers, "decode")
            self.decode_schedulers[instance_id].add_decode(req)
