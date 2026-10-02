#!/usr/bin/env python3
"""
Genetic-programming search for cache-aware LLMServingSim request-routing scores.

The router must accept the environment variable ROUTING_EXPR and evaluate it
for each candidate replica with terminals P and BS, choosing argmin(score).

Example:
    python scripts/search_router_gp.py \
      --cluster-config configs/cluster/single_node_qwen4_instance.json \
      --dataset workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl \
      --num-reqs 10 \
      --population 8 \
      --generations 4 \
      --workers 8 \
      --timeout 600 \
      --objective balanced
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from deap import algorithms, base, creator, gp, tools


EPS = 1e-12


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def pdiv(a, b):
    return a / b if abs(b) > EPS else 1.0


def sqrtabs(x):
    return math.sqrt(abs(x))


def logabs(x):
    return math.log1p(abs(x))


def square(x):
    return x * x


def absval(x):
    return abs(x)


class SimulatorEvaluator:
    def __init__(self, args):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.workdir = (self.repo_root / args.workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Cache fingerprints are computed once per search. This makes cache keys
        # change when the cluster config or workload contents change, even if
        # their filenames stay the same.
        self.cluster_config_path = (self.repo_root / args.cluster_config).resolve()
        self.dataset_path = (self.repo_root / args.dataset).resolve()
        self.cluster_config_sha256 = self._file_fingerprint(self.cluster_config_path)
        self.dataset_sha256 = self._file_fingerprint(self.dataset_path)

        self.cache_path = self.workdir / "evaluations.jsonl"
        self.cache = {}
        self._cache_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._load_cache()

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        """Hash file contents so cache entries follow config/workload changes."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _signature(self, expr: str) -> str:
        payload = {
            "expr": expr,
            "cluster_config": str(self.cluster_config_path),
            "cluster_config_sha256": self.cluster_config_sha256,
            "dataset": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "dtype": self.args.dtype,
            "block_size": self.args.block_size,
            "num_reqs": self.args.num_reqs,
        }
        return json.dumps(payload, sort_keys=True)

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    self.cache[record["signature"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue

    def _append_cache(self, record):
        # Multiple simulator evaluations may finish at the same time.
        # Protect both the in-memory cache and JSONL append.
        with self._cache_lock:
            self.cache[record["signature"]] = record
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def _print(self, *args, **kwargs):
        # Keep output from concurrent evaluations readable.
        kwargs.setdefault("flush", True)
        with self._print_lock:
            print(*args, **kwargs)

    @staticmethod
    def _stop_process_group(proc: subprocess.Popen, grace_s: float = 5.0):
        """Stop a simulator and any children it may have spawned."""
        if proc.poll() is not None:
            return

        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            proc.terminate()

        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass

        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            proc.kill()

        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass

    def _penalty_metrics(self, expr: str, wall_s: float, status: str, error: str):
        return {
            "status": status,
            "expression": expr,
            "sim_wall_s": wall_s,
            "requests": 0,
            "mean_ttft_ms": self.args.failure_penalty,
            "median_ttft_ms": self.args.failure_penalty,
            "p99_ttft_ms": self.args.failure_penalty,
            "mean_tpot_ms": self.args.failure_penalty,
            "median_tpot_ms": self.args.failure_penalty,
            "p99_tpot_ms": self.args.failure_penalty,
            "mean_request_latency_ms": self.args.failure_penalty,
            "total_latency_s": self.args.failure_penalty,
            "prefix_hit_ratio_pct": 0.0,
            "error": error,
        }

    @staticmethod
    def _read_request_csv(path: Path):
        ttft_ms = []
        tpot_ms = []
        latency_ms = []

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ttft_ns = float(row["TTFT"])
                tpot_ns = float(row["TPOT"])
                latency_ns = float(row["latency"])

                if ttft_ns >= 0:
                    ttft_ms.append(ttft_ns / 1_000_000.0)
                if tpot_ns >= 0:
                    tpot_ms.append(tpot_ns / 1_000_000.0)
                if latency_ns >= 0:
                    latency_ms.append(latency_ns / 1_000_000.0)

        if not ttft_ms or not tpot_ms:
            raise RuntimeError(f"No valid TTFT/TPOT rows found in {path}")

        return {
            "requests": len(ttft_ms),
            "mean_ttft_ms": float(np.mean(ttft_ms)),
            "median_ttft_ms": float(np.median(ttft_ms)),
            "p99_ttft_ms": float(np.percentile(ttft_ms, 99)),
            "mean_tpot_ms": float(np.mean(tpot_ms)),
            "median_tpot_ms": float(np.median(tpot_ms)),
            "p99_tpot_ms": float(np.percentile(tpot_ms, 99)),
            "mean_request_latency_ms": float(np.mean(latency_ms)) if latency_ms else math.nan,
        }

    def run_expression(self, expr: str):
        signature = self._signature(expr)
        with self._cache_lock:
            cached = self.cache.get(signature)
        if cached is not None:
            return cached

        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        output_csv = self.workdir / f"candidate_{digest}.csv"
        fail_log = self.workdir / f"candidate_{digest}.failed.log"

        if output_csv.exists():
            output_csv.unlink()

        env = os.environ.copy()
        env["ROUTING_EXPR"] = expr

        # Each candidate is already a separate OS process. Keep numerical
        # libraries inside that simulator from spawning extra thread pools and
        # oversubscribing the machine when many candidates run concurrently.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

        run_id = f"gp_{digest}_{os.getpid()}_{int(time.time() * 1000)}"

        cmd = [
            sys.executable,
            "-m",
            "serving",
            "--cluster-config",
            self.args.cluster_config,
            "--dtype",
            self.args.dtype,
            "--block-size",
            str(self.args.block_size),
            "--request-routing-policy",
            "CUSTOM",
            "--dataset",
            self.args.dataset,
            "--output",
            str(output_csv),
            "--run-id",
            run_id,
            "--log-level",
            "WARNING",
            "--log-interval",
            "1000000",
        ]

        if self.args.num_reqs > 0:
            cmd.extend(["--num-reqs", str(self.args.num_reqs)])

        start = time.monotonic()
        proc = None

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
            self._print(f"[START] pid={proc.pid:<7} expr={expr}")

            try:
                stdout, stderr = proc.communicate(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                wall_s = time.monotonic() - start
                self._stop_process_group(proc)
                # Drain any remaining captured output after termination.
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""

                error = (
                    f"Timed out after {self.args.timeout}s.\n"
                    f"STDOUT:\n{stdout or ''}\nSTDERR:\n{stderr or ''}"
                )
                fail_log.write_text(error, encoding="utf-8")
                metrics = self._penalty_metrics(
                    expr, wall_s, "timeout", error
                )
                self._print(
                    f"[TIMEOUT] pid={proc.pid:<7} wall={wall_s:8.2f}s "
                    f"expr={expr}\n"
                    f"          penalty={self.args.failure_penalty:g}; "
                    f"details={fail_log.name}"
                )
            else:
                wall_s = time.monotonic() - start

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Simulator returned {proc.returncode}\n"
                        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                    )

                metrics = self._read_request_csv(output_csv)

                total_latency_match = re.search(
                    r"Total latency \(s\):\s+([0-9.]+)", stdout
                )
                prefix_match = re.search(
                    r"Total prefix hit ratio \(%\):\s+([0-9.]+)", stdout
                )

                metrics.update(
                    {
                        "status": "ok",
                        "expression": expr,
                        "sim_wall_s": wall_s,
                        "total_latency_s": (
                            float(total_latency_match.group(1))
                            if total_latency_match
                            else math.nan
                        ),
                        "prefix_hit_ratio_pct": (
                            float(prefix_match.group(1))
                            if prefix_match
                            else math.nan
                        ),
                    }
                )

                self._print(
                    f"[DONE ] pid={proc.pid:<7} wall={wall_s:8.2f}s expr={expr}\n"
                    f"       TTFT={metrics['mean_ttft_ms']:.3f} ms, "
                    f"P99-TTFT={metrics['p99_ttft_ms']:.3f} ms, "
                    f"TPOT={metrics['mean_tpot_ms']:.3f} ms, "
                    f"prefix={metrics['prefix_hit_ratio_pct']:.2f}%"
                )

                if not self.args.keep_csv:
                    output_csv.unlink(missing_ok=True)

        except Exception as exc:
            wall_s = time.monotonic() - start
            if proc is not None and proc.poll() is None:
                self._stop_process_group(proc)

            error = str(exc)
            fail_log.write_text(error, encoding="utf-8")
            metrics = self._penalty_metrics(expr, wall_s, "failed", error)
            pid_text = str(proc.pid) if proc is not None else "n/a"
            self._print(
                f"[FAIL ] pid={pid_text:<7} wall={wall_s:8.2f}s expr={expr}\n"
                f"        penalty={self.args.failure_penalty:g}; "
                f"details={fail_log.name}"
            )

        if metrics["status"] != "ok" and not self.args.keep_csv:
            output_csv.unlink(missing_ok=True)

        record = {
            "signature": signature,
            **metrics,
        }
        self._append_cache(record)
        return record


def make_pset():
    pset = gp.PrimitiveSet("ROUTER_SCORE", 2)
    pset.renameArguments(ARG0="P", ARG1="BS")

    pset.addPrimitive(add, 2, name="add")
    pset.addPrimitive(sub, 2, name="sub")
    pset.addPrimitive(mul, 2, name="mul")
    pset.addPrimitive(pdiv, 2, name="pdiv")

    pset.addPrimitive(sqrtabs, 1, name="sqrtabs")
    pset.addPrimitive(logabs, 1, name="logabs")
    pset.addPrimitive(square, 1, name="square")
    pset.addPrimitive(absval, 1, name="absval")

    pset.addEphemeralConstant(
        "c",
        functools.partial(random.uniform, -2.0, 2.0),
    )
    return pset


def metric_to_fitness(metrics, baseline, objective):
    if metrics["status"] != "ok":
        return float(metrics["mean_ttft_ms"])

    if objective == "ttft":
        return metrics["mean_ttft_ms"]

    if objective == "p99_ttft":
        return metrics["p99_ttft_ms"]

    if objective == "balanced":
        ttft_ratio = metrics["mean_ttft_ms"] / baseline["mean_ttft_ms"]
        tpot_ratio = metrics["mean_tpot_ms"] / baseline["mean_tpot_ms"]
        return ttft_ratio * tpot_ratio

    raise ValueError(f"Unknown objective: {objective}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--cluster-config",
        default="configs/cluster/single_node_qwen4_instance.json",
    )
    parser.add_argument(
        "--dataset",
        default="workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-reqs", type=int, default=10)

    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--cxpb", type=float, default=0.6)
    parser.add_argument("--mutpb", type=float, default=0.4)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--max-height", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Number of candidate LLMServingSim processes to evaluate in "
            "parallel (default: min(8, logical CPU count))"
        ),
    )

    parser.add_argument(
        "--objective",
        choices=["ttft", "p99_ttft", "balanced"],
        default="balanced",
        help=(
            "balanced minimizes "
            "(mean_TTFT / LMETRIC_mean_TTFT) * "
            "(mean_TPOT / LMETRIC_mean_TPOT)"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=(
            "Per-candidate wall-clock timeout in seconds. A timeout is a search "
            "guard, not simulated time (default: 600)."
        ),
    )
    parser.add_argument("--failure-penalty", type=float, default=1e9)
    parser.add_argument("--workdir", default="outputs/gp_router_search")
    parser.add_argument("--keep-csv", action="store_true")
    args = parser.parse_args()

    if args.elite < 0 or args.elite >= args.population:
        parser.error("--elite must be >= 0 and < --population")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1 second")

    random.seed(args.seed)
    np.random.seed(args.seed)

    evaluator = SimulatorEvaluator(args)

    lmetric_expr = "mul(P, BS)"
    baseline = evaluator.run_expression(lmetric_expr)
    if baseline["status"] != "ok":
        raise RuntimeError("LMETRIC baseline failed; inspect the .failed.log file")

    print(
        "\nLMETRIC-via-CUSTOM baseline: "
        f"mean TTFT={baseline['mean_ttft_ms']:.3f} ms, "
        f"mean TPOT={baseline['mean_tpot_ms']:.3f} ms"
    )

    pset = make_pset()

    if not hasattr(creator, "RoutingFitnessMin"):
        creator.create("RoutingFitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "RoutingIndividual"):
        creator.create(
            "RoutingIndividual",
            gp.PrimitiveTree,
            fitness=creator.RoutingFitnessMin,
        )

    toolbox = base.Toolbox()
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=1,
        max_=3,
    )
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.RoutingIndividual,
        toolbox.expr,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_individual(individual):
        expr = str(individual)
        metrics = evaluator.run_expression(expr)
        fitness = metric_to_fitness(metrics, baseline, args.objective)
        return (fitness,)

    def evaluate_invalid(individuals):
        """Evaluate invalid individuals, running unique expressions in parallel.

        DEAP can produce the same expression more than once. Group duplicates so
        only one simulator process is launched for each unique expression; copy
        the resulting fitness to every equivalent individual.
        """
        invalid = [ind for ind in individuals if not ind.fitness.valid]
        if not invalid:
            return

        groups = {}
        for ind in invalid:
            groups.setdefault(str(ind), []).append(ind)

        expressions = list(groups)
        max_workers = min(args.workers, len(expressions))
        batch_start = time.monotonic()
        print(
            f"\n[BATCH] {len(expressions)} unique candidate(s), "
            f"up to {max_workers} parallel simulator(s)",
            flush=True,
        )

        def evaluate_expr(expr):
            metrics = evaluator.run_expression(expr)
            fitness = metric_to_fitness(metrics, baseline, args.objective)
            return expr, fitness

        if args.workers == 1 or len(expressions) == 1:
            results = [evaluate_expr(expr) for expr in expressions]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(evaluate_expr, expr): expr
                    for expr in expressions
                }
                for future in as_completed(futures):
                    results.append(future.result())

        for expr, fitness in results:
            for ind in groups[expr]:
                ind.fitness.values = (fitness,)

        batch_wall_s = time.monotonic() - batch_start
        print(
            f"[BATCH DONE] {len(expressions)} unique candidate(s) "
            f"in {batch_wall_s:.2f}s",
            flush=True,
        )

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=args.tournament_size,
    )
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

    toolbox.decorate(
        "mate",
        gp.staticLimit(
            key=operator.attrgetter("height"),
            max_value=args.max_height,
        ),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(
            key=operator.attrgetter("height"),
            max_value=args.max_height,
        ),
    )
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=len, max_value=args.max_nodes),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=len, max_value=args.max_nodes),
    )

    seed_exprs = [
        "mul(P, BS)",
        "P",
        "BS",
        "mul(P, sqrtabs(BS))",
        "mul(sqrtabs(P), BS)",
        "add(P, BS)",
        "pdiv(P, BS)",
        "mul(P, square(BS))",
    ]

    population = []
    for expr in seed_exprs[: args.population]:
        tree = gp.PrimitiveTree.from_string(expr, pset)
        population.append(creator.RoutingIndividual(tree))

    if len(population) < args.population:
        population.extend(
            toolbox.population(n=args.population - len(population))
        )

    hall = tools.HallOfFame(maxsize=20)

    evaluate_invalid(population)
    hall.update(population)

    def print_generation(gen, pop):
        best = tools.selBest(pop, 1)[0]
        metrics = evaluator.run_expression(str(best))
        print(
            f"\n=== Generation {gen} ===\n"
            f"best expr: {best}\n"
            f"fitness:   {best.fitness.values[0]:.6f}\n"
            f"TTFT:      {metrics['mean_ttft_ms']:.3f} ms\n"
            f"P99 TTFT:  {metrics['p99_ttft_ms']:.3f} ms\n"
            f"TPOT:      {metrics['mean_tpot_ms']:.3f} ms\n"
            f"prefix:    {metrics['prefix_hit_ratio_pct']:.2f}%\n"
            f"nodes:     {len(best)}, height={best.height}"
        )

    print_generation(0, population)

    for gen in range(1, args.generations + 1):
        elites = list(map(toolbox.clone, tools.selBest(population, args.elite)))

        parents = toolbox.select(
            population,
            args.population - args.elite,
        )
        parents = list(map(toolbox.clone, parents))

        offspring = algorithms.varAnd(
            parents,
            toolbox,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
        )

        evaluate_invalid(offspring)

        population = elites + offspring
        hall.update(population)
        print_generation(gen, population)

    ranked = sorted(hall, key=lambda ind: ind.fitness.values[0])

    summary = []
    print("\n\n================ FINAL HALL OF FAME ================")
    for rank, ind in enumerate(ranked[:10], start=1):
        metrics = evaluator.run_expression(str(ind))
        record = {
            "rank": rank,
            "expression": str(ind),
            "fitness": ind.fitness.values[0],
            "nodes": len(ind),
            "height": ind.height,
            "mean_ttft_ms": metrics["mean_ttft_ms"],
            "p99_ttft_ms": metrics["p99_ttft_ms"],
            "mean_tpot_ms": metrics["mean_tpot_ms"],
            "p99_tpot_ms": metrics["p99_tpot_ms"],
            "prefix_hit_ratio_pct": metrics["prefix_hit_ratio_pct"],
            "total_latency_s": metrics["total_latency_s"],
        }
        summary.append(record)

        print(
            f"{rank:2d}. fitness={record['fitness']:.6f} "
            f"TTFT={record['mean_ttft_ms']:.3f} "
            f"TPOT={record['mean_tpot_ms']:.3f} "
            f"nodes={record['nodes']:2d} :: {record['expression']}"
        )

    summary_path = evaluator.workdir / "hall_of_fame.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    best_expr_path = evaluator.workdir / "best_expression.txt"
    best_expr_path.write_text(str(ranked[0]) + "\n", encoding="utf-8")

    print(f"\nSaved hall of fame: {summary_path}")
    print(f"Saved best expression: {best_expr_path}")
    print(
        "\nFull-trace validation:\n"
        f"  ROUTING_EXPR='{ranked[0]}' python -m serving \\\n"
        f"    --cluster-config {args.cluster_config} \\\n"
        f"    --dtype {args.dtype} --block-size {args.block_size} \\\n"
        f"    --request-routing-policy CUSTOM \\\n"
        f"    --dataset {args.dataset} \\\n"
        f"    --output outputs/swebench_qwen4_SYMBOLIC.csv"
    )


if __name__ == "__main__":
    main()
