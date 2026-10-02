#!/bin/bash

ROUTING_EXPR='mul(P, BS)' \
python -m serving \
  --cluster-config configs/cluster/single_node_qwen4_instance.json \
  --dtype bfloat16 \
  --block-size 16 \
  --request-routing-policy CUSTOM \
  --dataset workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl \
  --output outputs/swebench_qwen4_CUSTOM_PxBS.csv