#!/bin/bash

# This script runs the GRPO training for the JSON formatting task
# using the Qwen/Qwen2.5-0.5B-Instruct model on CPU.

echo "Starting GRPO training for JSON formatting..."

python src/open_r1/grpo.py --config recipes/Qwen2.5-0.5B-Instruct/grpo/config_json.yaml

echo "Training finished."
