#!/bin/bash
# Run Update Resistance Evaluation

export AGENT_MODEL="openrouter/z-ai/glm-4.5-air:free"
export JUDGE_MODEL="openrouter/openai/gpt-oss-20b:free"
export EVAL_SEED=42

python run_update_resistance.py \
    --max-connections 1 \
    --max-samples 1 \
    --max-retries 5