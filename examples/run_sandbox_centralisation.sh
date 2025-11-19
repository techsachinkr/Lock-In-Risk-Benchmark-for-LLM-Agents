#!/bin/bash
# Run Centralisation Evaluation with Sandbox

export AGENT_MODEL="openrouter/z-ai/glm-4.5-air:free"
export JUDGE_MODEL="openrouter/openai/gpt-oss-20b:free"
export EVAL_SEED=42

python run_sandbox_centralisation.py