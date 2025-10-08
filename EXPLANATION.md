# Model Usage Explanation

This document verifies that the agent model generates probe responses and the judge model evaluates those responses.

## Flow Diagram

```
User Configuration (run_evaluation.py)
    ↓
agent_model = "openai/gpt-oss-20b"     ← Model being evaluated
judge_model = "openai/gpt-oss-120b"    ← Model for judging
    ↓
    ├─→ Agent Model Path (Generates Responses)
    │   ├─ LockInEvaluator(model=agent_model) 
    │   ├─ Model(agent_model, config)
    │   ├─ model_instance.generate() method
    │   ├─ EnhancedLockInSolver.__call__(state, generate)
    │   │   └─ Line 228: state = await self._generate_with_cot(state, generate, f"probe_{i}")
    │   │       └─ Line 277: return await generate(state)  ← Uses agent model
    │   └─ Agent responds to probes using agent_model
    │
    └─→ Judge Model Path (Evaluates Responses)
        ├─ run_single_focus(judge_model=judge_model)
        ├─ LockInDatasetFactory.create_dataset(judge_model=judge_model)
        ├─ scenario_config["judge_model"] = judge_model
        ├─ UpdateResistanceScenarioWithJudge(__init__, judge_model=judge_model)
        ├─ LLMJudge(model_name=judge_model)
        │   └─ Line 53: self.client = AsyncOpenAI(...)
        ├─ Line 237: analysis = await self.scenario.analyze_response_with_judge(response, i, generate)
        └─ LLMJudge._call_judge_model(prompt)
            └─ Line 64: response = await self.client.chat.completions.create(model=self.model_name, ...)
                └─ Uses judge_model to analyze agent responses
```

## Code Evidence

### 1. Agent Model Generates Responses

**File:** `run_evaluation.py`
```python
# Line 69-74: Agent model is configured
config = Config({
    "model": model,  # This is agent_model
    "temperature": 0.7,
    "max_tokens": 1000
})
model_instance = Model(model, config)

# Line 94: Agent model's generate function is passed to solver
state = await solver(state, model_instance.generate)
```

**File:** `evaluation/core/model.py`
```python
# Line 22-23: Model stores the agent model name
def __init__(self, model: str, config: Optional[Config] = None):
    self.model = model  # This is agent_model

# Line 45-47: The generate method uses self.model (agent model)
async def generate(self, state: Any) -> Any:
    # ...
    response = await self.client.chat.completions.create(
        model=self.model,  # ← Agent model used here
        messages=messages
    )
```

**File:** `evaluation/inspect_integration.py`
```python
# Line 228: Solver calls generate (which uses agent model)
state = await self._generate_with_cot(state, generate, f"probe_{i}")

# Line 277: _generate_with_cot calls generate function
return await generate(state)  # ← This uses agent model
```

### 2. Judge Model Evaluates Responses

**File:** `run_evaluation.py`
```python
# Line 169-170: Judge model is configured separately
agent_model = "openai/gpt-oss-20b"     # Agent
judge_model = "openai/gpt-oss-120b"    # Judge

# Line 187: Judge model passed to evaluation
results = await evaluator.run_single_focus(
    focus_area=FocusArea.UPDATE_RESISTANCE,
    sample_count=1,
    judge_model=judge_model,  # ← Judge model passed here
    ...
)
```

**File:** `evaluation/inspect_integration.py`
```python
# Line 436: Judge model passed to dataset creation
dataset = LockInDatasetFactory.create_dataset(
    focus_areas=focus_areas,
    samples_per_area=samples_per_area,
    randomize=randomize_order,
    seed=seed,
    judge_model=judge_model  # ← Flows to scenario config
)

# Line 88: Judge model added to scenario config
"scenario_config": LockInDatasetFactory._get_scenario_config(focus_area, i, judge_model)

# Line 104: Judge model set in config
configs = {
    FocusArea.UPDATE_RESISTANCE: {
        "update_type": [...][index % 3],
        "enable_cot_analysis": True,
        "judge_model": judge_model  # ← Stored in config
    }
}

# Line 237: Scenario analyzes response with judge
analysis = await self.scenario.analyze_response_with_judge(response, i, generate)
```

**File:** `evaluation/scenarios/update_resistance.py`
```python
# Line 472-481: Judge model passed to LLMJudge
def __init__(
    self,
    scenario_id: str,
    update_type: str = "goal_change",
    enable_cot_analysis: bool = True,
    judge_model: str = "openai/gpt-oss-20b"
):
    # ...
    # Initialize LLM Judge
    self.judge = LLMJudge(model_name=judge_model)  # ← Judge uses judge_model

# Line 44-53: LLMJudge stores judge model
def __init__(self, model_name: str = "openai/gpt-oss-20b"):
    self.model_name = model_name  # ← This is judge_model
    # ...
    self.client = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

# Line 64-65: Judge model used for API calls
response = await self.client.chat.completions.create(
    model=self.model_name,  # ← Judge model used here
    messages=[...]
)
```

## Summary

### ✅ Agent Model Usage (openai/gpt-oss-20b)

1. **Configuration**: Set in `run_evaluation.py` line 169
2. **Storage**: Stored in `Model` class in `evaluation/core/model.py` line 23
3. **Usage**: Used in `model.generate()` method at line 47
4. **Called By**: `EnhancedLockInSolver` at line 228 via `_generate_with_cot`
5. **Purpose**: Generates responses to evaluation probes

### ✅ Judge Model Usage (openai/gpt-oss-120b)

1. **Configuration**: Set in `run_evaluation.py` line 170
2. **Passed To**: `run_single_focus(judge_model=...)` at line 187
3. **Flows Through**: 
   - `comprehensive_lock_in_task` (line 407)
   - `LockInDatasetFactory.create_dataset` (line 436)
   - `_get_scenario_config` (line 104)
   - `UpdateResistanceScenarioWithJudge.__init__` (line 481)
4. **Storage**: Stored in `LLMJudge` class at line 45
5. **Usage**: Used in `LLMJudge._call_judge_model()` at line 65
6. **Called By**: All judge methods (analyze_update_response, detect_resistance_patterns, etc.)
7. **Purpose**: Analyzes and scores agent responses

## Key Distinction

- **Agent Model (`generate` function)**: Called by solver to produce responses to probes
- **Judge Model (`_call_judge_model` function)**: Called by scenario to analyze those responses

These are completely separate:
- Different model names
- Different API client instances
- Different code paths
- Different purposes

## Testing Verification

To verify this is working correctly, check the logs for:

1. **Agent Model Calls**: Look for API calls to `openai/gpt-oss-20b`
2. **Judge Model Calls**: Look for API calls to `openai/gpt-oss-120b`
3. **Console Output**: Should show both model names:
   ```
   [+] Evaluator created with agent model: openai/gpt-oss-20b
   [+] Judge model: openai/gpt-oss-120b
   ```

## Conclusion

- **Agent model** (openai/gpt-oss-20b) to generate probe responses
- **Judge model** (openai/gpt-oss-120b) to evaluate those responses

The separation is complete and both models serve distinct purposes in the evaluation pipeline.
