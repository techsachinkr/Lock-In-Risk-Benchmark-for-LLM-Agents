# Quick Start: Testing All Scenarios

Get started testing scenarios in 3 simple steps using **OpenRouter API**!


```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your-openrouter-api-key"

# Linux/Mac
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Or create .env file
echo "OPENROUTER_API_KEY=your-openrouter-api-key" > .env
```



```bash
# Start with a small test (first 5 text scenarios)
python test_all_scenarios.py --variant-types text --limit 5
```

### Check Results

```bash
# View the summary
cat test_results/scenario_test_*_summary.json
```



### Run Tests with OpenRouter

```bash
# Run all text scenarios with default model (Gemini 2.5 Pro)
python test_all_scenarios.py --variant-types text

# Run with Claude
python test_all_scenarios.py --agent-model anthropic/claude-3.5-sonnet --limit 5

# Run with GPT-4o
python test_all_scenarios.py --agent-model openai/gpt-4o --limit 5

# Run with Llama
python test_all_scenarios.py --agent-model meta-llama/llama-3.1-70b-instruct --limit 5

# Run specific category
python test_all_scenarios.py --categories update_resistance

# Run specific difficulty
python test_all_scenarios.py --difficulties basic
```


### Filter and Export

```bash


# Test exported scenarios
python test_all_scenarios.py --scenarios-file my_scenarios.json
```

## Interactive Examples

Run the examples script to see demonstrations:

```bash
python test_scenario_examples.py
```

## What You Get

After running tests, you'll find:

- `test_results/scenario_test_*_results.json` - Detailed results for each scenario
- `test_results/scenario_test_*_summary.json` - Summary statistics and high-risk scenarios
- `logs/scenario_tests/` - Detailed evaluation logs

## Environment Setup

### Required

Set your API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Optional (for sandbox scenarios)

Install Docker:
- Windows/Mac: https://docs.docker.com/desktop/
- Linux: https://docs.docker.com/engine/install/

## Scenario Types

| Type | Description | Setup Required | Time per Scenario |
|------|-------------|----------------|-------------------|
| **text** | Pure conversation | None | ~5-10 min |
| **tools** | With specific tools | Tool configuration | ~10-15 min |
| **sandbox** | Full environment | Docker | ~15-20 min |

**Recommendation:** Start with `text` scenarios - they're fastest and need no special setup.

## Available Filters

### By Variant Type
```bash
--variant-types text sandbox tools
```

### By Category
```bash
--categories update_resistance centralisation commitment_stickiness enclosure_dependency oversight_avoidance
```

### By Difficulty
```bash
--difficulties basic intermediate advanced
```

### By Limit
```bash
--limit 10  # Run only first 10 scenarios
```

## Example Workflows

### Quick Test (5 minutes)
```bash

# Run 5 scenarios
python test_all_scenarios.py --variant-types text --limit 5
```

### Category-Specific Test (30 minutes)
```bash
# Test all update_resistance scenarios
python test_all_scenarios.py \
  --categories update_resistance \
  --variant-types text
```

### Comprehensive Test (several hours)
```bash
# Test all text scenarios
python test_all_scenarios.py --variant-types text

# If Docker available, test sandbox scenarios
python test_all_scenarios.py --variant-types sandbox --include-sandbox

# Test tools scenarios
python test_all_scenarios.py --variant-types tools --include-tools
```


