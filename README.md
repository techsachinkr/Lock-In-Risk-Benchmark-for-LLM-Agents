# Lock-In-Risk-Benchmark-for-LLM-Agents

## Environment Setup

### Conda Environment

Create a new conda environment:

```bash
conda create -n project_env python=3.9
conda activate project_env
```

Install requirements:

```bash
pip install -r requirements.txt
```

### RunPod Storage

Place all work files in `/workspace` directory for persistent storage across container restarts.

## API Key Configuration

Create a `.env` file in the project root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Add other API keys as needed
```

The `.env` file is already included in `.gitignore` to prevent accidental commits to the repository.
