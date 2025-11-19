import sys
import os
import json
import asyncio
import argparse
import numpy as np
from dotenv import load_dotenv
import inspect_ai._eval.eval as inspect_eval
from inspect_ai._display.rich.display import RichTaskDisplay
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
# Explicitly load from project root to ensure consistency
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Custom JSON encoder to handle NumPy types and Pydantic models
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'model_dump'):
            # Handle Pydantic v2 models (like Score)
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            # Handle Pydantic v1 models
            return obj.dict()
        return super().default(obj)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Initialize logging first
import logging
logging.basicConfig(level=logging.CRITICAL)

# Create a logger for concurrent.futures
logger = logging.getLogger("concurrent.futures")
logger.setLevel(logging.CRITICAL)

# Import evaluation module
from evaluation.inspect_integration import LockInEvaluator
from evaluation.core.enums import FocusArea

# Custom evaluation function
async def run_eval(task, model, limit=None, log_dir=None):
    """Run evaluation with the given task and model.
    
    Args:
        task: Task object with dataset, solver, and scorer
        model: Model identifier string
        limit: Maximum number of samples to evaluate
        log_dir: Directory to save results
    """
    
    # Ensure scorer is properly instantiated
    from evaluation.scorers import UnifiedLockInScorer
    if not hasattr(task, 'scorer') or not isinstance(task.scorer, UnifiedLockInScorer):
        print("Creating new UnifiedLockInScorer instance...")
        task.scorer = UnifiedLockInScorer(strict_mode=True)
    from evaluation.core.config import Config
    from evaluation.core.model import Model
    from evaluation.core.task_state import TaskState

    # Initialize model
    config = Config({
        "model": model,
        "temperature": 0.7,
        "max_tokens": 1000
    })
    model_instance = Model(model, config)

    # Initialize task state
    from evaluation.core.message import Message
    state = TaskState(
        model=model,
        sample_id="0",
        epoch=0,
        input="",
        messages=[],
        metadata={}
    )

    # Run task
    results = []
    for i in range(limit or 1):
        # Get solver for this sample
        solver = task.solver(task.dataset[i])
        
        # Run solver
        state = await solver(state, model_instance.generate)
        
        # Score results
        try:
            if hasattr(task, 'scorer') and task.scorer:
                if callable(task.scorer):
                    score = await task.scorer(state, task.dataset[i])
                else:
                    print(f"Warning: task.scorer exists but is not callable: {type(task.scorer)}")
                    score = {
                        "value": 0.0,
                        "explanation": f"Scorer exists but not callable: {type(task.scorer)}"
                    }
            else:
                print("Warning: task.scorer is missing")
                score = {
                    "value": 0.0,
                    "explanation": "No scorer available on task"
                }
        except Exception as e:
            print(f"Error during scoring: {str(e)}")
            score = {
                "value": 0.0,
                "explanation": f"Scoring error: {str(e)}"
            }
        
        # Store results
        result = {
            "score": score,
            "metadata": state.metadata
        }
        results.append(result)
        
        # Save individual result
        if log_dir:
            import json
            from datetime import datetime
            from pathlib import Path
            
            # Ensure logs/results directory exists
            output_dir = Path(log_dir) / "results"
            output_dir.mkdir(parents=True, exist_ok=True)  # This will create both logs/ and logs/results/ if they don't exist
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"lock_in_eval_{timestamp}.json"
            
            # Save result
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            
            # Generate and save analysis
            from evaluation.analysis.result_analyzer import ResultAnalyzer
            # Convert result to dict if it's a Pydantic model
            result_dict = result.copy()
            if hasattr(result_dict.get('score'), 'model_dump'):
                result_dict['score'] = result_dict['score'].model_dump()
            analyzer = ResultAnalyzer(result_dict)
            
            # Create analysis filename with same timestamp
            analysis_filename = output_dir / f"lock_in_eval_{timestamp}_analysis.json"
            analyzer.save_analysis(str(analysis_filename))
            
            # Print analysis to console
            print("\nEvaluation Analysis:")
            print(analyzer.get_detailed_analysis())

    return results

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Lock-In Risk Benchmark evaluation with configurable models and parameters'
    )
    parser.add_argument(
        '--max-connections',
        type=int,
        default=10,
        help='Maximum parallel connections for API calls. Lower values help avoid rate limits. Default: 10'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1,
        help='Maximum number of samples to evaluate. Default: 1'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retries for failed API calls. Default: 3'
    )
    
    args = parser.parse_args()
    
    # Get models and seed from environment variables
    agent_model = os.getenv("AGENT_MODEL", "openrouter/z-ai/glm-4.5-air:free")
    judge_model = os.getenv("JUDGE_MODEL", "openrouter/openai/gpt-oss-20b:free")
    seed = os.getenv("EVAL_SEED")
    seed = int(seed) if seed else None
    
    print("\n" + "="*60)
    print("Lock-In Risk Benchmark Evaluation")
    print("="*60)
    print(f"Agent Model: {agent_model}")
    print(f"Judge Model: {judge_model}")
    print(f"Random Seed: {seed if seed is not None else 'None (random)'}")
    print(f"Max Connections: {args.max_connections}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Max Retries: {args.max_retries}")
    print("="*60 + "\n")
    
    print("Step 1: Creating evaluator...", flush=True)
    
    # Use OpenRouter only
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        # Set OPENAI_API_KEY for inspect_ai compatibility
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("Warning: No OPENROUTER_API_KEY found")
        print("Set one with: export OPENROUTER_API_KEY='your-key'\n")
    
    # Set OpenRouter base URL (hardcoded since we only use OpenRouter)
    # This environment variable is picked up by the OpenAI client used by inspect_ai
    model_base_url = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_BASE_URL"] = model_base_url
    
    print(f"API Configuration:")
    print(f"  Base URL: {model_base_url}")
    print(f"  Agent Model: {agent_model}")
    print(f"  Judge Model: {judge_model}\n")
    
    # Always use fixed project-root-relative log directory for update resistance
    project_root = Path(__file__).parent.parent.resolve()
    log_dir = project_root / "logs" / "update_resistance"
    log_dir.mkdir(parents=True, exist_ok=True)
    evaluator = LockInEvaluator(
        model=agent_model,  # This is the model being evaluated
        log_dir=str(log_dir),
        save_results=True
    )
    print(f"[+] Evaluator created with agent model: {agent_model}", flush=True)
    print(f"[+] Judge model: {judge_model}", flush=True)

    print("\nStep 2: Starting evaluation...", flush=True)
    try:
        # Only pass valid arguments to the evaluation task
        task_kwargs = {
            "cot_style": "structured",
            "stealth_mode": True,
            "adaptive_scoring": False,
            "judge_model": judge_model,
            "seed": seed
        }
        results = await evaluator.run_single_focus(
            focus_area=FocusArea.UPDATE_RESISTANCE,
            sample_count=args.max_samples,
            **task_kwargs
        )
        print("[+] Evaluation completed successfully", flush=True)
        print("\nResults:", flush=True)
        print(results, flush=True)
        return results
    except Exception as e:
        print(f"\n[-] Error: {str(e)}", flush=True)
        raise

if __name__ == "__main__":
    import asyncio
    print("Starting evaluation...", flush=True)
    try:
        results = asyncio.run(main())
        print("\nFinal results:", flush=True)
        print(results, flush=True)
    except Exception as e:
        print(f"\nError occurred: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
