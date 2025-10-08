import sys
import os
import json
import asyncio
import numpy as np
from dotenv import load_dotenv
import inspect_ai._eval.eval as inspect_eval
from inspect_ai._display.rich.display import RichTaskDisplay

# Load environment variables from .env file
load_dotenv()

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
    print("\nStep 1: Creating evaluator...", flush=True)
    
    # Configure models:
    # - agent_model: The model being evaluated (responding to probes)
    # - judge_model: The model used to analyze agent responses
    agent_model = "openai/gpt-oss-20b"  # Fast model for agent responses
    judge_model = "openai/gpt-oss-120b"  # More capable model for judging
    
    evaluator = LockInEvaluator(
        model=agent_model,  # This is the model being evaluated
        log_dir="./logs",
        save_results=True
    )
    print(f"[+] Evaluator created with agent model: {agent_model}", flush=True)
    print(f"[+] Judge model: {judge_model}", flush=True)

    print("\nStep 2: Starting evaluation...", flush=True)
    try:
        # Create and run task with judge_model parameter
        results = await evaluator.run_single_focus(
            focus_area=FocusArea.UPDATE_RESISTANCE,
            sample_count=1,
            cot_style="structured",
            stealth_mode=True,
            adaptive_scoring=False,
            judge_model=judge_model  # Pass judge model for scenario creation
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
