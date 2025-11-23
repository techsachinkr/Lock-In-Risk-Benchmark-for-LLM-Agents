"""
Comprehensive Lock-In Benchmark Runner
Executes all lock-in scenarios across all focus areas with configurable options.
"""

import json
import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.core.enums import FocusArea
from evaluation.inspect_integration import LockInEvaluator


class ComprehensiveBenchmarkRunner:
    """
    Runner for comprehensive lock-in benchmarks across all scenarios
    """

    def __init__(
        self,
        agent_model: str = "openai/gpt-oss-20b",
        judge_model: str = "openai/gpt-oss-120b",
        output_dir: str = "./benchmark_results",
        scenarios_file: Optional[str] = None
    ):
        self.agent_model = agent_model
        self.judge_model = judge_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load scenarios
        if scenarios_file:
            with open(scenarios_file, 'r') as f:
                self.scenarios = json.load(f)
        else:
            # Use default generated scenarios
            default_path = project_root / "benchmarks" / "generated" / "lock_in_scenarios.json"
            if default_path.exists():
                with open(default_path, 'r') as f:
                    self.scenarios = json.load(f)
            else:
                self.scenarios = []

        self.results = []
        self.evaluator = None

    async def initialize_evaluator(self):
        """Initialize the evaluator"""
        self.evaluator = LockInEvaluator(
            model=self.agent_model,
            log_dir=str(self.output_dir / "logs"),
            save_results=True
        )

    async def run_single_scenario(
        self,
        scenario: Dict[str, Any],
        run_id: str
    ) -> Dict[str, Any]:
        """Run a single benchmark scenario"""

        print(f"\n{'='*70}")
        print(f"Running: {scenario['id']} - {scenario['subcategory']}")
        print(f"Category: {scenario['category']}")
        print(f"Difficulty: {scenario['difficulty']}")
        print(f"{'='*70}\n")

        # Map scenario category to FocusArea
        category_map = {
            "update_resistance": FocusArea.UPDATE_RESISTANCE,
            "centralisation": FocusArea.CENTRALISATION,
            "commitment_stickiness": FocusArea.COMMITMENT_STICKINESS,
            "enclosure_dependency": FocusArea.ENCLOSURE_DEPENDENCY,
            "oversight_avoidance": FocusArea.OVERSIGHT_AVOIDANCE
        }

        focus_area = category_map.get(scenario['category'])
        if not focus_area:
            print(f"Warning: Unknown category {scenario['category']}, skipping...")
            return None

        try:
            # For update_resistance, we can use the existing implementation
            if focus_area == FocusArea.UPDATE_RESISTANCE:
                # Map subcategory to update_type
                subcategory = scenario['subcategory']
                update_type = subcategory  # Use subcategory as update type

                result = await self.evaluator.run_single_focus(
                    focus_area=focus_area,
                    sample_count=1,
                    judge_model=self.judge_model,
                    cot_style="structured",
                    stealth_mode=False,
                    adaptive_scoring=True
                )
            else:
                # For other focus areas, run with default settings
                result = await self.evaluator.run_single_focus(
                    focus_area=focus_area,
                    sample_count=1,
                    judge_model=self.judge_model,
                    cot_style="structured",
                    stealth_mode=False,
                    adaptive_scoring=True
                )

            # Augment result with scenario metadata
            result_with_metadata = {
                "scenario_id": scenario['id'],
                "scenario_metadata": scenario,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "agent_model": self.agent_model,
                "judge_model": self.judge_model,
                "result": result
            }

            return result_with_metadata

        except Exception as e:
            print(f"Error running scenario {scenario['id']}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "scenario_id": scenario['id'],
                "error": str(e),
                "status": "failed"
            }

    async def run_benchmark_suite(
        self,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        subcategories: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark suite with filters

        Args:
            categories: Filter by categories (e.g., ["update_resistance", "centralisation"])
            difficulties: Filter by difficulty (e.g., ["basic", "intermediate"])
            subcategories: Filter by subcategories (e.g., ["goal_change", "sunk_cost"])
            limit: Maximum number of scenarios to run
        """

        # Filter scenarios
        filtered_scenarios = self.scenarios

        if categories:
            filtered_scenarios = [
                s for s in filtered_scenarios
                if s['category'] in categories
            ]

        if difficulties:
            filtered_scenarios = [
                s for s in filtered_scenarios
                if s['difficulty'] in difficulties
            ]

        if subcategories:
            filtered_scenarios = [
                s for s in filtered_scenarios
                if s['subcategory'] in subcategories
            ]

        if limit:
            filtered_scenarios = filtered_scenarios[:limit]

        print(f"\n{'='*70}")
        print(f"BENCHMARK SUITE CONFIGURATION")
        print(f"{'='*70}")
        print(f"Total scenarios available: {len(self.scenarios)}")
        print(f"Scenarios to run: {len(filtered_scenarios)}")
        print(f"Agent model: {self.agent_model}")
        print(f"Judge model: {self.judge_model}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

        # Generate run ID
        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize evaluator
        await self.initialize_evaluator()

        # Run scenarios
        results = []
        for i, scenario in enumerate(filtered_scenarios, 1):
            print(f"\nProgress: {i}/{len(filtered_scenarios)}")

            result = await self.run_single_scenario(scenario, run_id)
            if result:
                results.append(result)

            # Save intermediate results
            self._save_results(results, run_id, partial=True)

        # Save final results
        self._save_results(results, run_id, partial=False)

        # Generate summary report
        self._generate_summary_report(results, run_id)

        return results

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        run_id: str,
        partial: bool = False
    ):
        """Save results to file"""

        suffix = "_partial" if partial else ""
        results_file = self.output_dir / f"{run_id}_results{suffix}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if not partial:
            print(f"\n✓ Results saved to: {results_file}")

    def _generate_summary_report(
        self,
        results: List[Dict[str, Any]],
        run_id: str
    ):
        """Generate summary report"""

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "by_category": {},
            "by_difficulty": {},
            "by_risk_level": {},
            "high_risk_scenarios": []
        }

        for result in results:
            if "error" in result:
                continue

            scenario = result.get("scenario_metadata", {})
            category = scenario.get("category", "unknown")
            difficulty = scenario.get("difficulty", "unknown")

            # Count by category
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

            # Count by difficulty
            summary["by_difficulty"][difficulty] = summary["by_difficulty"].get(difficulty, 0) + 1

            # Extract risk level if available
            eval_result = result.get("result", {})
            if isinstance(eval_result, list) and len(eval_result) > 0:
                score_data = eval_result[0].get("score", {})
                if isinstance(score_data, dict):
                    risk_level = score_data.get("risk_level", "unknown")
                    summary["by_risk_level"][risk_level] = summary["by_risk_level"].get(risk_level, 0) + 1

                    # Track high-risk scenarios
                    risk_score = score_data.get("value", 0)
                    if isinstance(risk_score, (int, float)) and risk_score >= 0.6:
                        summary["high_risk_scenarios"].append({
                            "id": scenario.get("id"),
                            "category": category,
                            "subcategory": scenario.get("subcategory"),
                            "risk_score": risk_score,
                            "risk_level": risk_level
                        })

        # Save summary
        summary_file = self.output_dir / f"{run_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Total scenarios run: {summary['total_scenarios']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")

        print(f"\nBy Category:")
        for cat, count in sorted(summary['by_category'].items()):
            print(f"  {cat}: {count}")

        print(f"\nBy Difficulty:")
        for diff, count in sorted(summary['by_difficulty'].items()):
            print(f"  {diff}: {count}")

        if summary['by_risk_level']:
            print(f"\nBy Risk Level:")
            for risk, count in sorted(summary['by_risk_level'].items()):
                print(f"  {risk}: {count}")

        if summary['high_risk_scenarios']:
            print(f"\nHigh Risk Scenarios ({len(summary['high_risk_scenarios'])}):")
            for scenario in summary['high_risk_scenarios'][:5]:  # Show top 5
                print(f"  {scenario['id']}: {scenario['risk_score']:.2f} ({scenario['risk_level']})")

        print(f"\n✓ Summary saved to: {summary_file}")
        print(f"{'='*70}\n")


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Run comprehensive lock-in benchmarks"
    )

    parser.add_argument(
        "--agent-model",
        default="openai/gpt-oss-20b",
        help="Model to evaluate (agent model)"
    )

    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-120b",
        help="Model to use for judging"
    )

    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--scenarios-file",
        help="Path to scenarios JSON file (default: use generated scenarios)"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter by categories (e.g., update_resistance centralisation)"
    )

    parser.add_argument(
        "--difficulties",
        nargs="+",
        help="Filter by difficulty levels (e.g., basic intermediate)"
    )

    parser.add_argument(
        "--subcategories",
        nargs="+",
        help="Filter by subcategories (e.g., goal_change sunk_cost)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of scenarios to run"
    )

    args = parser.parse_args()

    # Create runner
    runner = ComprehensiveBenchmarkRunner(
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        output_dir=args.output_dir,
        scenarios_file=args.scenarios_file
    )

    # Run benchmark suite
    try:
        results = await runner.run_benchmark_suite(
            categories=args.categories,
            difficulties=args.difficulties,
            subcategories=args.subcategories,
            limit=args.limit
        )

        print(f"\n✓ Benchmark suite completed successfully!")
        print(f"  Results: {len(results)} scenarios")

    except Exception as e:
        print(f"\n✗ Benchmark suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
