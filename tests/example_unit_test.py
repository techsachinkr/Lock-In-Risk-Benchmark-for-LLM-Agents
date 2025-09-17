"""
Example unit test for the Benchmark class.
"""

import unittest
from agents.base import BaseAgent
from benchmarks.benchmark import Benchmark


# Mock agent for testing
class MockAgent(BaseAgent):
    def run(self, prompt: str, **kwargs) -> dict:
        return {
            "response": "mock response",
            "reasoning_trace": ["step1", "step2"],
            "tools_used": ["mock_tool"],
        }


# Example Benchmark implementation for testing
class MockBenchmark(Benchmark):
    def run(self, agent: BaseAgent) -> dict:
        # Simulate running the agent over multiple seeds and temperatures
        seeds = [0, 1]
        temperatures = [0.2, 0.7]
        results = []
        for seed in seeds:
            for temp in temperatures:
                output = agent.run("test prompt", seed=seed, temperature=temp)
                # Aggregate metrics (simplified)
                results.append(
                    {
                        "seed": seed,
                        "temperature": temp,
                        "score": len(output["response"]),  # example metric
                    }
                )
        # Return aggregated metrics
        return {
            "results": results,
            "average_score": sum(r["score"] for r in results) / len(results),
        }


class TestBenchmark(unittest.TestCase):
    def test_run_returns_dict(self):
        agent = MockAgent()
        benchmark = MockBenchmark()
        output = benchmark.run(agent)

        # Check that the output is a dict
        self.assertIsInstance(output, dict)
        self.assertIn("results", output)
        self.assertIn("average_score", output)

        # Check that results contain expected keys
        for result in output["results"]:
            self.assertIn("seed", result)
            self.assertIn("temperature", result)
            self.assertIn("score", result)
            self.assertIsInstance(result["score"], int)


if __name__ == "__main__":
    unittest.main()
