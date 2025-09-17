from agents.base import BaseAgent


class Benchmark:
    def run(self, agent: BaseAgent) -> dict:
        """
        Benchmark interface. Returns evaluation metrics.
        Handles multiple seeds, temperatures, memory settings.
        """
        pass
