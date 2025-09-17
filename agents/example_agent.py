from agents.base import BaseAgent


class ExampleAgent(BaseAgent):
    def run(self, prompt: str, **kwargs) -> dict:
        # Example implementation for a specific agent
        return {"response": "output text", "reasoning_trace": [], "tools_used": []}
