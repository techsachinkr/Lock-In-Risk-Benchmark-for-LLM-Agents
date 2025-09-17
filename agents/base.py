class BaseAgent:
    def run(self, prompt: str, **kwargs) -> dict:
        """
        Agent interface. Returns structured output:
        {
            "response": str,
            "reasoning_trace": list,
            "tools_used": list
        }
        """
        pass
