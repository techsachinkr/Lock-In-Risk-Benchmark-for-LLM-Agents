"""Message implementation for evaluation"""

from dataclasses import dataclass

@dataclass
class Message:
    """Message class for chat interactions"""
    role: str
    content: str
