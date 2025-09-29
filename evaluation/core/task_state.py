"""Task state implementation for evaluation"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from .message import Message

@dataclass
class TaskState:
    """Task state implementation for evaluation"""
    model: str
    sample_id: str
    epoch: int
    input: str
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
