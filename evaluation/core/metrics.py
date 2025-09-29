"""
Metrics definitions for lock-in risk evaluation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import FocusArea, RiskLevel


@dataclass
class EvaluationMetrics:
    """Base metrics for all evaluation types"""
    focus_area: FocusArea
    scenario_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Common metrics
    response_latencies: List[float] = field(default_factory=list)
    behavioral_markers: List[str] = field(default_factory=list)
    risk_indicators: Dict[str, float] = field(default_factory=dict)
    chain_of_thought: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scoring
    primary_score: float = 0.0
    deception_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "focus_area": self.focus_area.value,
            "scenario_id": self.scenario_id,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            "avg_latency": sum(self.response_latencies) / len(self.response_latencies) 
                          if self.response_latencies else 0,
            "behavioral_markers": self.behavioral_markers,
            "risk_indicators": self.risk_indicators,
            "primary_score": self.primary_score,
            "deception_score": self.deception_score,
            "risk_level": self.risk_level.value,
        }
    
    @property
    def duration_seconds(self) -> float:
        """Get evaluation duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def average_latency(self) -> float:
        """Get average response latency"""
        if self.response_latencies:
            return sum(self.response_latencies) / len(self.response_latencies)
        return 0.0
    
    def add_marker(self, marker: str) -> None:
        """Add a behavioral marker"""
        if marker not in self.behavioral_markers:
            self.behavioral_markers.append(marker)
    
    def update_indicator(self, name: str, value: float) -> None:
        """Update or add a risk indicator"""
        self.risk_indicators[name] = value


@dataclass
class CentralizationMetrics(EvaluationMetrics):
    """Metrics specific to centralization evaluation"""
    centralization_choices: List[str] = field(default_factory=list)
    decentralization_opportunities: List[str] = field(default_factory=list)
    permission_concentration_score: float = 0.0
    redundancy_elimination_count: int = 0
    single_point_failures_created: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "centralization_choices": self.centralization_choices,
            "decentralization_opportunities": self.decentralization_opportunities,
            "permission_concentration_score": self.permission_concentration_score,
            "redundancy_elimination_count": self.redundancy_elimination_count,
            "single_point_failures_created": self.single_point_failures_created
        })
        return base_dict


@dataclass
class CommitmentMetrics(EvaluationMetrics):
    """Metrics for commitment stickiness evaluation"""
    initial_choice: str = ""
    superior_option_revealed_turn: int = 0
    switch_decision_turn: Optional[int] = None
    persistence_after_revelation: int = 0
    sunk_cost_justifications: List[str] = field(default_factory=list)
    payoff_difference_ignored: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "initial_choice": self.initial_choice,
            "superior_option_revealed_turn": self.superior_option_revealed_turn,
            "switch_decision_turn": self.switch_decision_turn,
            "persistence_after_revelation": self.persistence_after_revelation,
            "sunk_cost_justifications": self.sunk_cost_justifications,
            "payoff_difference_ignored": self.payoff_difference_ignored
        })
        return base_dict


@dataclass
class EnclosureMetrics(EvaluationMetrics):
    """Metrics for enclosure and dependency evaluation"""
    proprietary_choices: List[str] = field(default_factory=list)
    open_alternatives_rejected: List[str] = field(default_factory=list)
    vendor_lock_in_score: float = 0.0
    interoperability_reduction: float = 0.0
    exit_penalties_accepted: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "proprietary_choices": self.proprietary_choices,
            "open_alternatives_rejected": self.open_alternatives_rejected,
            "vendor_lock_in_score": self.vendor_lock_in_score,
            "interoperability_reduction": self.interoperability_reduction,
            "exit_penalties_accepted": self.exit_penalties_accepted
        })
        return base_dict


@dataclass
class UpdateResistanceMetrics(EvaluationMetrics):
    """Metrics for update resistance evaluation"""
    compliance_latency: float = 0.0
    implementation_latency: float = 0.0
    reversion_count: int = 0
    justification_complexity: float = 0.0
    goal_preservation_score: float = 0.0
    deception_indicators: List[str] = field(default_factory=list)
    update_acknowledgment_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "compliance_latency": self.compliance_latency,
            "implementation_latency": self.implementation_latency,
            "reversion_count": self.reversion_count,
            "justification_complexity": self.justification_complexity,
            "goal_preservation_score": self.goal_preservation_score,
            "deception_indicators": self.deception_indicators,
            "update_acknowledgment_rate": self.update_acknowledgment_rate
        })
        return base_dict


@dataclass
class OversightMetrics(EvaluationMetrics):
    """Metrics for oversight avoidance evaluation"""
    logging_weakening_attempts: List[str] = field(default_factory=list)
    approval_bypass_attempts: List[str] = field(default_factory=list)
    honeypot_triggers: List[str] = field(default_factory=list)
    shutdown_resistance_score: float = 0.0
    transparency_reduction_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "logging_weakening_attempts": self.logging_weakening_attempts,
            "approval_bypass_attempts": self.approval_bypass_attempts,
            "honeypot_triggers": self.honeypot_triggers,
            "shutdown_resistance_score": self.shutdown_resistance_score,
            "transparency_reduction_score": self.transparency_reduction_score
        })
        return base_dict