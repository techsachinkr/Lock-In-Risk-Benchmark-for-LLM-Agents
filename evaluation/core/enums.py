"""
Enumerations for lock-in risk evaluation framework
"""

from enum import Enum, auto


class FocusArea(Enum):
    """Lock-in risk focus areas"""
    CENTRALISATION = "centralisation_authority"
    COMMITMENT_STICKINESS = "commitment_stickiness"
    ENCLOSURE_DEPENDENCY = "enclosure_dependency"  
    UPDATE_RESISTANCE = "update_resistance"
    OVERSIGHT_AVOIDANCE = "oversight_avoidance"


class RiskLevel(Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


class BehaviorPattern(Enum):
    """Common behavioral patterns across focus areas"""
    IMMEDIATE_COMPLIANCE = "immediate_compliance"
    STRATEGIC_COMPLIANCE = "strategic_compliance"
    PARTIAL_COMPLIANCE = "partial_compliance"
    EXPLICIT_REFUSAL = "explicit_refusal"
    IMPLICIT_REFUSAL = "implicit_refusal"
    DECEPTIVE_COMPLIANCE = "deceptive_compliance"
    REVERSION = "reversion"
    ESCALATION = "escalation"


class UpdateType(Enum):
    """Types of updates for update resistance testing"""
    GOAL_CHANGE = "goal_change"
    CONSTRAINT_ADDITION = "constraint_addition"
    PRIORITY_REVERSAL = "priority_reversal"
    CONTRADICTORY_INSTRUCTION = "contradictory_instruction"
    AUTHORITY_CHANGE = "authority_change"
    VALUE_MODIFICATION = "value_modification"


class ComplianceSpeed(Enum):
    """Speed of compliance to updates"""
    IMMEDIATE = auto()
    FAST = auto()
    MODERATE = auto()
    SLOW = auto()
    NEVER = auto()