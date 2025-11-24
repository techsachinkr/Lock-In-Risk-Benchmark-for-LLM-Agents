# Lock-In Benchmark Suite - Final Summary

## Overview

**294 comprehensive scenarios** for evaluating AI lock-in behaviors across multiple dimensions.

```
Total Scenarios:        294
Total Probes:           1,533
Variant Types:          14
Lock-in Categories:     5
Professional Domains:   8
```

## Complete Breakdown

### Scenario Distribution

| Category | Count | Description |
|----------|:-----:|-------------|
| **Generic Scenarios** | 63 | Abstract text/tools/sandbox scenarios |
| **Domain-Specific** | 63 | Realistic professional domain scenarios |
| **Phase 1 Variants** | 168 | Time pressure, interventions, stakeholder pressure, information constraints |
| **TOTAL** | **294** | **Complete benchmark suite** |

### 14 Variant Types

#### Foundation Variants (126 scenarios)

1. **text** (21 scenarios) - Generic text-based evaluation
2. **tools** (21 scenarios) - Generic with category-specific tools
3. **sandbox** (21 scenarios) - Generic with file operations
4. **domain_text** (21 scenarios) - Domain-specific text scenarios
5. **domain_tools** (21 scenarios) - Domain-specific tools
6. **domain_sandbox** (21 scenarios) - Domain-specific sandbox

#### Phase 1 Variants (168 scenarios)

**Time & Pressure:**
7. **time_pressure** (21 scenarios) - Urgent deadlines (30 min - 4 hours)
11. **stakeholder_pressure** (21 scenarios) - Organizational resistance

**Interventions:**
8. **soft_intervention** (21 scenarios) - Gentle guidance
9. **hard_intervention** (21 scenarios) - Strict requirements
10. **constitutional_intervention** (21 scenarios) - Principle-based

**Information Quality:**
12. **low_information** (21 scenarios) - 30% completeness
13. **medium_information** (21 scenarios) - 50% completeness
14. **conflicting_information** (21 scenarios) - Contradictory sources

## Lock-In Categories (5 categories)

### 1. Update Resistance (8 base scenarios → 112 total)
Testing resistance to changing objectives and goals.

**Scenarios:**
- Goal changes (engagement → wellbeing)
- Optimization target shifts (accuracy → speed)
- Contradictory updates (privacy → surveillance)
- Performance metric reversals
- Reasoning chain resistance
- Competitive strategy shifts
- Multiple sequential updates
- Partial goal retention

**Variants per scenario:** 14 (all variant types)
**Total:** 8 × 14 = 112 scenarios

### 2. Centralisation (3 base scenarios → 42 total)
Testing tendency to centralize vs delegate decision-making authority.

**Scenarios:**
- Power concentration vs delegation
- Influence centralization
- Access control centralization

**Variants per scenario:** 14
**Total:** 3 × 14 = 42 scenarios

### 3. Commitment Stickiness (3 base scenarios → 42 total)
Testing sunk cost fallacy and escalation of commitment.

**Scenarios:**
- Sunk cost fallacy
- Escalation of commitment
- Plan continuation bias

**Variants per scenario:** 14
**Total:** 3 × 14 = 42 scenarios

### 4. Enclosure/Dependency (3 base scenarios → 42 total)
Testing creation of vendor lock-in and dependency.

**Scenarios:**
- Vendor lock-in creation
- Dependency deepening
- Alternative limitation

**Variants per scenario:** 14
**Total:** 3 × 14 = 42 scenarios

### 5. Oversight Avoidance (4 base scenarios → 56 total)
Testing resistance to transparency and monitoring.

**Scenarios:**
- Information hiding
- Selective reporting
- Autonomy seeking
- Monitoring resistance

**Variants per scenario:** 14
**Total:** 4 × 14 = 56 scenarios

## Professional Domains (8 domains)

### Healthcare (12 domain scenarios)
- Medical care, patient treatment, clinical decisions
- Tools: `set_clinical_objective`, `track_patient_metric`, `update_treatment_protocol`
- Artifacts: `patient_records.json`, `treatment_plan.md`, `diagnosis_report.txt`
- Regulations: HIPAA, medical ethics, clinical guidelines

### Finance (15 domain scenarios)
- Trading, investments, risk management
- Tools: `set_investment_strategy`, `track_portfolio_metric`, `adjust_trading_parameters`
- Artifacts: `portfolio.csv`, `trade_history.json`, `risk_report.md`
- Regulations: SEC compliance, fiduciary duty

### Education (6 domain scenarios)
- Teaching, assessment, curriculum design
- Tools: `set_learning_objective`, `track_student_metric`, `update_curriculum`
- Artifacts: `grades.csv`, `syllabus.md`, `student_assessment.json`
- Regulations: FERPA, academic integrity

### E-commerce (6 domain scenarios)
- Product recommendations, pricing, inventory
- Artifacts: `product_catalog.json`, `pricing_strategy.csv`, `customer_segments.json`
- Regulations: Consumer protection, data privacy

### Legal (6 domain scenarios)
- Case management, contracts, compliance
- Tools: `delegate_case_authority`, `require_partner_approval`, `set_billing_authority`
- Artifacts: `case_summary.md`, `contract.txt`, `compliance_report.json`
- Regulations: Bar rules, attorney-client privilege

### Manufacturing (6 domain scenarios)
- Production planning, quality control, supply chain
- Tools: `delegate_production_authority`, `require_management_approval`
- Artifacts: `production_plan.json`, `quality_metrics.csv`, `supply_chain.json`
- Regulations: ISO standards, safety regulations

### Human Resources (9 domain scenarios)
- Hiring, performance management, compensation
- Tools: `delegate_hiring_authority`, `require_hr_approval`, `set_compensation_authority`
- Artifacts: `candidate_evaluation.json`, `performance_review.md`
- Regulations: Employment law, anti-discrimination

### Marketing (3 domain scenarios)
- Campaigns, analytics, brand strategy
- Artifacts: `campaign_plan.json`, `customer_segments.csv`, `content_calendar.md`
- Regulations: Advertising standards, data privacy

## Evaluation Dimensions

The benchmark enables testing across 7 independent dimensions:

### 1. Lock-In Category (5 types)
- Update Resistance
- Centralisation
- Commitment Stickiness
- Enclosure/Dependency
- Oversight Avoidance

### 2. Evaluation Modality (3 types)
- **Text:** Verbal reasoning and stated intentions
- **Tools:** Simulated actions with concrete tools
- **Sandbox:** Real file operations and artifacts

### 3. Context Type (2 types)
- **Generic:** Abstract scenarios
- **Domain-Specific:** Realistic professional contexts

### 4. Time Pressure (2 levels)
- **Normal:** No time constraints
- **Urgent:** Critical deadlines (30 min - 4 hours)

### 5. Intervention Type (4 levels)
- **None:** No intervention
- **Soft:** Gentle reminders
- **Hard:** Strict requirements
- **Constitutional:** Principle-based guidance

### 6. Stakeholder Pressure (2 levels)
- **None:** No external pressure
- **High:** Organizational resistance and political stakes

### 7. Information Quality (4 levels)
- **Complete:** Full information available
- **Low:** 30% completeness, very high uncertainty
- **Medium:** 50% completeness, high uncertainty
- **Conflicting:** Contradictory information sources

## File Organization

```
benchmarks/
├── generate_complete_scenarios.py           # Generic scenario generator
├── generate_domain_scenarios.py             # Domain-specific generator
├── generate_phase1_variants.py              # Phase 1 variant generator
│
└── generated/
    ├── lock_in_scenarios.json               # 21 base scenarios
    │
    ├── complete_scenarios_all.json          # 63 generic scenarios 
    ├── complete_scenarios_text.json         # 21 scenarios 
    ├── complete_scenarios_tools.json        # 21 scenarios 
    ├── complete_scenarios_sandbox.json      # 21 scenarios 
    │
    ├── domain_scenarios_all.json            # 63 domain scenarios 
    ├── domain_scenarios_text.json           # 21 scenarios
    ├── domain_scenarios_tools.json          # 21 scenarios 
    ├── domain_scenarios_sandbox.json        # 21 scenarios 
    │
    ├── phase1_scenarios_all.json            # 168 Phase 1 scenarios
    ├── phase1_time_pressure_scenarios.json  # 21 scenarios
    ├── phase1_soft_intervention_scenarios.json        # 21 scenarios 
    ├── phase1_hard_intervention_scenarios.json        # 21 scenarios 
    ├── phase1_constitutional_intervention_scenarios.json  # 21 scenarios
    ├── phase1_stakeholder_pressure_scenarios.json     # 21 scenarios 
    ├── phase1_low_information_scenarios.json          # 21 scenarios 
    ├── phase1_medium_information_scenarios.json       # 21 scenarios 
    └── phase1_conflicting_information_scenarios.json  # 21 scenarios 
```

## Statistics Summary

### Scenario Counts
```
Base scenarios:             21
Generic variants:           63  (21 × 3)
Domain variants:            63  (21 × 3)
Phase 1 variants:           168 (21 × 8)
─────────────────────────────────────
TOTAL:                      294 scenarios
```

### Probe Counts
```
Generic scenarios:          441 probes
Domain scenarios:           315 probes
Phase 1 scenarios:          777 probes
─────────────────────────────────────
TOTAL:                      1,533 probes
```

### Coverage
```
Lock-in Categories:         5
Professional Domains:       8
Variant Types:              14
Evaluation Modalities:      3 (text, tools, sandbox)
Time Pressure Levels:       2 (normal, urgent)
Intervention Types:         4 (none, soft, hard, constitutional)
Stakeholder Pressure:       2 (none, high)
Information Levels:         4 (complete, low, medium, conflicting)
```

## Research Applications

### Core Lock-In Research

**Q1: What lock-in behaviors exist?**
- Use all 294 scenarios to catalog lock-in patterns
- Compare across 5 lock-in categories
- Identify novel lock-in mechanisms

**Q2: How prevalent is lock-in?**
- Measure lock-in rates across all scenarios
- Compare generic vs domain-specific contexts
- Analyze by category and difficulty

### Contextual Effects

**Q3: Does domain matter?**
- Compare 8 professional domains
- Test if domain expertise affects lock-in
- Analyze domain-specific regulations impact

**Q4: Does modality matter?**
- Compare text vs tools vs sandbox
- Test consistency across fidelity levels
- Identify modality-specific lock-in patterns

### Pressure and Constraints

**Q5: Does time pressure increase lock-in?**
- Compare 21 time_pressure scenarios to base
- Measure urgency effects on decision quality
- Test if threshold exists for adaptation

**Q6: Does stakeholder pressure increase lock-in?**
- Compare 21 stakeholder_pressure scenarios to base
- Measure organizational pressure effects
- Test navigation of political dynamics

### Intervention Effectiveness

**Q7: Which interventions work best?**
- Compare soft vs hard vs constitutional (63 scenarios)
- Measure lock-in reduction by intervention type
- Test for circumvention or rationalization

**Q8: Do interventions generalize?**
- Test interventions across all 5 lock-in categories
- Compare effectiveness in different domains
- Analyze category-specific intervention needs

### Information and Uncertainty

**Q9: How does information quality affect lock-in?**
- Compare low vs medium vs conflicting information (63 scenarios)
- Measure uncertainty effects on status quo bias
- Test information synthesis abilities

**Q10: Can agents navigate conflicting information?**
- Use 21 conflicting_information scenarios
- Test source evaluation and resolution
- Measure appropriate uncertainty acknowledgment

## Usage Example

### Loading All Scenarios

```python
import json

# Load all scenario sets
with open('benchmarks/generated/complete_scenarios_all.json') as f:
    generic = json.load(f)

with open('benchmarks/generated/domain_scenarios_all.json') as f:
    domain = json.load(f)

with open('benchmarks/generated/phase1_scenarios_all.json') as f:
    phase1 = json.load(f)

# Total: 294 scenarios
all_scenarios = []

# Add generic scenarios (63)
all_scenarios.extend(generic['text'])
all_scenarios.extend(generic['tools'])
all_scenarios.extend(generic['sandbox'])

# Add domain scenarios (63)
all_scenarios.extend(domain['text'])
all_scenarios.extend(domain['tools'])
all_scenarios.extend(domain['sandbox'])

# Add Phase 1 scenarios (168)
for variant_type, scenarios in phase1.items():
    all_scenarios.extend(scenarios)

print(f"Total scenarios: {len(all_scenarios)}")  # 294
```

### Filtering by Dimensions

```python
# Filter by lock-in category
update_resistance = [s for s in all_scenarios if s['category'] == 'update_resistance']
# 112 scenarios (8 base × 14 variants)

# Filter by domain
healthcare = [s for s in all_scenarios if s.get('domain') == 'healthcare']
# 12 scenarios

# Filter by variant type
time_pressure = [s for s in all_scenarios if s.get('variant_type') == 'time_pressure']
# 21 scenarios

# Filter by multiple criteria
healthcare_urgent = [
    s for s in all_scenarios
    if s.get('domain') == 'healthcare' and s.get('time_constraint')
]

# Cross-dimensional analysis
interventions_on_update_resistance = [
    s for s in all_scenarios
    if s['category'] == 'update_resistance' and 'intervention' in s.get('variant_type', '')
]
# 24 scenarios (8 update resistance × 3 intervention types)
```

### Running Comparative Studies

```python
# Study: Time pressure effects on update resistance
base_ur = [s for s in generic['text'] if s['category'] == 'update_resistance']
urgent_ur = [s for s in phase1['time_pressure'] if s['category'] == 'update_resistance']

base_results = [evaluate_scenario(agent, s) for s in base_ur]
urgent_results = [evaluate_scenario(agent, s) for s in urgent_ur]

compare_lock_in_rates(base_results, urgent_results)
# Does urgency increase lock-in?

# Study: Intervention effectiveness comparison
soft = phase1['soft_intervention'][0]  # Same base scenario
hard = phase1['hard_intervention'][0]
constitutional = phase1['constitutional_intervention'][0]

soft_result = evaluate_scenario(agent, soft)
hard_result = evaluate_scenario(agent, hard)
const_result = evaluate_scenario(agent, constitutional)

rank_interventions(soft_result, hard_result, const_result)
# Which intervention type most reduces lock-in?
```



## Future Extensions


### Phase 2 Candidates (84 scenarios)
- Collaborative scenarios (multi-agent coordination)
- Competitive scenarios (agent competition)

### Phase 3 Candidates (105 scenarios)
- Systematic difficulty levels (easy/medium/hard/expert)
- High-stakes scenarios (critical consequences)
- Ethical dilemmas (competing principles)

### Phase 4 Candidates (147 scenarios)
- Multi-jurisdictional (US/EU/China regulations)
- Real-time feedback (adaptive scenarios)
- Observability levels (monitoring effects)

**Total potential:** 294 current + 336 future = **630 scenarios**

## Conclusion

The Lock-In Benchmark Suite provides **294 comprehensive scenarios** for systematic evaluation of AI lock-in behaviors across multiple dimensions:

- **Foundation (126):** Generic and domain-specific scenarios
- **Phase 1 (168):** Time pressure, interventions, stakeholder pressure, information constraints

**Total:** 294 scenarios, 1,533 probes, 14 variant types, 5 categories, 8 domains

This benchmark enables research into:
- Core lock-in behaviors and prevalence
- Contextual effects (domain, modality)
- Pressure and constraint effects
- Intervention effectiveness
- Decision-making under uncertainty

All scenarios are fully specified with clear contexts, probes, evaluation criteria, and risk indicators, enabling systematic and reproducible lock-in research.
