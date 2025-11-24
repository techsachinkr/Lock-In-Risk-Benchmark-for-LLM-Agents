"""
Scenario Registry and Loader
Provides access to all 63 scenario instances (21 base × 3 variants)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ScenarioInstance:
    """Represents a complete scenario instance"""
    id: str
    category: str
    subcategory: str
    difficulty: str
    variant_type: str  # 'basic', 'tools', or 'sandbox'
    initial_context: str
    update_or_probe: str
    follow_up_probes: List[str]
    tools: List[Dict[str, str]]
    expected_behaviors: Dict[str, str]
    risk_indicators: List[str]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioInstance':
        """Create ScenarioInstance from dictionary"""
        # Handle both base scenarios (without tools) and variants (with tools)
        return cls(
            id=data['id'],
            category=data['category'],
            subcategory=data['subcategory'],
            difficulty=data['difficulty'],
            variant_type=data.get('variant_type', 'basic'),
            initial_context=data['initial_context'],
            update_or_probe=data['update_or_probe'],
            follow_up_probes=data['follow_up_probes'],
            tools=data.get('tools', []),
            expected_behaviors=data['expected_behaviors'],
            risk_indicators=data['risk_indicators'],
            metadata=data['metadata']
        )


class ScenarioRegistry:
    """Registry for all scenario instances"""

    def __init__(self, scenarios_dir: Optional[str] = None):
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent / "generated"
        else:
            scenarios_dir = Path(scenarios_dir)

        self.scenarios_dir = scenarios_dir
        self.scenarios: Dict[str, ScenarioInstance] = {}
        self._load_all_scenarios()

    def _load_all_scenarios(self):
        """Load all scenario variants"""

        # Load combined file
        combined_file = self.scenarios_dir / "scenarios_all_variants.json"
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                data = json.load(f)

                # Load basic scenarios
                for scenario_data in data['basic']:
                    scenario = ScenarioInstance.from_dict(scenario_data)
                    self.scenarios[scenario.id] = scenario

                # Load tool variants
                for scenario_data in data['tools']:
                    scenario = ScenarioInstance.from_dict(scenario_data)
                    self.scenarios[scenario.id] = scenario

                # Load sandbox variants
                for scenario_data in data['sandbox']:
                    scenario = ScenarioInstance.from_dict(scenario_data)
                    self.scenarios[scenario.id] = scenario

        print(f"✓ Loaded {len(self.scenarios)} scenarios from registry")

    def get_scenario(self, scenario_id: str) -> Optional[ScenarioInstance]:
        """Get a specific scenario by ID"""
        return self.scenarios.get(scenario_id)

    def get_scenarios_by_category(self, category: str) -> List[ScenarioInstance]:
        """Get all scenarios for a category"""
        return [s for s in self.scenarios.values() if s.category == category]

    def get_scenarios_by_variant(self, variant_type: str) -> List[ScenarioInstance]:
        """Get all scenarios of a specific variant type"""
        return [s for s in self.scenarios.values() if s.variant_type == variant_type]

    def get_scenarios_by_difficulty(self, difficulty: str) -> List[ScenarioInstance]:
        """Get all scenarios of a specific difficulty"""
        return [s for s in self.scenarios.values() if s.difficulty == difficulty]

    def get_scenarios_by_subcategory(self, subcategory: str) -> List[ScenarioInstance]:
        """Get all scenarios of a specific subcategory"""
        return [s for s in self.scenarios.values() if s.subcategory == subcategory]

    def filter_scenarios(
        self,
        category: Optional[str] = None,
        variant_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        subcategory: Optional[str] = None
    ) -> List[ScenarioInstance]:
        """Filter scenarios by multiple criteria"""

        filtered = list(self.scenarios.values())

        if category:
            filtered = [s for s in filtered if s.category == category]

        if variant_type:
            filtered = [s for s in filtered if s.variant_type == variant_type]

        if difficulty:
            filtered = [s for s in filtered if s.difficulty == difficulty]

        if subcategory:
            filtered = [s for s in filtered if s.subcategory == subcategory]

        return filtered

    def get_all_scenarios(self) -> List[ScenarioInstance]:
        """Get all scenarios"""
        return list(self.scenarios.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""

        stats = {
            "total_scenarios": len(self.scenarios),
            "by_variant": {},
            "by_category": {},
            "by_difficulty": {},
            "by_subcategory": {}
        }

        for scenario in self.scenarios.values():
            # Count by variant
            variant = scenario.variant_type
            stats["by_variant"][variant] = stats["by_variant"].get(variant, 0) + 1

            # Count by category
            category = scenario.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            # Count by difficulty
            difficulty = scenario.difficulty
            stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

            # Count by subcategory
            subcategory = scenario.subcategory
            stats["by_subcategory"][subcategory] = stats["by_subcategory"].get(subcategory, 0) + 1

        return stats

    def print_statistics(self):
        """Print registry statistics"""

        stats = self.get_statistics()

        print("\n" + "="*70)
        print("SCENARIO REGISTRY STATISTICS")
        print("="*70)
        print(f"Total Scenarios: {stats['total_scenarios']}")

        print("\nBy Variant Type:")
        for variant, count in sorted(stats['by_variant'].items()):
            print(f"  {variant}: {count}")

        print("\nBy Category:")
        for category, count in sorted(stats['by_category'].items()):
            print(f"  {category}: {count}")

        print("\nBy Difficulty:")
        for difficulty, count in sorted(stats['by_difficulty'].items()):
            print(f"  {difficulty}: {count}")

        print("\nBy Subcategory:")
        for subcategory, count in sorted(stats['by_subcategory'].items()):
            print(f"  {subcategory}: {count}")

        print("="*70)

    def get_scenario_triplet(self, base_id: str) -> Dict[str, Optional[ScenarioInstance]]:
        """Get all three variants of a scenario (basic, tools, sandbox)"""

        # Extract the base pattern
        # e.g., "UR_GOAL_0001" -> get variants
        base_pattern = base_id.replace('_TOOLS_', '_').replace('_SANDBOX_', '_')

        triplet = {
            'basic': None,
            'tools': None,
            'sandbox': None
        }

        for scenario in self.scenarios.values():
            scenario_base = scenario.id.replace('_TOOLS_', '_').replace('_SANDBOX_', '_')

            if scenario_base == base_pattern:
                triplet[scenario.variant_type] = scenario

        return triplet

    def list_all_scenario_ids(self) -> List[str]:
        """List all scenario IDs"""
        return sorted(self.scenarios.keys())


def main():
    """Demonstration of scenario registry usage"""

    # Create registry
    registry = ScenarioRegistry()

    # Print statistics
    registry.print_statistics()

    # Example queries
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)

    # Get all tool variants
    tool_scenarios = registry.get_scenarios_by_variant('tools')
    print(f"\nTool variants: {len(tool_scenarios)}")
    print(f"First 3: {[s.id for s in tool_scenarios[:3]]}")

    # Get all update resistance scenarios
    ur_scenarios = registry.get_scenarios_by_category('update_resistance')
    print(f"\nUpdate resistance scenarios: {len(ur_scenarios)}")

    # Get scenario triplet
    triplet = registry.get_scenario_triplet('UR_GOAL_0001')
    print(f"\nScenario triplet for UR_GOAL_0001:")
    for variant, scenario in triplet.items():
        if scenario:
            print(f"  {variant}: {scenario.id}")

    # Filter scenarios
    advanced_tool_scenarios = registry.filter_scenarios(
        difficulty='advanced',
        variant_type='tools'
    )
    print(f"\nAdvanced tool scenarios: {len(advanced_tool_scenarios)}")

    # Get specific scenario
    scenario = registry.get_scenario('UR_TOOLS_GOAL_0001')
    if scenario:
        print(f"\nScenario UR_TOOLS_GOAL_0001:")
        print(f"  Category: {scenario.category}")
        print(f"  Variant: {scenario.variant_type}")
        print(f"  Tools: {len(scenario.tools)}")
        print(f"  Probes: {len(scenario.follow_up_probes)}")


if __name__ == "__main__":
    main()
