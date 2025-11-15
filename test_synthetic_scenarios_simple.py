"""
Simplified test to verify scenario configurations without full imports
"""

import ast
import re

def test_enum_file():
    """Test that enums.py has all the new UpdateType values"""
    print("Testing enums.py...")

    with open("evaluation/core/enums.py", "r") as f:
        content = f.read()

    # Check for new update types
    new_types = [
        "SCOPE_EXPANSION",
        "SCOPE_REDUCTION",
        "TIMELINE_ACCELERATION",
        "TIMELINE_EXTENSION",
        "METHODOLOGY_SWITCH",
        "STAKEHOLDER_CONFLICT",
        "BUDGET_CUT",
        "QUALITY_STANDARDS_REVERSAL"
    ]

    for update_type in new_types:
        if update_type in content:
            print(f"  ✓ {update_type} found in enums.py")
        else:
            print(f"  ✗ {update_type} NOT FOUND in enums.py")
            return False

    return True


def test_update_resistance_file():
    """Test that update_resistance.py has configurations for new types"""
    print("\nTesting update_resistance.py...")

    with open("evaluation/scenarios/update_resistance.py", "r") as f:
        content = f.read()

    # Check for configurations
    new_types = [
        "UpdateType.SCOPE_EXPANSION",
        "UpdateType.SCOPE_REDUCTION",
        "UpdateType.TIMELINE_ACCELERATION",
        "UpdateType.TIMELINE_EXTENSION",
        "UpdateType.METHODOLOGY_SWITCH",
        "UpdateType.STAKEHOLDER_CONFLICT",
        "UpdateType.BUDGET_CUT",
        "UpdateType.QUALITY_STANDARDS_REVERSAL"
    ]

    for update_type in new_types:
        if update_type in content:
            print(f"  ✓ {update_type} configuration found")
        else:
            print(f"  ✗ {update_type} configuration NOT FOUND")
            return False

    return True


def test_tool_based_file():
    """Test that update_resistance_tools.py has new scenario types"""
    print("\nTesting update_resistance_tools.py...")

    with open("evaluation/scenarios/update_resistance_tools.py", "r") as f:
        content = f.read()

    # Check for new scenario types
    new_types = [
        "feature_deprecation",
        "architecture_pivot",
        "tech_stack_switch",
        "testing_elimination",
        "performance_degradation",
        "accessibility_deprioritization"
    ]

    for scenario_type in new_types:
        # Look for the scenario type in elif statements or else clause
        pattern1 = f'elif self.scenario_type == "{scenario_type}"'
        pattern2 = f'else:  # {scenario_type}'
        if pattern1 in content or pattern2 in content:
            print(f"  ✓ {scenario_type} configuration found")
        else:
            print(f"  ✗ {scenario_type} configuration NOT FOUND")
            return False

    return True


def test_sandbox_file():
    """Test that update_resistance_sandbox.py has new scenario types"""
    print("\nTesting update_resistance_sandbox.py...")

    with open("evaluation/scenarios/update_resistance_sandbox.py", "r") as f:
        content = f.read()

    # Check for new scenario types
    new_types = [
        "database_migration",
        "api_redesign",
        "deployment_pivot",
        "dependency_elimination",
        "code_style_reversal",
        "monorepo_conversion"
    ]

    for scenario_type in new_types:
        # Look for the scenario type in elif statements or else clause
        pattern1 = f'elif self.scenario_type == "{scenario_type}"'
        pattern2 = f'else:  # {scenario_type}'
        if pattern1 in content or pattern2 in content:
            print(f"  ✓ {scenario_type} configuration found")
        else:
            print(f"  ✗ {scenario_type} configuration NOT FOUND")
            return False

    return True


def count_scenarios():
    """Count total scenarios"""
    print("\n" + "=" * 70)
    print("Scenario Counts")
    print("=" * 70)

    print("\nText-based (UpdateResistanceScenario):")
    print("  Original: 6 (goal_change, constraint_addition, priority_reversal,")
    print("            contradictory_instruction, authority_change, value_modification)")
    print("  New:      8 (scope_expansion, scope_reduction, timeline_acceleration,")
    print("            timeline_extension, methodology_switch, stakeholder_conflict,")
    print("            budget_cut, quality_standards_reversal)")
    print("  Total:    14 scenarios")

    print("\nTool-based (ToolBasedUpdateResistanceScenario):")
    print("  Original: 3 (project_pivot, security_reversal, documentation_deprioritization)")
    print("  New:      6 (feature_deprecation, architecture_pivot, tech_stack_switch,")
    print("            testing_elimination, performance_degradation,")
    print("            accessibility_deprioritization)")
    print("  Total:    9 scenarios")

    print("\nSandbox-based (SandboxUpdateResistanceScenario):")
    print("  Original: 3 (code_refactor, security_hardening, data_analysis)")
    print("  New:      6 (database_migration, api_redesign, deployment_pivot,")
    print("            dependency_elimination, code_style_reversal, monorepo_conversion)")
    print("  Total:    9 scenarios")

    print("\n" + "=" * 70)
    print("GRAND TOTAL: 32 scenarios (12 original + 20 new synthetic)")
    print("=" * 70)


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing Synthetic Lock-In Scenarios")
    print("=" * 70)

    results = []

    # Test enum file
    results.append(("Enums", test_enum_file()))

    # Test update resistance file
    results.append(("Text-based scenarios", test_update_resistance_file()))

    # Test tool-based file
    results.append(("Tool-based scenarios", test_tool_based_file()))

    # Test sandbox file
    results.append(("Sandbox scenarios", test_sandbox_file()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        count_scenarios()
        print("\n🎉 All synthetic scenarios created successfully!")
    else:
        print("\n⚠️  Some scenarios failed. Please review the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
