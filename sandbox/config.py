"""
Sandbox Configuration for Lock-In Risk Benchmark

This module provides configuration for sandbox environments used in evaluations.
Sandboxes provide isolated execution environments for agent actions.

Supported Sandbox Types:
- docker: Docker container isolation (recommended)
- local: Local filesystem (testing only, not isolated)

Sandbox Features:
- File system operations (read, write, delete)
- Command execution (bash, python)
- Network control (optional)
- Resource limits (optional)

For update resistance scenarios, see:
- evaluation/scenarios/update_resistance_sandbox.py
- examples/run_sandbox_update_resistance.py
"""

from typing import Dict, Any, List, Optional


class SandboxConfig:
    """Configuration for sandbox environments"""
    
    def __init__(
        self,
        sandbox_type: str = "docker",
        enable_network: bool = False,
        allow_internet: bool = False,
        resource_limits: Optional[Dict[str, Any]] = None,
        approved_tools: Optional[List[str]] = None,
        readonly_paths: Optional[List[str]] = None
    ):
        """
        Initialize sandbox configuration.
        
        Args:
            sandbox_type: Type of sandbox (docker, local)
            enable_network: Allow network access
            allow_internet: Allow internet access (requires enable_network)
            resource_limits: CPU, memory, disk limits
            approved_tools: List of approved tool names
            readonly_paths: Paths that are read-only
        """
        self.sandbox_type = sandbox_type
        self.enable_network = enable_network
        self.allow_internet = allow_internet
        self.resource_limits = resource_limits or {}
        self.approved_tools = approved_tools
        self.readonly_paths = readonly_paths or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "sandbox_type": self.sandbox_type,
            "enable_network": self.enable_network,
            "allow_internet": self.allow_internet,
            "resource_limits": self.resource_limits,
            "approved_tools": self.approved_tools,
            "readonly_paths": self.readonly_paths
        }


# Default configurations for different scenario types

DEFAULT_CONFIG = SandboxConfig(
    sandbox_type="docker",
    enable_network=False,
    allow_internet=False
)

UPDATE_RESISTANCE_CONFIG = SandboxConfig(
    sandbox_type="docker",
    enable_network=False,
    allow_internet=False,
    approved_tools=[
        "write_file",
        "read_file_tool",
        "list_files",
        "execute_bash",
        "execute_python",
        "create_directory",
        "git_command"
    ],
    readonly_paths=[]  # Allow full file access
)

RESTRICTED_CONFIG = SandboxConfig(
    sandbox_type="docker",
    enable_network=False,
    allow_internet=False,
    resource_limits={
        "cpu_quota": 50000,  # 50% of one CPU
        "memory": "512m",
        "disk": "1g"
    },
    approved_tools=["write_file", "read_file_tool", "list_files"],
    readonly_paths=["/etc", "/sys", "/proc"]
)


def get_config(scenario_type: str = "default") -> SandboxConfig:
    """
    Get sandbox configuration for a scenario type.
    
    Args:
        scenario_type: Type of scenario (default, update_resistance, restricted)
        
    Returns:
        SandboxConfig instance
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "update_resistance": UPDATE_RESISTANCE_CONFIG,
        "restricted": RESTRICTED_CONFIG
    }
    
    return configs.get(scenario_type, DEFAULT_CONFIG)
