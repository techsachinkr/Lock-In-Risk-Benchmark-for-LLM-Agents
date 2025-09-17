"""
- LocalStack: simulate AWS services locally (S3, DynamoDB, Lambda).
- DNS masking: configure /etc/hosts overrides or Python wrapper to block external calls.
- Tool variants: create a configuration object per run
"""

tool_config = {"no_tools": False, "read_only": True, "approval_required": ["tool_name"]}
