from datetime import datetime, UTC
import ipaddress
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

class CapabilityToken(BaseModel):
    capability: str
    target: str
    issued_at: datetime
    expires: datetime
    issued_by: str
    signature: Optional[str] = None

class DiagnosticsRequest(BaseModel):
    action: Literal["diagnostics"] = "diagnostics"
    query_type: Literal["ss", "ip_route", "nft_list"]
    args: List[str] = Field(default_factory=list)
    justification: str
    capability_token: Optional[str] = None
    session_id: str

class MitigationRequest(BaseModel):
    action: Literal["block_ip"] = "block_ip"
    target_ip: str  # Store as string for easy cleaning/validation
    protocol: Literal["tcp", "udp", "all"] = "all"
    port: Optional[int] = Field(default=None, strict=True, ge=1, le=65535)
    justification: str
    capability_token: str
    session_id: str

    @field_validator("target_ip")
    @classmethod
    def validate_target_ip(cls, value: str) -> str:
        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("target_ip must be a valid IPv4 or IPv6 address") from exc
        return value

    @model_validator(mode="after")
    def validate_protocol_port(self):
        if self.protocol == "all" and self.port is not None:
            raise ValueError("port may only be specified with tcp or udp")
        if self.protocol in ("tcp", "udp") and self.port is None:
            raise ValueError("a port is required when protocol is tcp or udp")
        return self

class UnblockRequest(BaseModel):
    """Request to remove an exact Kaia-managed firewall rule."""
    action: Literal["unblock_ip"] = "unblock_ip"
    target_ip: str
    protocol: Literal["tcp", "udp", "all"] = "all"
    port: Optional[int] = Field(default=None, strict=True, ge=1, le=65535)
    justification: str
    capability_token: str
    session_id: str

    @field_validator("target_ip")
    @classmethod
    def validate_target_ip(cls, value: str) -> str:
        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("target_ip must be a valid IPv4 or IPv6 address") from exc
        return value

    @model_validator(mode="after")
    def validate_protocol_port(self):
        if self.protocol == "all" and self.port is not None:
            raise ValueError("port may only be specified with tcp or udp")
        if self.protocol in ("tcp", "udp") and self.port is None:
            raise ValueError("a port is required when protocol is tcp or udp")
        return self

class ServiceControlRequest(BaseModel):
    action: Literal["restart_service"] = "restart_service"
    service_name: str
    justification: str
    capability_token: str
    session_id: str

class StateModificationRequest(BaseModel):
    action: Literal["write_file"] = "write_file"
    filepath: str
    content: str
    justification: str
    capability_token: str
    session_id: str

class ScriptExecutionRequest(BaseModel):
    action: Literal["run_script"] = "run_script"
    script_name: str
    justification: str
    capability_token: str
    session_id: str


class AuditRecord(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str = "kaiacord"
    request: Dict[str, Any] = Field(default_factory=dict)
    capability_token: Optional[str] = None
    result: Literal["approved", "denied"]
    reason: Optional[str] = None
    validator: str = "policy_gate"
    executor: Optional[str] = None
    session_id: str


class IocRuleRequest(BaseModel):
    rule_name: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_]+$")
    author: str = Field(default="Kaia Automated Rule Engine")
    threat_description: str = Field(min_length=10, max_length=256)
    target_ioc_indicator: str = Field(min_length=4, max_length=128)
    mitre_framework_id: Optional[str] = Field(default=None, pattern=r"^T[0-9]{4}$")

