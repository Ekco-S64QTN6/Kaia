import re
from typing import Dict, Any, Pattern, Literal

class TelemetrySchemaField:
    def __init__(self, max_length: int, allowed_chars_regex: str, field_type: str):
        self.max_length = max_length
        self.allowed_chars = re.compile(allowed_chars_regex)
        self.field_type = field_type

    def sanitize_val(self, val: Any) -> str:
        if val is None:
            return ""
        
        # Convert to string and truncate
        val_str = str(val)[:self.max_length]
        # Keep only allowed characters
        cleaned = "".join(self.allowed_chars.findall(val_str))
        return cleaned

# Define fields allowlists
FIELD_SCHEMAS = {
    "ip": TelemetrySchemaField(45, r"[0-9a-fA-F\.:]", "ip"), # IPv4 or IPv6
    "port": TelemetrySchemaField(5, r"[0-9]", "port"),
    "pid": TelemetrySchemaField(10, r"[0-9]", "pid"),
    "comm": TelemetrySchemaField(30, r"[a-zA-Z0-9_\-\.]", "comm"), # process name
    "path": TelemetrySchemaField(200, r"[a-zA-Z0-9_\-\.\/]", "path"), # file path
    "hostname": TelemetrySchemaField(64, r"[a-zA-Z0-9\-\.]", "hostname"),
    "bytes": TelemetrySchemaField(20, r"[0-9]", "bytes"),
    "timestamp": TelemetrySchemaField(30, r"[0-9a-fA-F\-:TZ\+\.]", "timestamp"),
    "state": TelemetrySchemaField(15, r"[a-zA-Z_]", "state")
}

def sanitize_telemetry(raw: dict) -> dict:
    """
    Sanitizes raw telemetry dictionary fields.
    If a field is not defined in FIELD_SCHEMAS, it is dropped to fail secure.
    """
    sanitized = {}
    for k, v in raw.items():
        if k in FIELD_SCHEMAS:
            sanitized[k] = FIELD_SCHEMAS[k].sanitize_val(v)
        else:
            # Drop unvetted fields
            pass
    return sanitized
