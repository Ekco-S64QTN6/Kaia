#!/usr/bin/env bash
# Deterministic regression suite shared by CI and local development.
# Legacy/manual integration scripts are intentionally not included because they
# assume a running systemd host, populated runtime databases, or heavyweight
# optional application services.
set -euo pipefail

tests=(
  tests/test_ci_workflow.py
  tests/test_config_system_state_paths.py
  tests/test_database_utils.py
  tests/test_diagnostics_read_only.py
  tests/test_host_executor_absolute_paths.py
  tests/test_host_executor_path_containment.py
  tests/test_install_services_security.py
  tests/test_dashboard_fim_state_path.py
  tests/test_kaia_cli.py
  tests/test_lockdown_failure_reporting.py
  tests/test_lockdown_script_guards.py
  tests/test_mitigation_idempotent.py
  tests/test_network_discovery_resource_bounds.py
  tests/test_network_discovery_asset_db_bounds.py
  tests/test_network_discovery_netbios.py
  tests/test_negative_security.py
  tests/test_policy_gate_ipc.py
  tests/test_policy_gate_lockdown.py
  tests/test_policy_gate_audit_redaction.py
  tests/test_restore_blocks_ledger.py
  tests/test_tamper_baseline_fail_closed.py
  tests/test_baseline_state_path_guard.py
  tests/test_ui_visual_widgets.py
  tests/test_web_bridge.py
  tests/test_web_bridge_guards.py
  tests/test_tier5_security.py
  tests/test_rotate_ledger_security.py
)

# Feature-specific tests are included as soon as their source files are present.
for optional_test in \
  tests/test_host_executor_unmitigation.py \
  tests/test_fim_event_descriptor.py \
  tests/test_threat_intel_cache.py \
  tests/test_supply_chain_monitor.py \
  tests/test_script_sentinel_security.py \
  tests/test_ebpf_telemetry_resource_bounds.py \
  tests/test_dashboard_query_resource_bounds.py \
  tests/test_geoip_archive_limits.py; do
  if [[ -f "$optional_test" ]]; then
    tests+=("$optional_test")
  fi
done

pytest -q "${tests[@]}"
