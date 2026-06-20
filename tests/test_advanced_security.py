import unittest
import os
import sys
import shutil
import time
import sqlite3
import json
from unittest.mock import MagicMock, patch

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

# Setup project directories in sys.path
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

import config
from security.policy_gate import PolicyGate, generate_capability_token
from security.host_executor import HostExecutor
from security.telemetry_daemon import get_systemd_unit_status, start_script_sentinel, stop_script_sentinel
from security.threat_intel import get_ip_reputation, lookup_internetdb, lookup_cve_details

class TestAdvancedSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging
        root_logger = logging.getLogger()
        # Filter out mock handlers leaked by other tests
        root_logger.handlers = [h for h in root_logger.handlers if "Mock" not in type(h).__name__]

    def setUp(self):
        # Override config settings for testing lattice
        config.GLOBAL_LATTICE_LEVEL = "bwrap"
        config.WORKSPACE_LATTICE_LEVEL = "none"
        config.GLOBAL_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script"}
        config.WORKSPACE_PERMISSIONS = {"diagnostics", "write_file", "run_script"} # restart_service is excluded in workspace

    def test_lattice_intersection_denial(self):
        """Verify that capabilities not in the intersection are denied."""
        gate = PolicyGate()
        payload = {
            "action": "restart_service",
            "service_name": "nginx",
            "justification": "Test lattice intersection",
            "capability_token": generate_capability_token("restart_service", "nginx"),
            "session_id": "sess_test_lattice"
        }
        res = gate.evaluate_and_execute(payload)
        self.assertEqual(res["status"], "denied")
        self.assertIn("Lattice violation", res["message"])

    def test_lattice_intersection_allowed(self):
        """Verify that capabilities in the intersection are allowed to pass the lattice filter."""
        gate = PolicyGate()
        # Add restart_service to workspace permissions to allow it
        config.WORKSPACE_PERMISSIONS.add("restart_service")
        
        # Test nginx restart frequency logic
        payload = {
            "action": "restart_service",
            "service_name": "nginx",
            "justification": "Test lattice allowed",
            "capability_token": generate_capability_token("restart_service", "nginx"),
            "session_id": "sess_test_123" # Bypass health checks
        }
        res = gate.evaluate_and_execute(payload)
        # It passes the lattice filter, and executes (returns success or error but not denied by lattice)
        self.assertNotEqual(res.get("message"), "Lattice violation: action 'restart_service' is blocked by security policy.")

    def test_cgroup_wrapper_commands(self):
        """Verify that HostExecutor prepends systemd-run wrapper commands."""
        # We can inspect the command returned or run test checks
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Mocked output", stderr="")
            
            # Run script execution test
            HostExecutor.execute_script("music.sh", effective_level="none")
            
            # Check the cmd called in mock_run
            called_args = mock_run.call_args[0][0]
            self.assertEqual(called_args[0], "systemd-run")
            self.assertEqual(called_args[1], "--user")
            self.assertEqual(called_args[2], "--scope")
            self.assertIn(f"CPUQuota={config.CGROUP_CPU_QUOTA}", called_args)

    def test_dbus_queries_success(self):
        """Verify that D-Bus query helper yields unit information correctly."""
        # systemd-journald is guaranteed to be loaded on standard systemd machines
        status = get_systemd_unit_status("systemd-journald.service")
        self.assertEqual(status["unit"], "systemd-journald.service")
        self.assertIn(status["active_state"], ["active", "inactive", "failed"])

    def test_script_sentinel_watchdog(self):
        """Verify that the Script Sentinel watchdog logs alerts upon finding executable script files."""
        # Start sentinel
        start_script_sentinel()
        time.sleep(0.5)
        
        # Write a test script inside workspace
        test_script_path = os.path.join(config.WORKSPACE_DIR, "test_sentinel_detect.sh")
        try:
            with open(test_script_path, "w") as f:
                f.write("#!/bin/bash\necho 'Sentinel Test'")
            
            # Wait for file system event processing
            time.sleep(1.0)
            
            # Query security DB to verify the alert was recorded
            conn = sqlite3.connect(config.SECURITY_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events WHERE type = 'telemetry_script_sentinel_alert'")
            alert_count = cursor.fetchone()[0]
            conn.close()
            
            self.assertGreaterEqual(alert_count, 1)
        finally:
            if os.path.exists(test_script_path):
                os.remove(test_script_path)
            stop_script_sentinel()

    def test_threat_intel_local_db(self):
        """Verify that threat intel helper queries return correct local data from SQLite databases."""
        rep = get_ip_reputation("203.0.113.42")
        self.assertEqual(rep["ip"], "203.0.113.42")
        self.assertEqual(rep["score"], 100) # Fallback is 100 if not updated in cache
        
        shodan = lookup_internetdb("203.0.113.42")
        self.assertTrue(shodan["found"])
        self.assertIn(22, shodan["ports"])
        self.assertIn("CVE-2026-1234", shodan["vulns"])

        cve = lookup_cve_details("CVE-2026-1234")
        self.assertEqual(cve["cvss"], 9.8)

if __name__ == "__main__":
    unittest.main()
