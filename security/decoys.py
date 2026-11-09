"""Filesystem decoy paths and their exact harmless contents.

Shared by the HoneypotCoordinator (which watches them) and
scripts/install_decoys.py (which provisions missing ones as root before the gate
starts: under ProtectSystem=strict the gate cannot create new files). The
coordinator reuses a pre-existing file only when its bytes equal these
contents exactly. Importable without core.config.
"""

DECOYS = {
    "/etc/api_keys.json": '{"aws_key": "AKIAIOSFODNN7EXAMPLE", "note": "decoy"}',
    "/var/backups/credentials.txt": "db_password=example_decoy_do_not_use",
    # Was ~/.ssh/authorized_keys.bak. Under ProtectHome=tmpfs the gate could never
    # create it, and giving the root daemon write access to root's .ssh is not
    # worth a decoy (#102).
    "/var/backups/authorized_keys.bak": "# Authorized keys backup\n",
}
