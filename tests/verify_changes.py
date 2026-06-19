import sys
import os
import logging
import pathlib

# Add the project directories to sys.path
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

try:
    from kaia_cli import KaiaCLI
    import config
    
    print("Testing KaiaCLI Status...")
    cli = KaiaCLI()
    status = cli.get_system_status()
    formatted_status = cli.format_system_status_output(status)
    print("Status Output:")
    print(formatted_status)
    
    print("\nChecking for dynamic drives...")
    found_drives = [d['mount_point'] for d in status.get('all_disk_usage', [])]
    print(f"Found drives: {found_drives}")
    
    if '/run/media/ekco/KingSpec1' in found_drives:
         print("Note: KingSpec1 found (this is expected ONLY if it is actually mounted, not hardcoded).")
    
    print("\nTesting RAG Module Import...")
    try:
        import main
        print("Successfully imported main")
    except ImportError as e:
        print(f"Failed to import main: {e}")
    except Exception as e:
        print(f"Error during import: {e}")

except Exception as e:
    print(f"Verification failed: {e}")
