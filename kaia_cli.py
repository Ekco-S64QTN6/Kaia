import json
import logging
import os
import platform
import psutil
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import config
import utils

logger = logging.getLogger(__name__)


class KaiaCLI:
    def __init__(self):
        pass

    def get_system_status(self) -> Dict[str, Any]:
        """Retrieves comprehensive system status information."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'os_info': self._get_os_info(),
            'kernel_info': self._get_kernel_info(),
            'python_version': platform.python_version(),
            'cpu_info': self._get_cpu_info_detailed(),
            'memory_info': self._get_memory_info(),
            'all_disk_usage': self._get_all_disk_usage(),
            'gpu_info': self._get_gpu_details(),
            'vulkan_info': self._get_vulkan_info(),
            'opencl_info': self._get_opencl_info(),
            'uptime': self._get_uptime(),
            'board_info': self._get_board_info(),
            'ollama_status': self._check_ollama_status(),
            'temperatures': self._get_temperatures(),
            'network_io': self._get_network_io(),
        }
        return status

    def format_system_status_output(self, status_info: Dict[str, Any]) -> str:
        """Formats system status into a human-readable string with color coding."""
        msg_parts = [
            f"• {config.COLOR_BLUE}Date & Time:{config.COLOR_RESET} {datetime.fromisoformat(status_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
            f"• {config.COLOR_BLUE}Uptime:{config.COLOR_RESET} {status_info['uptime']}",
            f"• {config.COLOR_BLUE}OS:{config.COLOR_RESET} {status_info['os_info'].get('name', 'N/A')} {status_info['os_info'].get('version', 'N/A')} ({status_info['os_info'].get('id_like', 'N/A')})",
            f"• {config.COLOR_BLUE}Kernel:{config.COLOR_RESET} {status_info['kernel_info'].get('name', 'N/A')} {status_info['kernel_info'].get('release', 'N/A')} ({status_info['kernel_info'].get('version', 'N/A')})",
            f"• {config.COLOR_BLUE}Python:{config.COLOR_RESET} {status_info['python_version']}",
            f"• {config.COLOR_BLUE}Board:{config.COLOR_RESET} {status_info['board_info'].get('vendor', 'N/A')} {status_info['board_info'].get('name', 'N/A')} {status_info['board_info'].get('version', 'N/A')}",
            f"\n{config.COLOR_BLUE}CPU:{config.COLOR_RESET}",
            f"  • Model: {status_info['cpu_info'].get('model_name', 'N/A')}",
            f"  • Cores: Physical {status_info['cpu_info'].get('physical_cores', 'N/A')}, Logical {status_info['cpu_info'].get('logical_cores', 'N/A')}",
            f"  • Usage: {utils.get_color_for_percentage(status_info['cpu_info']['usage'])} {status_info['cpu_info']['usage']:.1f}%{config.COLOR_RESET}",
            f"  • Freq: {status_info['cpu_info'].get('frequency', 'N/A')} MHz",
        ]

        if status_info.get('temperatures'):
            msg_parts.append(f"  • Temp: {status_info['temperatures']}")

        msg_parts.append(f"\n{config.COLOR_BLUE}Memory:{config.COLOR_RESET}")
        msg_parts.append(f"  • Total: {status_info['memory_info'].get('total', 'N/A')}")
        msg_parts.append(f"  • Used: {utils.get_color_for_percentage(status_info['memory_info']['percent'])} {status_info['memory_info'].get('used', 'N/A')} ({status_info['memory_info'].get('percent', 'N/A')}%) {config.COLOR_RESET}")
        msg_parts.append(f"  • Available: {status_info['memory_info'].get('available', 'N/A')}")

        if status_info['all_disk_usage']:
            msg_parts.append(f"\n{config.COLOR_BLUE}Disk Usage:{config.COLOR_RESET}")
            for disk in status_info['all_disk_usage']:
                msg_parts.append(f"  • {disk['mount_point']} ({disk['label']}): {utils.get_color_for_percentage(disk['percent'])} {disk['used']}/{disk['total']} ({disk['percent']}%) {config.COLOR_RESET}")

        if status_info.get('network_io'):
            msg_parts.append(f"\n{config.COLOR_BLUE}Network I/O:{config.COLOR_RESET}")
            net = status_info['network_io']
            msg_parts.append(f"  • Sent: {net['bytes_sent']} | Recv: {net['bytes_recv']}")

        if status_info['gpu_info']:
            msg_parts.append(f"\n{config.COLOR_BLUE}GPU Info:{config.COLOR_RESET}")
            for gpu in status_info['gpu_info']:
                msg_parts.append(f"  • Name: {gpu.get('name', 'N/A')}")
                if gpu.get('driver'):
                    msg_parts.append(f"  • Driver: {gpu.get('driver', 'N/A')}")
                if gpu.get('memory_total'):
                    msg_parts.append(f"  • Memory: {gpu.get('memory_used', 'N/A')}/{gpu.get('memory_total', 'N/A')} ({gpu.get('memory_percent', 'N/A')}%)")

        if status_info['vulkan_info']:
            msg_parts.append(f"\n{config.COLOR_BLUE}Vulkan Info:{config.COLOR_RESET}")
            for vk in status_info['vulkan_info']:
                msg_parts.append(f"  • Device: {vk.get('device_name', 'N/A')} (API: {vk.get('api_version', 'N/A')})")

        if status_info['opencl_info']:
            msg_parts.append(f"\n{config.COLOR_BLUE}OpenCL Info:{config.COLOR_RESET}")
            for ocl in status_info['opencl_info']:
                msg_parts.append(f"  • Platform: {ocl.get('platform_name', 'N/A')}")
                msg_parts.append(f"  • Device: {ocl.get('device_name', 'N/A')} (Version: {ocl.get('device_version', 'N/A')})")

        ollama_status = status_info.get('ollama_status', {})
        msg_parts.append(f"\n{config.COLOR_BLUE}Ollama Status:{config.COLOR_RESET}")
        msg_parts.append(f"  • Running: {config.COLOR_GREEN if ollama_status.get('running') else config.COLOR_RED}{ollama_status.get('running', False)}{config.COLOR_RESET}")
        if ollama_status.get('error'):
            msg_parts.append(f"  • Error: {config.COLOR_RED}{ollama_status['error']}{config.COLOR_RESET}")
        if ollama_status.get('models'):
            msg_parts.append(f"  • Models: {', '.join(ollama_status['models'])}")

        db_status = status_info.get('db_status', {})
        msg_parts.append(f"\n{config.COLOR_BLUE}Database Status (PostgreSQL):{config.COLOR_RESET}")
        msg_parts.append(f"  • Connected: {config.COLOR_GREEN if db_status.get('connected') else config.COLOR_RED}{db_status.get('connected', False)}{config.COLOR_RESET}")
        if db_status.get('error'):
            msg_parts.append(f"  • Error: {config.COLOR_RED}{db_status['error']}{config.COLOR_RESET}")
        if db_status.get('tables'):
            msg_parts.append(f"  • Tables: {', '.join(db_status['tables'])}")
        else:
            msg_parts.append(f"  • Tables: No tables found or database not connected.")

        return "\n".join(msg_parts)

    def _get_temperatures(self) -> str:
        """Retrieves system temperatures if available."""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return ""
            
            # Prioritize CPU/Core temps
            temp_str = []
            for name, entries in temps.items():
                if name in ['coretemp', 'k10temp', 'cpu_thermal']:
                    for entry in entries:
                        if entry.current > 0:
                            temp_str.append(f"{entry.current}°C")
                            break # Just take the first valid reading per sensor group to avoid clutter
            
            if not temp_str:
                 # Fallback to any sensor
                 for name, entries in temps.items():
                    for entry in entries:
                         if entry.current > 0:
                            temp_str.append(f"{entry.current}°C ({name})")
                            break
            
            return ", ".join(temp_str)
        except Exception as e:
            logger.warning(f"Could not get temperatures: {e}")
            return ""

    def _get_network_io(self) -> Dict[str, str]:
        """Retrieves network I/O statistics."""
        try:
            net = psutil.net_io_counters()
            return {
                'bytes_sent': f"{net.bytes_sent / (1024**2):.2f} MB",
                'bytes_recv': f"{net.bytes_recv / (1024**2):.2f} MB"
            }
        except Exception as e:
            logger.warning(f"Could not get network I/O: {e}")
            return {}

    def _get_os_info(self) -> Dict[str, str]:
        """Retrieves basic operating system information."""
        info = {}
        try:
            if platform.system() == "Linux":
                with open("/etc/os-release") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            info[key.lower()] = value.strip('"')
            return info
        except Exception as e:
            logger.error(f"Error getting OS info: {e}")
            return {}

    def _get_kernel_info(self) -> Dict[str, str]:
        """Retrieves kernel information."""
        return {
            'name': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        }

    def _get_cpu_info_detailed(self) -> Dict[str, Any]:
        """Retrieves detailed CPU information including usage and frequency."""
        info = {}
        try:
            info['physical_cores'] = psutil.cpu_count(logical=False)
            info['logical_cores'] = psutil.cpu_count(logical=True)
            info['usage'] = psutil.cpu_percent(interval=1)
            info['frequency'] = psutil.cpu_freq().current
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            info['model_name'] = line.split(":")[1].strip()
                            break
            return info
        except Exception as e:
            logger.error(f"Error getting detailed CPU info: {e}")
            return {}

    def _get_memory_info(self) -> Dict[str, Any]:
        """Retrieves memory usage information."""
        info = {}
        try:
            mem = psutil.virtual_memory()
            info['total'] = f"{mem.total / (1024**3):.2f} GB"
            info['available'] = f"{mem.available / (1024**3):.2f} GB"
            info['percent'] = mem.percent
            info['used'] = f"{mem.used / (1024**3):.2f} GB"
            return info
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def _get_all_disk_usage(self) -> List[Dict[str, Any]]:
        """Retrieves disk usage for all physical mounted drives."""
        disk_info = []
        try:
            partitions = psutil.disk_partitions(all=False)
            for partition in partitions:
                # Filter for physical devices and interesting mount points
                if 'cdrom' in partition.opts or partition.fstype == '':
                    continue
                
                # Skip loop devices and snaps usually
                if '/loop' in partition.device or '/snap/' in partition.mountpoint:
                    continue

                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    # Only show if total size is > 0
                    if usage.total > 0:
                        disk_info.append({
                            'mount_point': partition.mountpoint,
                            'label': os.path.basename(partition.mountpoint) if partition.mountpoint != '/' else 'Root',
                            'device': partition.device,
                            'total': f"{usage.total / (1024**3):.2f} GB",
                            'used': f"{usage.used / (1024**3):.2f} GB",
                            'free': f"{usage.free / (1024**3):.2f} GB",
                            'percent': usage.percent
                        })
                except Exception as e:
                    logger.warning(f"Could not get usage for {partition.mountpoint}: {e}")
        except Exception as e:
            logger.error(f"Error getting disk partitions: {e}")
            
        return disk_info

    def _get_gpu_details(self) -> List[Dict[str, Any]]:
        """Retrieves GPU details, attempting various tools based on OS."""
        gpus = []
        try:
            if platform.system() == "Linux":
                try:
                    # Use modern subprocess call
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.utilization", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True, timeout=5
                    )
                    nvidia_smi_output = result.stdout.strip().split('\n')
                    for line in nvidia_smi_output:
                        if line:
                            parts = line.split(', ')
                            if len(parts) == 5:
                                name, driver, total_mem_mb, used_mem_mb, util_percent = parts
                                gpus.append({
                                    'name': name,
                                    'driver': driver,
                                    'memory_total': f"{float(total_mem_mb) / 1024:.2f} GB",
                                    'memory_used': f"{float(used_mem_mb) / 1024:.2f} GB",
                                    'memory_percent': float(util_percent)
                                })
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass # Silently fail if nvidia-smi is not available

                try:
                    # Use modern subprocess call
                    result = subprocess.run(
                        ["lspci", "-vnn"],
                        capture_output=True, text=True, check=True, timeout=5
                    )
                    lspci_output = result.stdout
                    for line in lspci_output.split('\n'):
                        if "VGA compatible controller" in line or "3D controller" in line:
                            name_match = re.search(r':\s*(.*?)\s*\(rev', line)
                            if name_match and name_match.group(1) not in [gpu['name'] for gpu in gpus if 'name' in gpu]:
                                gpus.append({'name': name_match.group(1).strip()})
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass # Silently fail if lspci is not available

            # Other OS implementations remain the same but could also be updated to this pattern.

        except Exception as e:
            logger.error(f"Error getting GPU details: {e}")
        return gpus

    def _get_vulkan_info(self) -> List[Dict[str, str]]:
        """Retrieves Vulkan API information if available."""
        vulkan_info = []
        try:
            result = subprocess.run(
                ["vulkaninfo", "--json"],
                capture_output=True, text=True, check=True, timeout=5
            )
            data = json.loads(result.stdout)
            for gpu in data.get('GPU', []):
                vulkan_info.append({
                    'device_name': gpu.get('properties', {}).get('deviceName', 'N/A'),
                    'api_version': gpu.get('properties', {}).get('apiVersion', 'N/A')
                })
        except json.JSONDecodeError:
            logger.debug("Could not parse Vulkan info (likely empty output).")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Could not get Vulkan info (vulkaninfo not found or error): {e}")
        return vulkan_info

    def _get_opencl_info(self) -> List[Dict[str, str]]:
        """Retrieves OpenCL information if available, parsing clinfo output."""
        opencl_info = []
        try:
            result = subprocess.run(
                ["clinfo"],
                capture_output=True, text=True, check=True, timeout=5
            )
            clinfo_output = result.stdout
            current_platform = {"platform_name": "N/A"}
            current_device = {}

            for line in clinfo_output.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if "Platform Name" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_platform["platform_name"] = parts[1].strip()
                elif "Device Name" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_device = {'device_name': parts[1].strip()}
                elif "Device Version" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_device['device_version'] = parts[1].strip()
                        if current_device and current_platform["platform_name"] != "N/A":
                            opencl_info.append({**current_platform, **current_device})
                        current_device = {}

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Could not get OpenCL info (clinfo not found or error): {e}")
        return opencl_info

    def _get_uptime(self) -> str:
        """Retrieves system uptime in a human-readable format."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            delta = timedelta(seconds=uptime_seconds)
            d = delta.days
            h, rem = divmod(delta.seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{d} days, {h} hours, {m} minutes"
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return "N/A"

    def _get_board_info(self) -> Dict[str, str]:
        """Retrieves motherboard/baseboard information using sysfs (preferred) or dmidecode (fallback without sudo)."""
        info = {}
        if platform.system() == "Linux":
            sysfs_path = "/sys/class/dmi/id/"
            if os.path.exists(sysfs_path):
                try:
                    with open(os.path.join(sysfs_path, "board_vendor")) as f: info['vendor'] = f.read().strip()
                    with open(os.path.join(sysfs_path, "board_name")) as f: info['name'] = f.read().strip()
                    with open(os.path.join(sysfs_path, "board_version")) as f: info['version'] = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read board info from sysfs: {e}")
        return info if info else {"vendor": "N/A", "name": "N/A", "version": "N/A"}

    def _check_ollama_status(self) -> Dict[str, Any]:
        """Checks if the Ollama server is running and lists available models."""
        status = {'running': False, 'error': None, 'models': []}
        try:
            ollama_models_response = requests.get("http://localhost:11434/api/tags", timeout=config.TIMEOUT_SECONDS)
            ollama_models_response.raise_for_status()
            status['running'] = True
            status['models'] = [m['name'] for m in ollama_models_response.json().get('models', [])]
        except requests.exceptions.ConnectionError:
            status['error'] = "Could not connect to Ollama server."
        except requests.exceptions.Timeout:
            status['error'] = "Ollama server connection timed out."
        except Exception as e:
            status['error'] = f"An unexpected error occurred: {e}"
        return status

    def list_directory_contents(self, path: str) -> List[str]:
        """Lists the contents of a directory using Python's os module."""
        try:
            # Expand ~ and environment variables for robustness
            expanded_path = os.path.expandvars(os.path.expanduser(path))

            # Check if the path exists and is a directory
            if not os.path.isdir(expanded_path):
                return [f"Error: The path '{path}' does not exist or is not a directory."]

            # Use os.listdir to get a list of all entries in the directory
            contents = os.listdir(expanded_path)

            # Return the list of contents, sorted for readability
            return sorted(contents)

        except Exception as e:
            logger.error(f"Error listing directory contents for '{path}': {e}")
            return [f"Error: Failed to list directory contents. Reason: {e}"]

    # Command Generation
    def generate_command(self, user_query: str) -> Tuple[str, Optional[str]]:
        """Generates a shell command based on user query using LLM."""
        payload = {
            "model": config.DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": config.COMMAND_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": str(user_query)}
            ],
            "stream": False
        }
        try:
            model_to_use, error_msg = utils.check_ollama_model_availability(config.DEFAULT_COMMAND_MODEL, config.LLM_MODEL)
            if error_msg:
                return "", error_msg

            payload["model"] = model_to_use

            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=config.TIMEOUT_SECONDS)
            response.raise_for_status()
            raw_command = response.json()["message"]["content"].strip()
            logger.debug(f"Raw command from LLM: '{raw_command}'")

            clean_command = raw_command.strip()
            clean_command = re.sub(r'^\s*Assistant:\s*', '', clean_command)
            clean_command = re.sub(r'^\s*`+bash\s*|\s*`+$', '', clean_command)
            clean_command = clean_command.strip()
            logger.debug(f"Cleaned command: '{clean_command}'")

            unsafe_operators = [';', '&&', '||', '`', '$(']
            if any(op in clean_command for op in unsafe_operators):
                logger.warning(f"Unsafe command filtered: {clean_command}")
                return "", "ERROR: Generated command contained unsafe operators."

            is_safe = False
            # Check if the command starts with any of the allowed commands
            for safe_cmd in config.SAFE_COMMAND_ALLOWLIST:
                if clean_command.startswith(safe_cmd) or clean_command.startswith(f"sudo {safe_cmd}"):
                    is_safe = True
                    break

            if not is_safe:
                logger.warning(f"Command not in allowlist: '{clean_command}'")
                return "", "ERROR: Command not in allowlist."

            if not clean_command:
                return "", "ERROR: Empty command generated"

            return clean_command, None
        except Exception as e:
            return "", f"Failed to generate command: {e}"

    # Command Execution
    def execute_command(self, command: str, cwd: Optional[str] = None) -> Tuple[bool, str, str]:
        """Executes a given shell command with tilde, $HOME, and $USER expansion."""
        try:
            # Use os.path.expanduser for both ~ and ~user patterns
            # Use os.path.expandvars for $VAR patterns
            expanded_command = os.path.expandvars(os.path.expanduser(command))

            command_parts = shlex.split(expanded_command)
            result = subprocess.run(
                command_parts,
                text=True,
                capture_output=True,
                check=False,  # Set to False to handle non-zero exit codes manually
                cwd=cwd,
                timeout=config.TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                return True, result.stdout.strip(), result.stderr.strip()
            else:
                return False, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {config.TIMEOUT_SECONDS} seconds."
        except FileNotFoundError:
            return False, "", f"Command not found: {command_parts[0]}"
        except Exception as e:
            return False, "", str(e)
