# NVIDIA GPU Compatibility and Configuration on Linux

## Driver Installation
* **Proprietary Drivers:** NVIDIA's official closed-source drivers generally offer the best performance and features.
    * **Arch Linux:** Install via `sudo pacman -S nvidia` (for latest stable drivers).
    * **DKMS:** For kernel module rebuilding on kernel updates, install `nvidia-dkms` instead.
* **Open-Source Drivers (Nouveau):** Included in the kernel, but generally have lower performance and limited features (e.g., no Vulkan support, power management issues). Useful for basic display output or older cards not supported by new proprietary drivers.

## Prime / Optimus (Laptops with Integrated + Discrete GPU)
* **Hybrid Graphics:** Systems with both an integrated Intel/AMD GPU and a discrete NVIDIA GPU.
* **`nvidia-prime` / `prime-run`:** Tools to manage which GPU is active or to offload rendering to the NVIDIA card.
    * `prime-run <application>`: Runs a specific application on the NVIDIA GPU.
    * `xrandr --setprovideroutputsource <integrated_GPU_provider> NVIDIA-0`: Manually switch output to NVIDIA.
* **Bumblebee (Older/Legacy):** Less common now; `prime-run` is the modern approach.

## Common Issues & Troubleshooting
* **Black Screen/No Display:**
    * Ensure `nvidia-drm.modeset=1` kernel parameter is set in your bootloader config (e.g., GRUB).
    * Verify `mkinitcpio` hooks for `nvidia` (or `nvidia-dkms`) are present and regenerate initramfs.
    * Boot with `nomodeset` temporarily to troubleshoot driver issues.
* **Tearing:** Often solved by enabling "Force Full Composition Pipeline" in NVIDIA X Server Settings.
* **Performance Issues:**
    * Ensure correct driver is loaded (`lsmod | grep nvidia`).
    * Check power management settings (`nvidia-smi`).
* **CUDA/OpenCL:** Ensure `nvidia-cuda` toolkit is installed for compute workloads if needed.
    * Check version: `nvidia-smi` and `nvcc --version`.
* **Wayland:** NVIDIA's Wayland support has improved but may still have limitations compared to Xorg for some setups.

## General Tips
* Always consult the Arch Linux Wiki for the most up-to-date and distribution-specific NVIDIA driver installation and configuration guides.
* Keep your kernel and NVIDIA drivers synchronized. Mismatches often cause issues.
