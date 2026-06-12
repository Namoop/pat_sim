"""Satellite communication establishment simulation."""

import os
import sys

__version__ = "0.1.0"

# Auto-bootstrap CUDA environment for NVIDIA user-space wheels
if os.environ.get("SATELLITE_NO_GPU") != "1" and not os.environ.get("_SATELLITE_BOOTSTRAPPED"):
    version_suffix = f"python{sys.version_info.major}.{sys.version_info.minor}"
    local_packages = os.path.expanduser(f"~/.local/lib/{version_suffix}/site-packages")
    if os.path.exists(local_packages):
        cuda_lib_path = os.path.join(local_packages, "nvidia/cuda_runtime/lib")
        cuda_home_path = os.path.join(local_packages, "nvidia/cuda_nvcc")
        if os.path.exists(cuda_lib_path):
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if cuda_lib_path not in ld_path.split(":"):
                os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + (":" + ld_path if ld_path else "")
                os.environ["CUDA_HOME"] = cuda_home_path
                os.environ["_SATELLITE_BOOTSTRAPPED"] = "1"
                try:
                    if sys.argv[0] == "-m":
                        args = [sys.executable, "-m", "satellite"] + sys.argv[1:]
                    else:
                        args = [sys.executable] + sys.argv
                    print(f"[bootstrap] Re-executing: {sys.executable} with args: {args}", flush=True)
                    os.execve(sys.executable, args, os.environ)
                except Exception as e:
                    print(f"[bootstrap] Re-exec failed: {e}", flush=True)
