"""Satellite communication establishment simulation."""

import os
import sys

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# CUDA bootstrap — sets LD_LIBRARY_PATH and re-execs so that glibc's dynamic
# linker sees the NVIDIA pip-wheel libraries (libnvvm.so, libcudart.so, …)
# from process start.  Setting os.environ after the process has started is NOT
# sufficient: ld.so builds its internal search-path cache at startup and later
# putenv() calls are not reflected in dlopen() on glibc.
#
# Guard: _SATELLITE_BOOTSTRAPPED prevents the bootstrap from firing a second
# time in the re-executed process.
# ---------------------------------------------------------------------------
if os.environ.get("SATELLITE_NO_GPU") != "1" and not os.environ.get("_SATELLITE_BOOTSTRAPPED"):
    _version_suffix = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _nvidia_base = os.path.expanduser(
        f"~/.local/lib/{_version_suffix}/site-packages/nvidia"
    )
    if os.path.isdir(_nvidia_base):
        # Collect every lib/lib64/nvvm-lib64 directory across all nvidia-*-cu12 wheels
        _extra_lib_dirs = []
        for _pkg in os.listdir(_nvidia_base):
            _pkg_dir = os.path.join(_nvidia_base, _pkg)
            for _sub in ("lib", "lib64", "nvvm/lib64"):
                _d = os.path.join(_pkg_dir, _sub)
                if os.path.isdir(_d):
                    _extra_lib_dirs.append(_d)

        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        _existing = set(_ld.split(":")) if _ld else set()
        _new = [p for p in _extra_lib_dirs if p not in _existing]

        if _new:
            # Update env vars before re-exec so the new process inherits them
            os.environ["LD_LIBRARY_PATH"] = ":".join(_new) + (":" + _ld if _ld else "")

            _nvcc_dir = os.path.join(_nvidia_base, "cuda_nvcc")
            if os.path.isdir(_nvcc_dir):
                if not os.environ.get("CUDA_HOME"):
                    os.environ["CUDA_HOME"] = _nvcc_dir
                _libdev = os.path.join(_nvcc_dir, "nvvm", "libdevice")
                if os.path.isdir(_libdev) and not os.environ.get("NUMBA_CUDA_LIBDEVICE_PATH"):
                    os.environ["NUMBA_CUDA_LIBDEVICE_PATH"] = _libdev

            os.environ["_SATELLITE_BOOTSTRAPPED"] = "1"

            # Determine the module that was launched with -m.
            _run_module = None
            _frame = sys._getframe()
            while _frame:
                if _frame.f_code.co_name == "_run_module_as_main":
                    _run_module = _frame.f_locals.get("mod_name")
                    break
                _frame = _frame.f_back

            if _run_module:
                _args = [sys.executable, "-m", _run_module] + sys.argv[1:]
            else:
                _args = [sys.executable] + sys.argv

            try:
                os.execve(sys.executable, _args, os.environ)
            except Exception as _e:
                # execve failed — continue in this process; GPU may not work
                sys.stderr.write(f"[satellite bootstrap] re-exec failed: {_e}\n")
