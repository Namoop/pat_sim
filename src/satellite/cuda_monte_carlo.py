"""CUDA-accelerated Monte Carlo simulation runner."""

from __future__ import annotations

import os
import sys
import math

# Bootstrap: add all NVIDIA pip-wheel library directories before importing Numba.
# Numba needs libcudart.so (cuda_runtime/lib) AND libnvvm.so (cuda_nvcc/nvvm/lib64).
_version_suffix = f"python{sys.version_info.major}.{sys.version_info.minor}"
_nvidia_base = os.path.expanduser(f"~/.local/lib/{_version_suffix}/site-packages/nvidia")
if os.path.isdir(_nvidia_base):
    _extra = []
    for _pkg in os.listdir(_nvidia_base):
        for _sub in ("lib", "lib64", "nvvm/lib64"):
            _d = os.path.join(_nvidia_base, _pkg, _sub)
            if os.path.isdir(_d):
                _extra.append(_d)
    if _extra:
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        _have = set(_ld.split(":")) if _ld else set()
        _new = [p for p in _extra if p not in _have]
        if _new:
            os.environ["LD_LIBRARY_PATH"] = ":".join(_new) + (":" + _ld if _ld else "")
    _nvcc = os.path.join(_nvidia_base, "cuda_nvcc")
    if os.path.isdir(_nvcc) and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = _nvcc
    _libdev = os.path.join(_nvcc, "nvvm", "libdevice")
    if os.path.isdir(_libdev) and not os.environ.get("NUMBA_CUDA_LIBDEVICE_PATH"):
        os.environ["NUMBA_CUDA_LIBDEVICE_PATH"] = _libdev
    del _extra, _nvcc, _libdev
del _version_suffix, _nvidia_base

import numpy as np
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

FLOAT_DTYPE = np.float32

from dataclasses import dataclass

@dataclass
class _MockSchedule:
    total_duration: float

@dataclass
class _MockScenarioResult:
    success: bool
    hit_at_t: float | None
    strategy_name: str | None
    schedule: _MockSchedule
    config: object

from satellite.config import (
    MonteCarloConfig,
    ScenarioInstance,
    build_scenario_config,
    load_simulation_config,
    load_monte_carlo_config,
)
from satellite.math3d import Vec3
from satellite.monte_carlo import MonteCarloRunResult, MonteCarloSummary, _build_monte_carlo_summary, sample_offsets
from satellite.strategy.movements import Hold, Reset, Spiral, SerpentineRaster, Rosette, Lissajous
from satellite.strategy.base import StrategyContext, StrategyResult
from satellite.strategy.meta import MetaStrategy
from satellite.strategy.actions import StrategyScript

# ==============================================================================
# GPU Device math helpers
# ==============================================================================

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def norm_device(v):
        return (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5

    @cuda.jit(device=True)
    def normalize_device(v, out):
        n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
        if n > 1e-15:
            out[0] = v[0] / n
            out[1] = v[1] / n
            out[2] = v[2] / n
        else:
            out[0] = 0.0
            out[1] = 0.0
            out[2] = 0.0

    @cuda.jit(device=True)
    def dot_device(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    @cuda.jit(device=True)
    def cross_device(a, b, out):
        out[0] = a[1]*b[2] - a[2]*b[1]
        out[1] = a[2]*b[0] - a[0]*b[2]
        out[2] = a[0]*b[1] - a[1]*b[0]

    @cuda.jit(device=True)
    def angle_between_device(a, b):
        a_u = cuda.local.array(3, dtype=FLOAT_DTYPE)
        b_u = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device(a, a_u)
        normalize_device(b, b_u)
        cos_theta = a_u[0]*b_u[0] + a_u[1]*b_u[1] + a_u[2]*b_u[2]
        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:
            cos_theta = -1.0
        return math.acos(cos_theta)

    @cuda.jit(device=True)
    def slerp_device(a, b, t, out):
        a_u = cuda.local.array(3, dtype=FLOAT_DTYPE)
        b_u = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device(a, a_u)
        normalize_device(b, b_u)
        cos_theta = a_u[0]*b_u[0] + a_u[1]*b_u[1] + a_u[2]*b_u[2]
        if cos_theta > 0.9999:
            diff_x = b_u[0] - a_u[0]
            diff_y = b_u[1] - a_u[1]
            diff_z = b_u[2] - a_u[2]
            v_x = a_u[0] + t * diff_x
            v_y = a_u[1] + t * diff_y
            v_z = a_u[2] + t * diff_z
            n = (v_x*v_x + v_y*v_y + v_z*v_z) ** 0.5
            out[0] = v_x / n
            out[1] = v_y / n
            out[2] = v_z / n
            return
        if cos_theta < -0.9999:
            cos_theta = -0.9999
        theta = math.acos(cos_theta)
        sin_theta = math.sin(theta)
        w1 = math.sin((1.0 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        out[0] = w1 * a_u[0] + w2 * b_u[0]
        out[1] = w1 * a_u[1] + w2 * b_u[1]
        out[2] = w1 * a_u[2] + w2 * b_u[2]

    @cuda.jit(device=True)
    def rotate_vector_device(v, axis, angle, out):
        ax = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device(axis, ax)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        c_x = ax[1]*v[2] - ax[2]*v[1]
        c_y = ax[2]*v[0] - ax[0]*v[2]
        c_z = ax[0]*v[1] - ax[1]*v[0]
        d = ax[0]*v[0] + ax[1]*v[1] + ax[2]*v[2]
        factor = d * (1.0 - cos_a)
        out[0] = v[0]*cos_a + c_x*sin_a + ax[0]*factor
        out[1] = v[1]*cos_a + c_y*sin_a + ax[1]*factor
        out[2] = v[2]*cos_a + c_z*sin_a + ax[2]*factor

    @cuda.jit(device=True)
    def rotate_toward_device(from_dir, to_dir, max_angle, out):
        angle = angle_between_device(from_dir, to_dir)
        if angle <= max_angle:
            normalize_device(to_dir, out)
            return
        slerp_device(from_dir, to_dir, max_angle / angle, out)

    @cuda.jit(device=True)
    def spherical_angles_from_direction_device(direction, out_angles):
        unit = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device(direction, unit)
        phi = math.atan2(unit[1], unit[0])
        theta = math.atan2((unit[0] ** 2 + unit[1] ** 2) ** 0.5, unit[2])
        out_angles[0] = theta
        out_angles[1] = phi

    @cuda.jit(device=True)
    def spherical_to_cartesian_device(theta, phi, out):
        sin_theta = math.sin(theta)
        out[0] = sin_theta * math.cos(phi)
        out[1] = sin_theta * math.sin(phi)
        out[2] = math.cos(theta)

    @cuda.jit(device=True)
    def transmitter_basis_device(theta_0, phi_0, u_x, u_y, u_z):
        spherical_to_cartesian_device(theta_0, phi_0, u_z)
        u_x[0] = math.cos(theta_0) * math.cos(phi_0)
        u_x[1] = math.cos(theta_0) * math.sin(phi_0)
        u_x[2] = -math.sin(theta_0)
        u_y[0] = -math.sin(phi_0)
        u_y[1] = math.cos(phi_0)
        u_y[2] = 0.0

    @cuda.jit(device=True)
    def direction_with_tangent_offset_device(base, tangent_u, tangent_v, offset_u, offset_v, out):
        axis = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device(base, axis)
        w_x = offset_u * tangent_u[0] + offset_v * tangent_v[0]
        w_y = offset_u * tangent_u[1] + offset_v * tangent_v[1]
        w_z = offset_u * tangent_u[2] + offset_v * tangent_v[2]
        w_mag = (w_x*w_x + w_y*w_y + w_z*w_z) ** 0.5
        if w_mag < 1e-15:
            out[0] = axis[0]
            out[1] = axis[1]
            out[2] = axis[2]
            return
        rot_axis = cuda.local.array(3, dtype=FLOAT_DTYPE)
        c_x = axis[1]*w_z - axis[2]*w_y
        c_y = axis[2]*w_x - axis[0]*w_z
        c_z = axis[0]*w_y - axis[1]*w_x
        n = (c_x*c_x + c_y*c_y + c_z*c_z) ** 0.5
        rot_axis[0] = c_x / n
        rot_axis[1] = c_y / n
        rot_axis[2] = c_z / n
        rotate_vector_device(axis, rot_axis, w_mag, out)

    @cuda.jit(device=True)
    def direction_with_local_offset_device(base, theta_offset, phi_offset, out):
        out_angles = cuda.local.array(2, dtype=FLOAT_DTYPE)
        spherical_angles_from_direction_device(base, out_angles)
        u_x = cuda.local.array(3, dtype=FLOAT_DTYPE)
        u_y = cuda.local.array(3, dtype=FLOAT_DTYPE)
        u_z = cuda.local.array(3, dtype=FLOAT_DTYPE)
        transmitter_basis_device(out_angles[0], out_angles[1], u_x, u_y, u_z)
        direction_with_tangent_offset_device(u_z, u_x, u_y, theta_offset, phi_offset, out)

    @cuda.jit(device=True)
    def offsets_to_target_device(bench_boresight, target, out_offsets):
        tilt = angle_between_device(bench_boresight, target)
        if tilt < 1e-15:
            out_offsets[0] = 0.0
            out_offsets[1] = 0.0
            return
        axis_perp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        cross_device(bench_boresight, target, axis_perp)
        n_mag = (axis_perp[0]**2 + axis_perp[1]**2 + axis_perp[2]**2) ** 0.5
        if n_mag < 1e-15:
            out_offsets[0] = 0.0
            out_offsets[1] = 0.0
            return
        n = cuda.local.array(3, dtype=FLOAT_DTYPE)
        n[0] = axis_perp[0] / n_mag
        n[1] = axis_perp[1] / n_mag
        n[2] = axis_perp[2] / n_mag
        
        n_cross_base = cuda.local.array(3, dtype=FLOAT_DTYPE)
        cross_device(n, bench_boresight, n_cross_base)
        
        w_x = tilt * n_cross_base[0]
        w_y = tilt * n_cross_base[1]
        w_z = tilt * n_cross_base[2]
        
        out_angles = cuda.local.array(2, dtype=FLOAT_DTYPE)
        spherical_angles_from_direction_device(bench_boresight, out_angles)
        u_x = cuda.local.array(3, dtype=FLOAT_DTYPE)
        u_y = cuda.local.array(3, dtype=FLOAT_DTYPE)
        u_z = cuda.local.array(3, dtype=FLOAT_DTYPE)
        transmitter_basis_device(out_angles[0], out_angles[1], u_x, u_y, u_z)
        
        out_offsets[0] = w_x*u_x[0] + w_y*u_x[1] + w_z*u_x[2]
        out_offsets[1] = w_x*u_y[0] + w_y*u_y[1] + w_z*u_y[2]

    @cuda.jit(device=True)
    def beam_hits_dish_device(
        tx_pos, rx_pos, dish_boresight, cos_fov, beam_axis, cos_alpha, beam_length, rx_body_radius
    ):
        rx_mount = cuda.local.array(3, dtype=FLOAT_DTYPE)
        rx_mount[0] = rx_pos[0] + dish_boresight[0] * rx_body_radius
        rx_mount[1] = rx_pos[1] + dish_boresight[1] * rx_body_radius
        rx_mount[2] = rx_pos[2] + dish_boresight[2] * rx_body_radius
        
        px = rx_mount[0] - tx_pos[0]
        py = rx_mount[1] - tx_pos[1]
        pz = rx_mount[2] - tx_pos[2]
        dist_sq = px * px + py * py + pz * pz
        if dist_sq <= 0.0:
            return False
        dist = dist_sq**0.5
        if dist > beam_length:
            return False
            
        tx_x, tx_y, tx_z = px / dist, py / dist, pz / dist
        ix, iy, iz = -tx_x, -tx_y, -tx_z
        
        dot_dt = dish_boresight[0] * ix + dish_boresight[1] * iy + dish_boresight[2] * iz
        if dot_dt < cos_fov:
            return False
            
        dot_bp = beam_axis[0] * tx_x + beam_axis[1] * tx_y + beam_axis[2] * tx_z
        return dot_bp >= cos_alpha

    # ==============================================================================
    # Leg target evaluation
    # ==============================================================================

    @cuda.jit(device=True)
    def get_aim_device(leg_type, leg_params, local_t, duration, step_start_aim, u_x, u_y, u_z, out_aim):
        if leg_type == 0: # Hold
            out_aim[0] = step_start_aim[0]
            out_aim[1] = step_start_aim[1]
            out_aim[2] = step_start_aim[2]
        elif leg_type == 1: # Reset
            if duration <= 0.0:
                out_aim[0] = u_z[0]
                out_aim[1] = u_z[1]
                out_aim[2] = u_z[2]
            else:
                progress = min(local_t / duration, 1.0)
                slerp_device(step_start_aim, u_z, progress, out_aim)
        elif leg_type == 2: # Spiral
            w = leg_params[0]
            k = leg_params[1]
            max_radius = leg_params[2]
            speed = leg_params[3]
            if w <= 0.0:
                out_aim[0] = u_z[0]
                out_aim[1] = u_z[1]
                out_aim[2] = u_z[2]
                return
            
            if max_radius == 0.0:
                R_start = angle_between_device(step_start_aim, u_z)
                if R_start <= 0.0 or duration <= 0.0:
                    out_aim[0] = u_z[0]
                    out_aim[1] = u_z[1]
                    out_aim[2] = u_z[2]
                    return
                t_reverse = duration - local_t
                if t_reverse < 0.0:
                    t_reverse = 0.0
                
                effective_speed = speed
                if k <= 0.0:
                    u = effective_speed * t_reverse
                else:
                    Y = (k * effective_speed * t_reverse) / w
                    if Y <= 0.0:
                        u = 0.0
                    else:
                        x = math.sqrt(2.0 * Y) if Y > 2.0 else Y
                        for _ in range(3):
                            sqrt_term = math.sqrt(1.0 + x * x)
                            h_x = 0.5 * (x * sqrt_term + math.log(x + sqrt_term))
                            diff = h_x - Y
                            x = x - diff / sqrt_term
                        u = x / k
                theta_l = w * u
                u_start = R_start / w
                phi_l = k * u_start * 2.0 - k * u
            else:
                effective_speed = speed
                
                if k <= 0.0:
                    u = effective_speed * local_t
                else:
                    Y = (k * effective_speed * local_t) / w
                    if Y <= 0.0:
                        u = 0.0
                    else:
                        x = math.sqrt(2.0 * Y) if Y > 2.0 else Y
                        for _ in range(3):
                            sqrt_term = math.sqrt(1.0 + x * x)
                            h_x = 0.5 * (x * sqrt_term + math.log(x + sqrt_term))
                            diff = h_x - Y
                            x = x - diff / sqrt_term
                        u = x / k
                if w * u > max_radius:
                    u = max_radius / w
                theta_l, phi_l = w * u, k * u
            sin_theta = math.sin(theta_l)
            cos_theta = math.cos(theta_l)
            sin_phi = math.sin(phi_l)
            cos_phi = math.cos(phi_l)
            al0 = sin_theta * cos_phi
            al1 = sin_theta * sin_phi
            al2 = cos_theta
            asx = al0 * u_x[0] + al1 * u_y[0] + al2 * u_z[0]
            asy = al0 * u_x[1] + al1 * u_y[1] + al2 * u_z[1]
            asz = al0 * u_x[2] + al1 * u_y[2] + al2 * u_z[2]
            normalize_device((asx, asy, asz), out_aim)
        elif leg_type == 3: # Raster
            radius = leg_params[0]
            steps = int(leg_params[1])
            horizontal = leg_params[2] > 0.5
            serpentine = leg_params[3] > 0.5
            if duration <= 0.0 or steps <= 1:
                normalize_device(u_z, out_aim)
                return
            p = local_t / duration
            line_progress = p * steps
            line_idx = min(int(line_progress), steps - 1)
            t_line = line_progress - line_idx
            line_offset = -radius + (line_idx / (steps - 1)) * 2.0 * radius
            chord_half_length = math.sqrt(max(0.0, radius**2 - line_offset**2))
            if serpentine and line_idx % 2 == 1:
                scan_offset = chord_half_length * (1.0 - 2.0 * t_line)
            else:
                scan_offset = chord_half_length * (2.0 * t_line - 1.0)
            if horizontal:
                u_off = scan_offset
                v_off = line_offset
            else:
                u_off = line_offset
                v_off = scan_offset
            v_x = u_z[0] + u_off * u_x[0] + v_off * u_y[0]
            v_y = u_z[1] + u_off * u_x[1] + v_off * u_y[1]
            v_z = u_z[2] + u_off * u_x[2] + v_off * u_y[2]
            normalize_device((v_x, v_y, v_z), out_aim)
        elif leg_type == 4: # Rosette
            A = leg_params[0]
            w1 = leg_params[1]
            w2 = leg_params[2]
            r = A * math.cos(w2 * local_t)
            u_off = r * math.cos(w1 * local_t)
            v_off = r * math.sin(w1 * local_t)
            v_x = u_z[0] + u_off * u_x[0] + v_off * u_y[0]
            v_y = u_z[1] + u_off * u_x[1] + v_off * u_y[1]
            v_z = u_z[2] + u_off * u_x[2] + v_off * u_y[2]
            normalize_device((v_x, v_y, v_z), out_aim)
        elif leg_type == 5: # Lissajous
            A = leg_params[0]
            wx = leg_params[1]
            wy = leg_params[2]
            delta = leg_params[3]
            u_off = A * math.sin(wx * local_t + delta)
            v_off = A * math.sin(wy * local_t)
            v_x = u_z[0] + u_off * u_x[0] + v_off * u_y[0]
            v_y = u_z[1] + u_off * u_x[1] + v_off * u_y[1]
            v_z = u_z[2] + u_off * u_x[2] + v_off * u_y[2]
            normalize_device((v_x, v_y, v_z), out_aim)

    # ==============================================================================
    # Parallel Simulation Kernel
    # ==============================================================================

    @cuda.jit
    def simulate_batch_kernel(
        offsets,        # (B * N, 4) -> s1_theta, s1_phi, s2_theta, s2_phi
        legs_s1,        # (B, M, 7) -> start, end, type, p1, p2, p3, p4
        legs_s2,        # (B, M, 7)
        hw_steps_s1,    # (B, H, 3) -> time, target_type (0=beam, 1=rx), enabled (0 or 1)
        hw_steps_s2,    # (B, H, 3)
        sim_params,     # (B, 10) -> [t_step, timeout, beam_length, body_radius, dish_fov, cos_dish_fov, max_beam_speed, max_fsm_speed, alpha, cos_alpha]
        positions,      # (B, 6) -> [s1_x, s1_y, s1_z, s2_x, s2_y, s2_z]
        results,        # Output: (B * N, 2) -> locked (1.0 or 0.0), hit_at_t
        runs_per_trial  # scalar int (N)
    ):
        idx = cuda.grid(1)
        if idx >= offsets.shape[0]:
            return

        trial_idx = idx // runs_per_trial

        # Load positions
        s1_pos = (positions[trial_idx, 0], positions[trial_idx, 1], positions[trial_idx, 2])
        s2_pos = (positions[trial_idx, 3], positions[trial_idx, 4], positions[trial_idx, 5])
        
        # Load physics constants
        t_step = sim_params[trial_idx, 0]
        timeout = sim_params[trial_idx, 1]
        beam_length = sim_params[trial_idx, 2]
        body_radius = sim_params[trial_idx, 3]
        dish_fov = sim_params[trial_idx, 4]
        cos_dish_fov = sim_params[trial_idx, 5]
        max_beam_speed = sim_params[trial_idx, 6]
        max_fsm_speed = sim_params[trial_idx, 7]
        alpha = sim_params[trial_idx, 8]
        cos_alpha = sim_params[trial_idx, 9]
        max_fsm_radius = sim_params[trial_idx, 10]


        # Scenario initial error offsets
        s1_theta_off = offsets[idx, 0]
        s1_phi_off   = offsets[idx, 1]
        s2_theta_off = offsets[idx, 2]
        s2_phi_off   = offsets[idx, 3]

        # Calculate initial pointing direction vectors
        # Toward partner nominal
        dx = s2_pos[0] - s1_pos[0]
        dy = s2_pos[1] - s1_pos[1]
        dz = s2_pos[2] - s1_pos[2]
        toward_s2 = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device((dx, dy, dz), toward_s2)
        
        toward_s1 = cuda.local.array(3, dtype=FLOAT_DTYPE)
        normalize_device((-dx, -dy, -dz), toward_s1)

        # Apply initial pointing error offsets
        s1_initial_boresight = cuda.local.array(3, dtype=FLOAT_DTYPE)
        direction_with_local_offset_device(toward_s2, s1_theta_off, s1_phi_off, s1_initial_boresight)
        
        s2_initial_boresight = cuda.local.array(3, dtype=FLOAT_DTYPE)
        direction_with_local_offset_device(toward_s1, s2_theta_off, s2_phi_off, s2_initial_boresight)

        # Precompute initial bases
        s1_u_x = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_u_y = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_u_z = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_angles = cuda.local.array(2, dtype=FLOAT_DTYPE)
        spherical_angles_from_direction_device(s1_initial_boresight, s1_angles)
        transmitter_basis_device(s1_angles[0], s1_angles[1], s1_u_x, s1_u_y, s1_u_z)

        s2_u_x = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_u_y = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_u_z = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_angles = cuda.local.array(2, dtype=FLOAT_DTYPE)
        spherical_angles_from_direction_device(s2_initial_boresight, s2_angles)
        transmitter_basis_device(s2_angles[0], s2_angles[1], s2_u_x, s2_u_y, s2_u_z)

        # Active state variables
        s1_bench_boresight = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_bench_boresight[0] = s1_initial_boresight[0]
        s1_bench_boresight[1] = s1_initial_boresight[1]
        s1_bench_boresight[2] = s1_initial_boresight[2]

        s2_bench_boresight = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_bench_boresight[0] = s2_initial_boresight[0]
        s2_bench_boresight[1] = s2_initial_boresight[1]
        s2_bench_boresight[2] = s2_initial_boresight[2]

        s1_has_seen = False
        s1_incident = 0.0
        s1_track_target = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_fsm_theta = 0.0
        s1_fsm_phi = 0.0
        s1_fsm_locked = False
        s1_slew_complete = False
        s1_bench_slew_rate = 0.0

        s2_has_seen = False
        s2_incident = 0.0
        s2_track_target = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_fsm_theta = 0.0
        s2_fsm_phi = 0.0
        s2_fsm_locked = False
        s2_slew_complete = False
        s2_bench_slew_rate = 0.0

        s1_step_start_aim = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s1_step_start_aim[0] = s1_initial_boresight[0]
        s1_step_start_aim[1] = s1_initial_boresight[1]
        s1_step_start_aim[2] = s1_initial_boresight[2]

        s2_step_start_aim = cuda.local.array(3, dtype=FLOAT_DTYPE)
        s2_step_start_aim[0] = s2_initial_boresight[0]
        s2_step_start_aim[1] = s2_initial_boresight[1]
        s2_step_start_aim[2] = s2_initial_boresight[2]

        s1_beam_enabled = True
        s1_rx_enabled = True
        s2_beam_enabled = True
        s2_rx_enabled = True

        hw_idx_s1 = 0
        hw_idx_s2 = 0

        prev_leg_idx_s1 = -1
        prev_leg_idx_s2 = -1

        local_t = 0.0
        success = False
        hit_time = -1.0

        # Phase 4: Consolidate scratch arrays at top of kernel to reduce register pressure
        aim_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        rx_aim_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        angles_temp = cuda.local.array(2, dtype=FLOAT_DTYPE)
        ux_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        uy_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        uz_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        mount_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        toward_temp = cuda.local.array(3, dtype=FLOAT_DTYPE)
        fsm_targets_temp = cuda.local.array(2, dtype=FLOAT_DTYPE)

        # Loop through simulation steps
        while local_t <= timeout + 1e-12:
            # 1. Update hardware states based on timeline
            while hw_idx_s1 < hw_steps_s1.shape[1]:
                hw_time = hw_steps_s1[trial_idx, hw_idx_s1, 0]
                if hw_time > local_t + 1e-12:
                    break
                hw_target = hw_steps_s1[trial_idx, hw_idx_s1, 1]
                hw_enabled = hw_steps_s1[trial_idx, hw_idx_s1, 2] > 0.5
                if hw_target == 0.0:
                    s1_beam_enabled = hw_enabled
                else:
                    s1_rx_enabled = hw_enabled
                hw_idx_s1 += 1

            while hw_idx_s2 < hw_steps_s2.shape[1]:
                hw_time = hw_steps_s2[trial_idx, hw_idx_s2, 0]
                if hw_time > local_t + 1e-12:
                    break
                hw_target = hw_steps_s2[trial_idx, hw_idx_s2, 1]
                hw_enabled = hw_steps_s2[trial_idx, hw_idx_s2, 2] > 0.5
                if hw_target == 0.0:
                    s2_beam_enabled = hw_enabled
                else:
                    s2_rx_enabled = hw_enabled
                hw_idx_s2 += 1

            # 2. Update pointing aims
            # S1
            if s1_has_seen:
                pass
            else:
                # Find current movement step
                leg_idx = -1
                for i in range(legs_s1.shape[1]):
                    if local_t < legs_s1[trial_idx, i, 1] - 1e-9:
                        leg_idx = i
                        break
                if leg_idx == -1:
                    leg_idx = legs_s1.shape[1] - 1
                
                # Check for leg boundary transition
                if leg_idx != prev_leg_idx_s1:
                    prev_leg_idx_s1 = leg_idx
                    s1_step_start_aim[0] = s1_bench_boresight[0]
                    s1_step_start_aim[1] = s1_bench_boresight[1]
                    s1_step_start_aim[2] = s1_bench_boresight[2]
                
                leg_start = legs_s1[trial_idx, leg_idx, 0]
                leg_duration = legs_s1[trial_idx, leg_idx, 1] - leg_start
                leg_type = int(legs_s1[trial_idx, leg_idx, 2])
                leg_params = (legs_s1[trial_idx, leg_idx, 3], legs_s1[trial_idx, leg_idx, 4], legs_s1[trial_idx, leg_idx, 5], legs_s1[trial_idx, leg_idx, 6])
                
                get_aim_device(leg_type, leg_params, local_t - leg_start, leg_duration, s1_step_start_aim, s1_u_x, s1_u_y, s1_u_z, aim_temp)
                s1_bench_boresight[0] = aim_temp[0]
                s1_bench_boresight[1] = aim_temp[1]
                s1_bench_boresight[2] = aim_temp[2]

            # S2
            if s2_has_seen:
                pass
            else:
                leg_idx = -1
                for i in range(legs_s2.shape[1]):
                    if local_t < legs_s2[trial_idx, i, 1] - 1e-9:
                        leg_idx = i
                        break
                if leg_idx == -1:
                    leg_idx = legs_s2.shape[1] - 1
                
                if leg_idx != prev_leg_idx_s2:
                    prev_leg_idx_s2 = leg_idx
                    s2_step_start_aim[0] = s2_bench_boresight[0]
                    s2_step_start_aim[1] = s2_bench_boresight[1]
                    s2_step_start_aim[2] = s2_bench_boresight[2]
                
                leg_start = legs_s2[trial_idx, leg_idx, 0]
                leg_duration = legs_s2[trial_idx, leg_idx, 1] - leg_start
                leg_type = int(legs_s2[trial_idx, leg_idx, 2])
                leg_params = (legs_s2[trial_idx, leg_idx, 3], legs_s2[trial_idx, leg_idx, 4], legs_s2[trial_idx, leg_idx, 5], legs_s2[trial_idx, leg_idx, 6])
                
                get_aim_device(leg_type, leg_params, local_t - leg_start, leg_duration, s2_step_start_aim, s2_u_x, s2_u_y, s2_u_z, aim_temp)
                s2_bench_boresight[0] = aim_temp[0]
                s2_bench_boresight[1] = aim_temp[1]
                s2_bench_boresight[2] = aim_temp[2]

            # 3. Check link visibility
            # S1 to S2 pointing
            if s1_has_seen:
                spherical_angles_from_direction_device(s1_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s1_bench_boresight, ux_temp, uy_temp, s1_fsm_theta, s1_fsm_phi, aim_temp)
            else:
                aim_temp[0] = s1_bench_boresight[0]
                aim_temp[1] = s1_bench_boresight[1]
                aim_temp[2] = s1_bench_boresight[2]
            
            if s2_has_seen:
                spherical_angles_from_direction_device(s2_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s2_bench_boresight, ux_temp, uy_temp, s2_fsm_theta, s2_fsm_phi, rx_aim_temp)
            else:
                rx_aim_temp[0] = s2_bench_boresight[0]
                rx_aim_temp[1] = s2_bench_boresight[1]
                rx_aim_temp[2] = s2_bench_boresight[2]
                
            visible_12 = s1_beam_enabled and s2_rx_enabled and beam_hits_dish_device(
                s1_pos, s2_pos, rx_aim_temp, cos_dish_fov, aim_temp, cos_alpha, beam_length, body_radius
            )

            # S2 to S1 pointing
            if s2_has_seen:
                spherical_angles_from_direction_device(s2_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s2_bench_boresight, ux_temp, uy_temp, s2_fsm_theta, s2_fsm_phi, aim_temp)
            else:
                aim_temp[0] = s2_bench_boresight[0]
                aim_temp[1] = s2_bench_boresight[1]
                aim_temp[2] = s2_bench_boresight[2]
            
            if s1_has_seen:
                spherical_angles_from_direction_device(s1_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s1_bench_boresight, ux_temp, uy_temp, s1_fsm_theta, s1_fsm_phi, rx_aim_temp)
            else:
                rx_aim_temp[0] = s1_bench_boresight[0]
                rx_aim_temp[1] = s1_bench_boresight[1]
                rx_aim_temp[2] = s1_bench_boresight[2]
                
            visible_21 = s2_beam_enabled and s1_rx_enabled and beam_hits_dish_device(
                s2_pos, s1_pos, rx_aim_temp, cos_dish_fov, aim_temp, cos_alpha, beam_length, body_radius
            )

            # 4. Acquisition logic updates
            # S1 acquisition
            s1_just_detected = False
            s1_slewed = False
            if visible_21 and not s1_has_seen:
                mount_temp[0] = s1_pos[0] + s1_bench_boresight[0] * body_radius
                mount_temp[1] = s1_pos[1] + s1_bench_boresight[1] * body_radius
                mount_temp[2] = s1_pos[2] + s1_bench_boresight[2] * body_radius
                
                dx_m = s2_pos[0] - mount_temp[0]
                dy_m = s2_pos[1] - mount_temp[1]
                dz_m = s2_pos[2] - mount_temp[2]
                normalize_device((dx_m, dy_m, dz_m), toward_temp)
                
                s1_incident = angle_between_device(s1_bench_boresight, toward_temp)
                s1_track_target[0] = toward_temp[0]
                s1_track_target[1] = toward_temp[1]
                s1_track_target[2] = toward_temp[2]
                
                s1_has_seen = True
                s1_fsm_locked = False
                s1_slew_complete = False
                s1_bench_slew_rate = max_beam_speed
                
                offsets_to_target_device(s1_bench_boresight, toward_temp, fsm_targets_temp)
                target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                if target_dist > max_fsm_radius:
                    s1_fsm_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                    s1_fsm_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                    s1_fsm_locked = False
                else:
                    s1_fsm_theta = fsm_targets_temp[0]
                    s1_fsm_phi = fsm_targets_temp[1]
                    s1_fsm_locked = True
                
                if max_beam_speed == 0.0:
                    s1_bench_boresight[0] = toward_temp[0]
                    s1_bench_boresight[1] = toward_temp[1]
                    s1_bench_boresight[2] = toward_temp[2]
                    s1_slew_complete = True
                s1_just_detected = True

            # S2 acquisition
            s2_just_detected = False
            s2_slewed = False
            if visible_12 and not s2_has_seen:
                mount_temp[0] = s2_pos[0] + s2_bench_boresight[0] * body_radius
                mount_temp[1] = s2_pos[1] + s2_bench_boresight[1] * body_radius
                mount_temp[2] = s2_pos[2] + s2_bench_boresight[2] * body_radius
                
                dx_m = s1_pos[0] - mount_temp[0]
                dy_m = s1_pos[1] - mount_temp[1]
                dz_m = s1_pos[2] - mount_temp[2]
                normalize_device((dx_m, dy_m, dz_m), toward_temp)
                
                s2_incident = angle_between_device(s2_bench_boresight, toward_temp)
                s2_track_target[0] = toward_temp[0]
                s2_track_target[1] = toward_temp[1]
                s2_track_target[2] = toward_temp[2]
                
                s2_has_seen = True
                s2_fsm_locked = False
                s2_slew_complete = False
                s2_bench_slew_rate = max_beam_speed
                
                offsets_to_target_device(s2_bench_boresight, toward_temp, fsm_targets_temp)
                target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                if target_dist > max_fsm_radius:
                    s2_fsm_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                    s2_fsm_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                    s2_fsm_locked = False
                else:
                    s2_fsm_theta = fsm_targets_temp[0]
                    s2_fsm_phi = fsm_targets_temp[1]
                    s2_fsm_locked = True
                
                if max_beam_speed == 0.0:
                    s2_bench_boresight[0] = toward_temp[0]
                    s2_bench_boresight[1] = toward_temp[1]
                    s2_bench_boresight[2] = toward_temp[2]
                    s2_slew_complete = True
                s2_just_detected = True

            # 5. Tracking logic updates (slews)
            # S1 Slews
            if s1_has_seen:
                offsets_to_target_device(s1_bench_boresight, s1_track_target, fsm_targets_temp)
                target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                clamped_target_theta = fsm_targets_temp[0]
                clamped_target_phi = fsm_targets_temp[1]
                if target_dist > max_fsm_radius:
                    clamped_target_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                    clamped_target_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                
                if max_fsm_speed <= 0.0:
                    s1_fsm_theta = clamped_target_theta
                    s1_fsm_phi = clamped_target_phi
                    s1_fsm_locked = (target_dist <= max_fsm_radius + 1e-12)
                else:
                    du = clamped_target_theta - s1_fsm_theta
                    dv = clamped_target_phi - s1_fsm_phi
                    dist = (du*du + dv*dv) ** 0.5
                    max_step = max_fsm_speed * t_step
                    if dist <= max_step + 1e-12:
                        s1_fsm_theta = clamped_target_theta
                        s1_fsm_phi = clamped_target_phi
                        s1_fsm_locked = (target_dist <= max_fsm_radius + 1e-12)
                    else:
                        s1_fsm_theta += (du / dist) * max_step
                        s1_fsm_phi += (dv / dist) * max_step
                        s1_fsm_locked = False

                if not s1_just_detected and not s1_slew_complete:
                    max_step = s1_bench_slew_rate * t_step
                    remaining = angle_between_device(s1_bench_boresight, s1_track_target)
                    rotate_toward_device(s1_bench_boresight, s1_track_target, max_step, aim_temp)
                    s1_bench_boresight[0] = aim_temp[0]
                    s1_bench_boresight[1] = aim_temp[1]
                    s1_bench_boresight[2] = aim_temp[2]
                    
                    offsets_to_target_device(s1_bench_boresight, s1_track_target, fsm_targets_temp)
                    target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                    if target_dist > max_fsm_radius:
                        s1_fsm_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                        s1_fsm_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                        s1_fsm_locked = False
                    else:
                        s1_fsm_theta = fsm_targets_temp[0]
                        s1_fsm_phi = fsm_targets_temp[1]
                        s1_fsm_locked = True
                    
                    if remaining <= max_step + 1e-12:
                        s1_bench_boresight[0] = s1_track_target[0]
                        s1_bench_boresight[1] = s1_track_target[1]
                        s1_bench_boresight[2] = s1_track_target[2]
                        s1_slew_complete = True
                    s1_slewed = True

            # S2 Slews
            if s2_has_seen:
                offsets_to_target_device(s2_bench_boresight, s2_track_target, fsm_targets_temp)
                target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                clamped_target_theta = fsm_targets_temp[0]
                clamped_target_phi = fsm_targets_temp[1]
                if target_dist > max_fsm_radius:
                    clamped_target_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                    clamped_target_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                
                if max_fsm_speed <= 0.0:
                    s2_fsm_theta = clamped_target_theta
                    s2_fsm_phi = clamped_target_phi
                    s2_fsm_locked = (target_dist <= max_fsm_radius + 1e-12)
                else:
                    du = clamped_target_theta - s2_fsm_theta
                    dv = clamped_target_phi - s2_fsm_phi
                    dist = (du*du + dv*dv) ** 0.5
                    max_step = max_fsm_speed * t_step
                    if dist <= max_step + 1e-12:
                        s2_fsm_theta = clamped_target_theta
                        s2_fsm_phi = clamped_target_phi
                        s2_fsm_locked = (target_dist <= max_fsm_radius + 1e-12)
                    else:
                        s2_fsm_theta += (du / dist) * max_step
                        s2_fsm_phi += (dv / dist) * max_step
                        s2_fsm_locked = False

                if not s2_just_detected and not s2_slew_complete:
                    max_step = s2_bench_slew_rate * t_step
                    remaining = angle_between_device(s2_bench_boresight, s2_track_target)
                    rotate_toward_device(s2_bench_boresight, s2_track_target, max_step, aim_temp)
                    s2_bench_boresight[0] = aim_temp[0]
                    s2_bench_boresight[1] = aim_temp[1]
                    s2_bench_boresight[2] = aim_temp[2]
                    
                    offsets_to_target_device(s2_bench_boresight, s2_track_target, fsm_targets_temp)
                    target_dist = (fsm_targets_temp[0]*fsm_targets_temp[0] + fsm_targets_temp[1]*fsm_targets_temp[1]) ** 0.5
                    if target_dist > max_fsm_radius:
                        s2_fsm_theta = (fsm_targets_temp[0] / target_dist) * max_fsm_radius
                        s2_fsm_phi = (fsm_targets_temp[1] / target_dist) * max_fsm_radius
                        s2_fsm_locked = False
                    else:
                        s2_fsm_theta = fsm_targets_temp[0]
                        s2_fsm_phi = fsm_targets_temp[1]
                        s2_fsm_locked = True
                    
                    if remaining <= max_step + 1e-12:
                        s2_bench_boresight[0] = s2_track_target[0]
                        s2_bench_boresight[1] = s2_track_target[1]
                        s2_bench_boresight[2] = s2_track_target[2]
                        s2_slew_complete = True
                    s2_slewed = True

            # 6. Evaluate final lock state at this step
            if s1_has_seen and s2_has_seen:
                s1_final_aim = s1_bench_boresight
                s2_final_aim = s2_bench_boresight
                
                spherical_angles_from_direction_device(s2_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s2_bench_boresight, ux_temp, uy_temp, s2_fsm_theta, s2_fsm_phi, rx_aim_temp)
                
                final_12 = s1_beam_enabled and s2_rx_enabled and beam_hits_dish_device(
                    s1_pos, s2_pos, rx_aim_temp, cos_dish_fov, s1_final_aim, cos_alpha, beam_length, body_radius
                )
                
                spherical_angles_from_direction_device(s1_bench_boresight, angles_temp)
                transmitter_basis_device(angles_temp[0], angles_temp[1], ux_temp, uy_temp, uz_temp)
                direction_with_tangent_offset_device(s1_bench_boresight, ux_temp, uy_temp, s1_fsm_theta, s1_fsm_phi, rx_aim_temp)
                
                final_21 = s2_beam_enabled and s1_rx_enabled and beam_hits_dish_device(
                    s2_pos, s1_pos, rx_aim_temp, cos_dish_fov, s2_final_aim, cos_alpha, beam_length, body_radius
                )

                if final_12 and final_21 and s1_slew_complete and s2_slew_complete:
                    success = True
                    hit_time = local_t
                    break

            local_t += t_step

        results[idx, 0] = 1.0 if success else 0.0
        results[idx, 1] = hit_time

# ==============================================================================
# Helper to check if a strategy chain is supported on GPU
# ==============================================================================

_gpu_compat_cache: dict[tuple[str, ...], bool] = {}

def is_strategy_chain_supported_on_gpu(mc: MonteCarloConfig) -> bool:
    """Check if all movement patterns in the config strategy chain are GPU-compatible."""
    import os
    if os.environ.get("SATELLITE_NO_GPU") == "1":
        return False
    if not CUDA_AVAILABLE:
        return False
        
    cache_key = mc.chain
    if cache_key in _gpu_compat_cache:
        return _gpu_compat_cache[cache_key]

    sim = load_simulation_config(mc.simulation_path)
    from satellite.sda.satellite import Satellite
    dummy_pos = np.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    partner_pos = np.array([1000.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    
    from satellite.config import BenchOffsetConfig
    dummy_offset = BenchOffsetConfig(0.0, 0.0)
    mock_config = build_scenario_config(sim, ScenarioInstance("dummy_s", dummy_offset, dummy_offset, overrides=getattr(mc, "overrides", {})), strategy=mc.strategy)
    
    s1 = Satellite.build("S1", mock_config.s1, partner_pos, mock_config)
    s2 = Satellite.build("S2", mock_config.s2, dummy_pos, mock_config)
    ctx = StrategyContext(s1=s1, s2=s2, config=mock_config)
    
    meta = MetaStrategy.from_config(mock_config)
    for strategy in meta.strategies:
        script = strategy.build_script(ctx)
        for step in script.s1.movement_steps:
            if not isinstance(step.movement, (Hold, Reset, Spiral, SerpentineRaster, Rosette, Lissajous)):
                _gpu_compat_cache[cache_key] = False
                return False
        for step in script.s2.movement_steps:
            if not isinstance(step.movement, (Hold, Reset, Spiral, SerpentineRaster, Rosette, Lissajous)):
                _gpu_compat_cache[cache_key] = False
                return False
                
    _gpu_compat_cache[cache_key] = True
    return True

# ==============================================================================
# GPU Orchestration Runner
# ==============================================================================

def run_monte_carlo_cuda(mc: MonteCarloConfig) -> MonteCarloSummary:
    """Launch Monte Carlo simulation batch on the GPU using Numba CUDA."""
    return run_monte_carlo_cuda_batch([mc])[0]

def run_monte_carlo_cuda_batch(configs: list[MonteCarloConfig]) -> list[MonteCarloSummary]:
    """Launch multiple Monte Carlo configs in a single GPU kernel batch."""
    import time
    
    for c in configs:
        if not c.chain:
            raise ValueError("monte_carlo.chain is required to run a Monte Carlo simulation")

    B = len(configs)
    if B == 0:
        return []
        
    mc = configs[0]
    sim = load_simulation_config(mc.simulation_path)
    
    from satellite.sda.satellite import Satellite
    from satellite.config import BenchOffsetConfig
    
    dummy_pos = np.array([0.0, 0.0, 0.0], dtype=FLOAT_DTYPE)
    partner_pos = np.array([sim.simulation.distance, 0.0, 0.0], dtype=FLOAT_DTYPE)
    dummy_offset = BenchOffsetConfig(0.0, 0.0)
    
    all_legs_s1 = []
    all_legs_s2 = []
    all_hw_s1 = []
    all_hw_s2 = []
    all_global_t_starts = []
    
    for mc_cfg in configs:
        mock_config = build_scenario_config(sim, ScenarioInstance("dummy_s", dummy_offset, dummy_offset, overrides=getattr(mc_cfg, "overrides", {})), strategy=mc_cfg.strategy)
        s1 = Satellite.build("S1", mock_config.s1, partner_pos, mock_config)
        s2 = Satellite.build("S2", mock_config.s2, dummy_pos, mock_config)
        ctx = StrategyContext(s1=s1, s2=s2, config=mock_config)
        meta = MetaStrategy.from_config(mock_config)
        
        legs_s1_list = []
        legs_s2_list = []
        hw_s1_list = []
        hw_s2_list = []
        
        global_t_start = 0.0
        for strategy in meta.strategies:
            script = strategy.build_script(ctx)
            
            for step in script.s1.movement_steps:
                t_type = 0
                p = [0.0, 0.0, 0.0, 0.0]
                mv = step.movement
                if isinstance(mv, Hold):
                    t_type = 0
                elif isinstance(mv, Reset):
                    t_type = 1
                elif isinstance(mv, Spiral):
                    t_type = 2
                    p = [mv.w, mv.k, mv.max_radius, s1.bench.max_beam_speed]
                elif isinstance(mv, SerpentineRaster):
                    t_type = 3
                    p = [mv.radius, float(mv.steps), 1.0 if mv.horizontal else 0.0, 1.0]
                elif isinstance(mv, Rosette):
                    t_type = 4
                    p = [mv.A, mv.w1, mv.w2, 0.0]
                elif isinstance(mv, Lissajous):
                    t_type = 5
                    p = [mv.A, mv.wx, mv.wy, mv.delta]
                
                legs_s1_list.append([global_t_start + step.start, global_t_start + step.end, float(t_type), p[0], p[1], p[2], p[3]])
                
            for step in script.s2.movement_steps:
                t_type = 0
                p = [0.0, 0.0, 0.0, 0.0]
                mv = step.movement
                if isinstance(mv, Hold):
                    t_type = 0
                elif isinstance(mv, Reset):
                    t_type = 1
                elif isinstance(mv, Spiral):
                    t_type = 2
                    p = [mv.w, mv.k, mv.max_radius, s2.bench.max_beam_speed]
                elif isinstance(mv, SerpentineRaster):
                    t_type = 3
                    p = [mv.radius, float(mv.steps), 1.0 if mv.horizontal else 0.0, 1.0]
                elif isinstance(mv, Rosette):
                    t_type = 4
                    p = [mv.A, mv.w1, mv.w2, 0.0]
                elif isinstance(mv, Lissajous):
                    t_type = 5
                    p = [mv.A, mv.wx, mv.wy, mv.delta]
                
                legs_s2_list.append([global_t_start + step.start, global_t_start + step.end, float(t_type), p[0], p[1], p[2], p[3]])

            for step in script.s1.hardware_steps:
                target_type = 0 if step.target == "beam" else 1
                hw_s1_list.append([global_t_start + step.time, float(target_type), 1.0 if step.enabled else 0.0])
                
            for step in script.s2.hardware_steps:
                target_type = 0 if step.target == "beam" else 1
                hw_s2_list.append([global_t_start + step.time, float(target_type), 1.0 if step.enabled else 0.0])
                
            global_t_start += script.total_duration

        if not hw_s1_list:
            hw_s1_list.append([0.0, 0.0, 1.0])
        if not hw_s2_list:
            hw_s2_list.append([0.0, 0.0, 1.0])
            
        all_legs_s1.append(legs_s1_list)
        all_legs_s2.append(legs_s2_list)
        all_hw_s1.append(hw_s1_list)
        all_hw_s2.append(hw_s2_list)
        all_global_t_starts.append(global_t_start)

    max_legs_s1 = max(len(l) for l in all_legs_s1)
    max_legs_s2 = max(len(l) for l in all_legs_s2)
    max_hw_s1 = max(len(l) for l in all_hw_s1)
    max_hw_s2 = max(len(l) for l in all_hw_s2)
    
    legs_s1_arr = np.zeros((B, max_legs_s1, 7), dtype=FLOAT_DTYPE)
    legs_s2_arr = np.zeros((B, max_legs_s2, 7), dtype=FLOAT_DTYPE)
    hw_s1_arr = np.zeros((B, max_hw_s1, 3), dtype=FLOAT_DTYPE)
    hw_s2_arr = np.zeros((B, max_hw_s2, 3), dtype=FLOAT_DTYPE)
    
    for b in range(B):
        l_s1 = all_legs_s1[b]
        legs_s1_arr[b, :len(l_s1), :] = l_s1
        if len(l_s1) < max_legs_s1:
            legs_s1_arr[b, len(l_s1):, :] = l_s1[-1]
            
        l_s2 = all_legs_s2[b]
        legs_s2_arr[b, :len(l_s2), :] = l_s2
        if len(l_s2) < max_legs_s2:
            legs_s2_arr[b, len(l_s2):, :] = l_s2[-1]
            
        h_s1 = all_hw_s1[b]
        hw_s1_arr[b, :len(h_s1), :] = h_s1
        if len(h_s1) < max_hw_s1:
            hw_s1_arr[b, len(h_s1):, 0] = all_global_t_starts[b] + 1.0
            
        h_s2 = all_hw_s2[b]
        hw_s2_arr[b, :len(h_s2), :] = h_s2
        if len(h_s2) < max_hw_s2:
            hw_s2_arr[b, len(h_s2):, 0] = all_global_t_starts[b] + 1.0

    for b in range(B):
        hw_s1_arr[b] = hw_s1_arr[b, np.argsort(hw_s1_arr[b, :, 0])]
        hw_s2_arr[b] = hw_s2_arr[b, np.argsort(hw_s2_arr[b, :, 0])]

    sim_params_arr = np.zeros((B, 11), dtype=FLOAT_DTYPE)
    positions_arr = np.zeros((B, 6), dtype=FLOAT_DTYPE)
    
    runs_per_config = configs[0].runs
    total_runs = B * runs_per_config
    offsets_arr = np.zeros((total_runs, 4), dtype=FLOAT_DTYPE)
    
    for b, mc_cfg in enumerate(configs):
        mock_config = build_scenario_config(sim, ScenarioInstance("dummy_s", dummy_offset, dummy_offset), strategy=mc_cfg.strategy)
        s1 = Satellite.build("S1", mock_config.s1, partner_pos, mock_config)
        s2 = Satellite.build("S2", mock_config.s2, dummy_pos, mock_config)
        
        beam_length = mock_config.simulation.beam_length or (sim.simulation.distance + mock_config.simulation.boresight_extension)
        sim_params_arr[b] = [
            mock_config.simulation.t_step,
            all_global_t_starts[b],
            beam_length,
            mock_config.satellite.body_radius,
            mock_config.satellite.dish_fov,
            float(np.cos(mock_config.satellite.dish_fov)),
            mock_config.satellite.max_beam_speed,
            mock_config.satellite.max_fsm_speed,
            mock_config.satellite.alpha,
            float(np.cos(mock_config.satellite.alpha)),
            mock_config.satellite.max_fsm_radius,
        ]
        positions_arr[b] = [
            s1.position[0], s1.position[1], s1.position[2],
            s2.position[0], s2.position[1], s2.position[2]
        ]
        
        chain_rng = np.random.default_rng(mc_cfg.seed)
        seeds = chain_rng.integers(0, 2**32 - 1, size=runs_per_config).tolist()
        for r in range(runs_per_config):
            rng = np.random.default_rng(seeds[r])
            s1_off, s2_off = sample_offsets(mc_cfg.error, rng)
            offsets_arr[b * runs_per_config + r] = [
                s1_off.bench_theta_offset,
                s1_off.bench_phi_offset,
                s2_off.bench_theta_offset,
                s2_off.bench_phi_offset
            ]

    d_offsets = cuda.to_device(offsets_arr)
    d_legs_s1 = cuda.to_device(legs_s1_arr)
    d_legs_s2 = cuda.to_device(legs_s2_arr)
    d_hw_steps_s1 = cuda.to_device(hw_s1_arr)
    d_hw_steps_s2 = cuda.to_device(hw_s2_arr)
    d_sim_params = cuda.to_device(sim_params_arr)
    d_positions = cuda.to_device(positions_arr)
    
    d_results = cuda.device_array((total_runs, 2), dtype=FLOAT_DTYPE)

    threads_per_block = 256
    blocks_per_grid = (total_runs + (threads_per_block - 1)) // threads_per_block

    t0 = time.perf_counter()
    simulate_batch_kernel[blocks_per_grid, threads_per_block](
        d_offsets, d_legs_s1, d_legs_s2, d_hw_steps_s1, d_hw_steps_s2, d_sim_params, d_positions, d_results, runs_per_config
    )
    cuda.synchronize()
    gpu_time_ms = (time.perf_counter() - t0) * 1000.0

    results_arr = d_results.copy_to_host()

    from satellite.config import positions_for_distance, ScenarioConfig, SatelliteInstanceConfig, StrategyConfig
    s1_pos, s2_pos = positions_for_distance(sim.simulation.distance)
    
    summaries = []
    for b, mc_cfg in enumerate(configs):
        mock_config = build_scenario_config(sim, ScenarioInstance("dummy_s", dummy_offset, dummy_offset), strategy=mc_cfg.strategy)
        s1 = Satellite.build("S1", mock_config.s1, partner_pos, mock_config)
        s2 = Satellite.build("S2", mock_config.s2, dummy_pos, mock_config)
        ctx = StrategyContext(s1=s1, s2=s2, config=mock_config)
        meta = MetaStrategy.from_config(mock_config)
        
        # Precompute strategy durations
        strategy_durations = []
        for strategy in meta.strategies:
            strategy_durations.append((strategy.name, strategy.build_script(ctx).total_duration))
            
        mock_schedule = _MockSchedule(total_duration=all_global_t_starts[b])
        
        # Precompute run strategy configs
        run_strategy_configs = []
        run_strat = StrategyConfig(
            k=mc_cfg.strategy.k,
            chain=mc_cfg.chain,
            params=mc_cfg.strategy.params,
        )
        for _ in range(runs_per_config):
            run_strategy_configs.append(run_strat)
                
        run_results = []
        for r in range(runs_per_config):
            idx = b * runs_per_config + r
            locked = results_arr[idx, 0] > 0.5
            hit_at_t = float(results_arr[idx, 1]) if locked else None
            
            winning_strat_name = None
            if locked:
                accum_t = 0.0
                for strat_name, duration in strategy_durations:
                    if hit_at_t <= accum_t + duration + 1e-9:
                        winning_strat_name = strat_name
                        break
                    accum_t += duration

            run_strategy_config = run_strategy_configs[r]
            
            s1_theta = float(offsets_arr[idx, 0])
            s1_phi   = float(offsets_arr[idx, 1])
            s2_theta = float(offsets_arr[idx, 2])
            s2_phi   = float(offsets_arr[idx, 3])
            
            run_config = ScenarioConfig(
                name=f"mc_run_{r}",
                s1=SatelliteInstanceConfig(
                    position=s1_pos,
                    bench_theta_offset=s1_theta,
                    bench_phi_offset=s1_phi,
                ),
                s2=SatelliteInstanceConfig(
                    position=s2_pos,
                    bench_theta_offset=s2_theta,
                    bench_phi_offset=s2_phi,
                ),
                satellite=sim.satellite,
                simulation=sim.simulation,
                three_d_viz=sim.three_d_viz,
                map_viz=sim.map_viz,
                strategy=run_strategy_config,
            )
            
            mock_result = _MockScenarioResult(
                success=locked,
                hit_at_t=hit_at_t,
                strategy_name=winning_strat_name,
                schedule=mock_schedule,
                config=run_config
            )
            
            run_result = MonteCarloRunResult(
                run_index=r,
                s1_theta=s1_theta,
                s1_phi=s1_phi,
                s2_theta=s2_theta,
                s2_phi=s2_phi,
                result=mock_result,
                computation_time_ms=gpu_time_ms / total_runs,
            )
            run_results.append(run_result)

        summary = _build_monte_carlo_summary(mc_cfg, run_results, interrupted=False)
        summaries.append(summary)

    return summaries

