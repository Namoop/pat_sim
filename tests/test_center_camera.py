"""Center camera pose for 3D visualization."""

from __future__ import annotations

import numpy as np

from satellite.visualize.panels.view3d import center_camera_pose


def test_center_camera_focal_point_is_midpoint():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1000.0, 0.0, 0.0])
    pose = center_camera_pose(p1, p2, body_radius=0.5)
    np.testing.assert_allclose(pose.focal_point, (500.0, 0.0, 0.0))


def test_center_camera_distance_fits_both_satellites():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1000.0, 0.0, 0.0])
    body_radius = 0.5
    pose = center_camera_pose(p1, p2, body_radius=body_radius)

    focal = np.asarray(pose.focal_point)
    position = np.asarray(pose.position)
    to_camera = position - focal
    distance = float(np.linalg.norm(to_camera))
    look_dir = (focal - position) / np.linalg.norm(focal - position)
    half_fov = np.radians(pose.view_angle / 2.0)
    for sat in (p1, p2):
        to_sat = sat - position
        angle = np.arccos(
            np.clip(np.dot(look_dir, to_sat) / np.linalg.norm(to_sat), -1.0, 1.0)
        )
        assert angle <= half_fov + np.radians(4.5)

    half_extent = 500.0 + body_radius
    fov_rad = np.radians(pose.view_angle)
    tight_distance = half_extent / np.tan(fov_rad / 2.0)
    assert distance < tight_distance
