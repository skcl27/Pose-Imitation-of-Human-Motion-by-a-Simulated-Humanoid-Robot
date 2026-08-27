from __future__ import annotations

import math

from src.retargeting.mapper import RetargetingMapper, default_joint_limits
from src.type_defs import Keypoint, PoseFrame


def test_mapper_outputs_expected_joints() -> None:
    # Camera-frame mm coordinates (x right, y down, z forward/away) -- a
    # person standing with arms hanging, facing the camera square-on (z=0
    # for every landmark, so this also exercises the z==0 edge case of
    # ``_horizontal_extent``).
    pose = PoseFrame(
        timestamp_s=0.0,
        frame_index=0,
        keypoints={
            "left_shoulder": Keypoint(-100.0, -600.0, 0.0),
            "right_shoulder": Keypoint(100.0, -600.0, 0.0),
            "left_elbow": Keypoint(-150.0, -400.0, 0.0),
            "right_elbow": Keypoint(150.0, -400.0, 0.0),
            "left_wrist": Keypoint(-200.0, -200.0, 0.0),
            "right_wrist": Keypoint(200.0, -200.0, 0.0),
            "left_hip": Keypoint(-50.0, -200.0, 0.0),
            "right_hip": Keypoint(50.0, -200.0, 0.0),
            "left_knee": Keypoint(-50.0, 200.0, 0.0),
            "right_knee": Keypoint(50.0, 200.0, 0.0),
        },
    )

    mapper = RetargetingMapper(default_joint_limits())
    out = mapper.map_pose(pose)

    assert set(out.joint_angles_rad.keys()) == {
        "LShoulderPitch",
        "RShoulderPitch",
        "LElbowRoll",
        "RElbowRoll",
        "LHipPitch",
        "RHipPitch",
        "TorsoPitch",
    }


def test_mapper_uses_depth_not_just_lateral_extent() -> None:
    """Unlike the old MediaPipe-era mapper (which ignored z entirely, since
    MediaPipe's depth was unreliable), the horizontal extent of a limb should
    now account for real forward/backward motion too: an elbow raised the
    same amount but reaching FORWARD instead of SIDEWAYS should read as the
    same shoulder pitch, not a much steeper one."""
    base = {
        "right_shoulder": Keypoint(100.0, -600.0, 0.0),
        "right_elbow": Keypoint(150.0, -400.0, 0.0),
        "right_wrist": Keypoint(200.0, -200.0, 0.0),
        "left_hip": Keypoint(-50.0, -200.0, 0.0),
        "right_hip": Keypoint(50.0, -200.0, 0.0),
        "left_knee": Keypoint(-50.0, 200.0, 0.0),
        "right_knee": Keypoint(50.0, 200.0, 0.0),
    }
    sideways = PoseFrame(
        timestamp_s=0.0, frame_index=0,
        keypoints={
            **base,
            "left_shoulder": Keypoint(-100.0, -600.0, 0.0),
            "left_elbow": Keypoint(-250.0, -500.0, 0.0),   # dx=-150, dy=+100, dz=0
        },
    )
    forward = PoseFrame(
        timestamp_s=0.0, frame_index=0,
        keypoints={
            **base,
            "left_shoulder": Keypoint(-100.0, -600.0, 0.0),
            "left_elbow": Keypoint(-100.0, -500.0, 150.0),  # dx=0, dy=+100, dz=150
        },
    )
    mapper = RetargetingMapper(default_joint_limits())
    pitch_sideways = mapper.map_pose(sideways).joint_angles_rad["LShoulderPitch"]
    pitch_forward = mapper.map_pose(forward).joint_angles_rad["LShoulderPitch"]
    assert abs(pitch_sideways - pitch_forward) < 1e-4


def test_mapper_clips_joint_limits() -> None:
    pose = PoseFrame(
        timestamp_s=0.0,
        frame_index=1,
        keypoints={
            "left_shoulder": Keypoint(0.0, 0.0, 0.0),
            "right_shoulder": Keypoint(0.0, 0.0, 0.0),
            "left_elbow": Keypoint(0.0, -400.0, 0.0),
            "right_elbow": Keypoint(0.0, -400.0, 0.0),
            "left_wrist": Keypoint(0.0, -900.0, 0.0),
            "right_wrist": Keypoint(0.0, -900.0, 0.0),
            "left_hip": Keypoint(0.0, 300.0, 0.0),
            "right_hip": Keypoint(0.0, 300.0, 0.0),
            "left_knee": Keypoint(0.0, 1000.0, 0.0),
            "right_knee": Keypoint(0.0, 1000.0, 0.0),
        },
    )

    mapper = RetargetingMapper(default_joint_limits())
    out = mapper.map_pose(pose)
    limits = default_joint_limits()

    for joint, angle in out.joint_angles_rad.items():
        assert limits[joint].min_rad <= angle <= limits[joint].max_rad

    assert math.isfinite(out.joint_angles_rad["TorsoPitch"])
