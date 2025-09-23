"""
Robot configuration utilities for MoMaGen.
"""
import torch as th
from omnigibson.robots.r1 import R1
from omnigibson.robots.tiago import Tiago

# Robot types
ROBOT_R1 = "R1"
ROBOT_TIAGO = "Tiago"

# Default robot type
DEFAULT_ROBOT_TYPE = ROBOT_R1

# Robot-specific link names
ROBOT_LINK_NAMES = {
    ROBOT_R1: {
        "torso": "torso_link4"
    },
    ROBOT_TIAGO: {
        "torso": "torso_lift_link"
    }
}

def get_torso_link_name(robot):
    """Get the torso link name for a given robot instance."""
    if isinstance(robot, Tiago):
        return ROBOT_LINK_NAMES[ROBOT_TIAGO]["torso"]
    elif isinstance(robot, R1):
        return ROBOT_LINK_NAMES[ROBOT_R1]["torso"]
    else:
        raise ValueError(f"Robot type {type(robot)} not supported")

def get_robot_type_from_instance(robot):
    """Get robot type string from robot instance."""
    if isinstance(robot, Tiago):
        return ROBOT_TIAGO
    elif isinstance(robot, R1):
        return ROBOT_R1
    else:
        raise ValueError(f"Robot type {type(robot)} not supported")

def get_tiago_config():
    """Get Tiago robot configuration."""
    reset_joint_pos = th.tensor([
        0.0000,  0.0000,  0.0003,  0.0000,  0.0000,
       -0.0000,  0.3500,  0.8637,      0.8401,      0.0000,
        -0.8935,     -0.8862,     -0.4500,      1.8286,      1.8267,
         1.1199,      1.1741,      1.1771,      1.1749,     -1.4134,
        -1.2823, -1.0891, -1.0891,  0.0450,  0.0450,
        0.0450,  0.0450
    ])

    controller_config = {
        'arm_left': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'arm_right': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'gripper_left': {
            'name': 'MultiFingerGripperController',
            'mode': 'smooth',
            'command_input_limits': 'default',
            'command_output_limits': 'default'
        },
        'gripper_right': {
            'name': 'MultiFingerGripperController',
            'mode': 'smooth',
            'command_input_limits': 'default',
            'command_output_limits': 'default'
        },
        'base': {
            'name': 'HolonomicBaseJointController',
            'motor_type': 'velocity',
            'vel_kp': 150,
            'command_input_limits': [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            'command_output_limits': [[-1.5, -1.5, -3.1415927], [1.5, 1.5, 3.1415927]],
            'use_impedances': False
        },
        'trunk': {
            'name': 'JointController',
            'motor_type': 'position',
            'pos_kp': 150,
            'command_input_limits': None,
            'command_output_limits': None,
            'use_impedances': False,
            'use_delta_commands': False
        },
        'camera': {
            'name': 'JointController',
            'motor_type': 'position',
            'use_impedances': False,
            'use_delta_commands': False
        }
    }

    return reset_joint_pos, controller_config

def configure_tiago_env_meta(env_meta):
    """Configure environment metadata for Tiago robot."""
    env_meta["env_kwargs"]["robots"][0]["type"] = "Tiago"
    reset_joint_pos, controller_config = get_tiago_config()
    env_meta["env_kwargs"]["robots"][0]["reset_joint_pos"] = reset_joint_pos.tolist()
    env_meta["env_kwargs"]["robots"][0]["controller_config"] = controller_config

    if env_meta["env_kwargs"]["scene"]["scene_model"] == "house_single_floor":
        env_meta["env_kwargs"]["scene"]["load_room_types"] = ["kitchen"]

    return env_meta