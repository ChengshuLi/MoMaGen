"""
A collection of classes used to represent waypoints and trajectories.
"""
import json
import time
import numpy as np
import copy
from copy import deepcopy

import momagen.utils.pose_utils as PoseUtils

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
import torch as th
import omnigibson as og
from omnigibson.robots.r1 import R1
from omnigibson.robots.tiago import Tiago

from scipy.spatial.transform import Rotation as R

class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (np.array): 4x4 pose target for robot controller
            gripper_action (np.array): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1
    
    def merge_wp(self, other):
        """
        Merge another Waypoint object into this one.
        """
        self.pose = np.concatenate([self.pose, other.pose], axis=0)
        self.gripper_action = np.concatenate([self.gripper_action, other.gripper_action], axis=0)
        self.noise = min(self.noise, other.noise)
        # TODO: the noise here is set to 0, can be change to help reduce the sim to real gap due to the sensor observation noises
        self.noise = 0.0


class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be an list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses, 
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a 
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        # handle scalar to numpy array conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)

    def merge(self, other):
        """
        Merge another WaypointSequence object into this one.
        """
        self.sequence += other.sequence

class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.
        
        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if (len(self.waypoint_sequences) == 0):
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([[gripper_action] for _ in range(num_steps)])
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            if last_waypoint.pose.shape[0] == 8:
                # here is when transforming the two arms altogher, should be corresponding to the bimanual-coordinated phase
                poses_left, num_steps_2_left = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[0:4, :],
                    pose_2=pose[0:4, :],
                    num_steps=num_steps,
                )
                poses_right, num_steps_2_right = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[4:, :],
                    pose_2=pose[4:, :],
                    num_steps=num_steps,
                )
                poses = np.concatenate([poses_left, poses_right], axis=1)
                assert num_steps_2_left == num_steps_2_right
                num_steps_2 = num_steps_2_left
            else:
                # suitable for single arm transformation
                poses, num_steps_2 = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose,
                    pose_2=pose,
                    num_steps=num_steps,
                )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant 
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = (need_interp or need_fixed)

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose, # 8x4
                    gripper_action=target_for_interpolation.gripper_action, #2,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                    bimanual=bimanual,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                    bimanual=bimanual,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def _pad_tensors(self, tensor1, tensor2):
        M, _ = tensor1.shape
        N, _ = tensor2.shape
        max_size = max(M, N)

        def pad_tensor(tensor, size):
            if tensor.shape[0] < size:
                last_row = tensor[-1].unsqueeze(0)  # Extract last row
                repeat_count = size - tensor.shape[0]
                padding = last_row.repeat(repeat_count, 1)  # Repeat last row
                tensor = th.cat([tensor, padding], dim=0)
            return tensor

        tensor1 = pad_tensor(tensor1, max_size)
        tensor2 = pad_tensor(tensor2, max_size)

        return tensor1, tensor2
 
    def _subsample_tensor(self, tensor, num_samples=8):
        N = tensor.shape[0]

        if N <= num_samples:
            return tensor  # If N is less than or equal to num_samples, return as is

        indices = th.linspace(0, N - 1, steps=num_samples).long()  # Evenly spaced indices
        return tensor[indices]
    
    def downsample_replay_traj(self, left_replay_waypoints, right_reaplay_waypoints, ds_ratio=1, asyn_ds_ratio=True):
        # downsample the replay waypoints to reduce the hesitation problem
        len_left_wp = len(left_replay_waypoints)
        len_right_wp = len(right_reaplay_waypoints)

        if ds_ratio == 1 or ds_ratio is None:
            print('the replay waypoints are not downsampled')
            return left_replay_waypoints, right_reaplay_waypoints
        
        # TODO: the grasping motion should not be downsampled??
        # detect whether the left and right gripper action are changing

        if asyn_ds_ratio:
            # asyn downsample the waypoints regarding the gripper actions
            print('asyn downsample the waypoints regarding the gripper actions')

            # check when grasping starts for both hands
            left_gripper_actions = np.array([waypoint.gripper_action[0] for waypoint in left_replay_waypoints])
            right_gripper_actions = np.array([waypoint.gripper_action[1] for waypoint in right_reaplay_waypoints])
            left_gripper_actions_diff = np.diff(left_gripper_actions)
            right_gripper_actions_diff = np.diff(right_gripper_actions)
            # check when the gripper actions are changing
            left_gripper_actions_diff_idx = np.where(left_gripper_actions_diff != 0)[0]
            right_gripper_actions_diff_idx = np.where(right_gripper_actions_diff != 0)[0]
            if left_gripper_actions_diff_idx.size == 0: left_gripper_actions_diff_idx = np.array([len_left_wp])
            if right_gripper_actions_diff_idx.size == 0: right_gripper_actions_diff_idx = np.array([len_left_wp])
            # get the min number of the changing points
            grasp_start_idx = np.min([left_gripper_actions_diff_idx[0], right_gripper_actions_diff_idx[0]])

            left_before_grasp_ds = left_replay_waypoints[:grasp_start_idx:ds_ratio]
            right_before_grasp_ds = right_reaplay_waypoints[:grasp_start_idx:ds_ratio]

            grasp_ds_ratio = 2
            left_after_grasp_ds = left_replay_waypoints[grasp_start_idx::grasp_ds_ratio]
            right_after_grasp_ds = right_reaplay_waypoints[grasp_start_idx::grasp_ds_ratio]

            # concatenate the downsampled waypoints
            left_reaplay_wp_ds = left_before_grasp_ds + left_after_grasp_ds
            right_reaplay_wp_ds = right_before_grasp_ds + right_after_grasp_ds

            assert len(left_reaplay_wp_ds) == len(right_reaplay_wp_ds)
            print('downsample wp from {} to {}'.format(len_left_wp, len(right_reaplay_wp_ds)))
        
        else:        
            # uniformly downsample the waypoints
            print('uniformly downsample the waypoints')
            left_reaplay_wp_ds = left_replay_waypoints[::ds_ratio]
            right_reaplay_wp_ds = right_reaplay_waypoints[::ds_ratio]
            assert len(left_reaplay_wp_ds) == len(right_reaplay_wp_ds)
            print('downsample wp from {} to {}'.format(len_left_wp, len(right_reaplay_wp_ds)))
        
        return left_reaplay_wp_ds, right_reaplay_wp_ds
    
    def setup_phase_logs(self, phase_type, baseline=None):
        current_phase_logs = dict()
        current_phase_logs["phase_type"] = phase_type
        current_phase_logs["base_sampling_time"] = dict()
        current_phase_logs["base_mp_planning_time"] = dict()
        current_phase_logs["base_mp_execution_time"] = dict()

        current_phase_logs["arm_mp_planning_time"] = dict()
        current_phase_logs["arm_mp_execution_time"] = dict()
        current_phase_logs["arm_replay_execution_time"] = dict()
        if baseline == "mimicgen":
            current_phase_logs["arm_interp_execution_time"] = dict()

        current_phase_logs["full_retract_mp_planning_time"] = dict()
        current_phase_logs["full_retract_mp_execution_time"] = dict()
        current_phase_logs["torso_retract_mp_planning_time"] = dict()
        current_phase_logs["torso_retract_mp_execution_time"] = dict()
        current_phase_logs["full_retract_mp_err"] = dict()
        current_phase_logs["torso_retract_mp_err"] = dict()

        current_phase_logs["visibility_stats"] = dict()

        return current_phase_logs
    
    def obtain_attached_object(self, env, robot):
        grasp_action = {"left": 1.0, "right": 1.0}
        attached_obj = {}
        attached_obj_scale = {}
        for local_arm_side in ["left", "right"]:  
            is_grasping = robot.is_grasping(arm=local_arm_side)
            # print("local_arm_side is_grasping: ", local_arm_side, is_grasping)
            if is_grasping == og.controllers.IsGraspingState.TRUE: 
                grasp_action[local_arm_side] = -1.0
                # Find the object that the robot is grapsing in that arm
                task_relevant_objs = env._get_task_relevant_objs()
                for task_relevant_obj in task_relevant_objs:
                    # TODO: remove the stationay object hardcoding. Make it more general
                    if all(keyword not in task_relevant_obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                        is_grasping_candidate_obj = robot.is_grasping(arm=local_arm_side, candidate_obj=task_relevant_obj)
                        # print("local_arm_side is_grasping_candidate_obj: ", local_arm_side, is_grasping_candidate_obj, task_relevant_obj.root_link.name) 
                        if is_grasping_candidate_obj == og.controllers.IsGraspingState.TRUE:
                            print(f"arm {local_arm_side} is_grasping {task_relevant_obj.root_link.name}") 
                            attached_obj[f"{local_arm_side}_eef_link"] = task_relevant_obj.root_link
                            attached_obj_scale[f"{local_arm_side}_eef_link"] = 0.9
                            # robot can only be holding one object at a time
                            break
        retval = dict(
            grasp_action=grasp_action,
            attached_obj=attached_obj,
            attached_obj_scale=attached_obj_scale,
        )
        return retval
    
    def reset_visibility_counter(self, env):
        """
        Reset the visibility counter for each sensor.
        """
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                env.num_frames_with_obj_visible[shortened_sensor_name] = 0
        env.num_frames_with_obj_visible["any"] = 0

    def check_ref_obj_visibility(self, env, obs, obs_info, ref_obj):
        any_visible = False
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                seg_instance = obs[f"{env.robot_name}::{sensor_name}::seg_instance"]
                seg_instance_info = obs_info[f"{env.robot_name}"][sensor_name]["seg_instance"]
                obj_key = next((key for key, value in seg_instance_info.items() if value == ref_obj.name), None)
                if obj_key is None:
                    count = 0
                    # if shortened_sensor_name == "eyes":
                    #     print("not found")
                else:
                    count = (seg_instance == obj_key).sum().item()
                    # if shortened_sensor_name == "eyes":
                    #     print("found")
                if count > 0:
                    env.num_frames_with_obj_visible[shortened_sensor_name] += 1
                    any_visible = True

        if any_visible:
            env.num_frames_with_obj_visible["any"] += 1

    def execute_baseline(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None,
        src_curr_phase_actions=None,
        baseline=None,
    ):
        if object_ref["arm_right"] is None:
            ref_object = object_ref["arm_left"]
        elif object_ref["arm_left"] is None:
            ref_object = object_ref["arm_right"]
        else:
            ref_object = object_ref["arm_right"]
        
        ref_obj = None
        if ref_object is not None:
            if "torso" in ref_object:
                if isinstance(env.robot, Tiago):
                    torso_link_name = "torso_lift_link"
                elif isinstance(env.robot, R1):
                    torso_link_name = "torso_link4"
                else:
                    raise ValueError("Robot type not supported")
                ref_obj = env.env.robots[0].links[torso_link_name]
            else:
                ref_obj = env.env.scene.object_registry("name", ref_object)
            print("ref_obj: ", ref_obj.name)
        robot = env.env.robots[0]
        
        # TODO: implement early stopping on 1. collision 2. attached object misatch
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            init_state = og.sim.dump_state()
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            init_global_env_step = env.global_env_step
            nav_execution_start_time = time.time()
            init_arm_left_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
            init_arm_right_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            for temp_idx, src_action in enumerate(src_curr_phase_actions):
                
                # To skip initial stationary actions during human data collection
                if env.execution_phase_ind == 0 and temp_idx < env.start_nav_step:
                    continue
                action = env.primitive._empty_action()
                action[robot.base_action_idx] = th.tensor(src_action[robot.base_action_idx], dtype=th.float32)
                action[robot.arm_action_idx["left"]] = init_arm_left_pos
                action[robot.arm_action_idx["right"]] = init_arm_right_pos
                if attached_obj["left"] is not None:
                    action[robot.gripper_action_idx["left"]] = -1
                if attached_obj["right"] is not None:
                    action[robot.gripper_action_idx["right"]] = -1
                state = env.get_state()["states"]
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=action)
                env.step(action, video_writer)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                # Check reference object visibility
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)

            # apply a zero action
            action = env.primitive._empty_action()
            action[robot.base_action_idx] = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
            action[robot.arm_action_idx["left"]] = init_arm_left_pos
            action[robot.arm_action_idx["right"]] = init_arm_right_pos
            env.step(action, video_writer)

            nav_execution_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0] = round(nav_execution_finish_time - nav_execution_start_time, 2)
            print("nav execution time: ", phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for nav_repeat {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
            print(f"Visibility stats for nav_repeat any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"])

            MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
            left_mp_ranges = [0, 0]
            right_mp_ranges = [0, 0]
            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
            # In this case MP succeeded and phase was actually executed
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results

        else:
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type, baseline=baseline)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

            
            # If baseline is mimicgen, perform interpolation + replay
            if baseline == "mimicgen":
                # ========================================= ARM INTERPOLATION START =============================================
                step_size = 0.005
                current_left_eef_pose = robot.get_eef_pose("left")
                if object_ref["arm_left"] is None:
                    poses_left = th.tensor(T.pose2mat(current_left_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_left, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_left_eef_pose),
                        pose_2=th.tensor(left_waypoint.pose[:4], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_left = th.tensor(poses_left, dtype=th.float32)

                current_right_eef_pose = robot.get_eef_pose("right")
                if object_ref["arm_right"] is None:
                    poses_right = th.tensor(T.pose2mat(current_right_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_right, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_right_eef_pose),
                        pose_2=th.tensor(right_waypoint.pose[4:], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_right = th.tensor(poses_right, dtype=th.float32)
                
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*current_left_eef_pose)
                    env.eef_current_marker_right.set_position_orientation(*current_right_eef_pose)
                    interp_target_left = T.mat2pose(poses_left[-1])
                    interp_target_right = T.mat2pose(poses_right[-1])
                    env.eef_goal_marker_left.set_position_orientation(*interp_target_left)
                    env.eef_goal_marker_right.set_position_orientation(*interp_target_right)

                
                print("len(poses_left): ", len(poses_left))
                print("len(poses_right): ", len(poses_right))
                
                # Perform padding
                if len(poses_left) < len(poses_right):
                    repeat_times = len(poses_right) - len(poses_left)
                    poses_left = th.cat((poses_left, poses_left[-1].repeat(repeat_times, 1, 1)))
                elif len(poses_right) < len(poses_left):
                    repeat_times = len(poses_left) - len(poses_right)
                    poses_right = th.cat((poses_right, poses_right[-1].repeat(repeat_times, 1, 1)))

                if len(poses_left) != len(poses_right):
                    assert len(poses_left) == len(poses_right)
                poses = np.concatenate([poses_left, poses_right], axis=1)

                init_global_env_step = env.global_env_step
                arm_interp_start_time = time.time()
                for pose in poses:
                    interp_action = env_interface.target_pose_to_action(target_pose=pose)

                    interp_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                    interp_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]

                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=interp_action)
                    env.step(interp_action, video_writer)
                    left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                    right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(interp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    if ref_obj is not None:
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                arm_interp_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0] = round(arm_interp_finish_time - arm_interp_start_time, 2)
                print("Time taken for arm interpolation: ", phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0])

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_interp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_interp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"])

                # Setting the interpolation ranges
                MP_end_step_local_list = [local_env_step, local_env_step]
                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]
                # =============================================== ARM INTERPOLATION END ==================================================

            # If baseline is skillgen, perform mp + replay  
            elif baseline == "skillgen":
                # =============================================== Arm MP Planning =============================================
                
                # If at least one hand has motion planner waypoints, plan the motion
                if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                    target_pos = {
                        robot.eef_link_names["left"]: left_waypoint_pos,
                        robot.eef_link_names["right"]: right_waypoint_pos,
                    }
                    target_quat = {
                        robot.eef_link_names["left"]: left_waypoint_ori,
                        robot.eef_link_names["right"]: right_waypoint_ori,
                    }
                    emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                    
                    # Use OG to know attached objects
                    retval = self.obtain_attached_object(env, robot)
                    attached_obj = retval["attached_obj"]
                    attached_obj_scale = retval["attached_obj_scale"]

                    # If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                    if object_ref["arm_right"] is None:
                        del target_pos["right_eef_link"]
                        del target_quat["right_eef_link"]
                    elif object_ref["arm_left"] is None:
                        del target_pos["left_eef_link"]
                        del target_quat["left_eef_link"]

                    print("ARM MP START")
                    eyes_target_pos, eyes_target_quat = None, None

                    if enable_marker_vis:
                        env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                        env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                        env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                        env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)

                    # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                    num_tries = 3
                    arm_mp_trial = 0
                    new_target_pos = copy.deepcopy(target_pos)
                    while True:
                        
                        # Base condition 
                        if arm_mp_trial > 0:
                            
                            # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                            if env.retry_nav_on_arm_mp_failure:
                                base_condition = arm_mp_trial == num_tries
                            else:
                                base_condition = arm_mp_trial == num_tries or ("IK Fail" in mp_results[0].status.value)
                            
                            if base_condition:
                                print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                                if "TrajOpt Fail" in mp_results[0].status.value:
                                    env.err = "ArmMPTrajOptFailed"
                                elif "IK Fail" in mp_results[0].status.value:
                                    env.err = "ArmMPIKFailed"
                                else:
                                    env.err = "ArmMPOtherFailed"
                                env.valid_env = False 
                                env.execution_phase_ind += 1
                                return None
                                    
                        # Aggregate target_pos and target_quat to match batch_size
                        new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                        new_target_quat = {
                            k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                        }
                        
                        arm_mp_planning_start_time = time.time()
                        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                        mp_results, traj_paths = env.cmg.compute_trajectories(
                            target_pos=new_target_pos,
                            target_quat=new_target_quat,
                            is_local=False,
                            max_attempts=50,
                            timeout=60.0,
                            ik_fail_return=10,
                            enable_finetune_trajopt=True,
                            finetune_attempts=1,
                            return_full_result=True,
                            success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                            attached_obj=attached_obj,
                            attached_obj_scale=attached_obj_scale,
                            emb_sel=emb_sel,
                            eyes_target_pos=eyes_target_pos,
                            eyes_target_quat=eyes_target_quat,
                        )
                        arm_mp_planning_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                        successes = mp_results[0].success 
                        print("Arm MP successes: ", successes)
                        success_idx = th.where(successes)[0].cpu()
                        
                        if len(success_idx) == 0:
                            print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                            arm_mp_trial += 1
                            # modify target_pos a bit
                            for k in target_pos.keys():
                                new_target_pos[k] = target_pos[k] + th.rand(3) * 0.01 - 0.005
                            continue
                        else:
                            traj_path = traj_paths[success_idx[0]]
                            break
                
                    print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                    # ========================================================= End of Arm MP Planning ==========================================================

                    # ========================================================== Arm MP Execution ==========================================================
                    arm_mp_execution_start_time = time.time()

                    # Convert planned joint trajectory to actions
                    # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                    q_traj = q_traj.cpu()
                    mp_actions = []
                    for j_pos in q_traj:

                        # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                        if object_ref["arm_left"] is None:
                            j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        elif object_ref["arm_right"] is None:
                            j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                        action = robot.q_to_action(j_pos).cpu().numpy()

                        # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        if left_gripper_action is not None:
                            action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                        if right_gripper_action is not None:
                            action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                        
                        mp_actions.append(action)

                    left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                    right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                    # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                    if len(left_mp_waypoints) == 0:
                        # We need to pad the left hand waypoints to match the length of the MP trajectory
                        if len(left_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                                left_replay_waypoints.append(last_waypoint)

                        left_eef_poses = []
                        # We convert the target pose of the left hand to replay_action
                        # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                            left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_left"]
                            action[action_idx] = replay_action[action_idx]
                            action[env_interface.gripper_action_dim[0]] = left_replay_waypoints[i].gripper_action[0]

                        # We remove the waypoints that have been replayed for the left arm
                        left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                    # Same logic as above but for the right hand
                    elif len(right_mp_waypoints) == 0:
                        if len(right_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                                right_replay_waypoints.append(last_waypoint)
                        right_eef_poses = []
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                            right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_right"]
                            action[action_idx] = replay_action[action_idx]
                            action[env_interface.gripper_action_dim[1]] = right_replay_waypoints[i].gripper_action[1]

                        right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                    assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                    init_global_env_step = env.global_env_step
                    num_repeat = 1
                    for i, mp_action in enumerate(mp_actions):
                        for _ in range(num_repeat):
                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            # TODO: Check if we can use primtiive stack execute action here. This will allow for checking convergence errors etc.
                            env.step(mp_action, video_writer)
                            if enable_marker_vis:
                                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                                env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                                env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            if ref_obj is not None:
                                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]

                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]
                
                
                MP_end_step_local = copy.deepcopy(local_env_step)
                # left MP points
                if len(left_mp_waypoints) == 0: 
                    left_MP_end_step_local = 0
                else: 
                    left_MP_end_step_local = MP_end_step_local
                if len(right_mp_waypoints) == 0: 
                    right_MP_end_step_local = 0
                else: 
                    right_MP_end_step_local = MP_end_step_local

                MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

                arm_mp_execution_finish_time = time.time()
                # Since there is only 1 trial for arm MP execution, we set the 0th index
                phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
                print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])
                
                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

                # ============================================== End of Arm MP ==========================================================

            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)
            
            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()
            
            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None:
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None:
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos
            
            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            init_global_env_step = env.global_env_step
            for left_waypoint, right_waypoint in zip(left_replay_waypoints, right_replay_waypoints):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None:
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None:
                    pose[:4, :] = current_left_ee_pose
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                replay_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                replay_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            # =================================================== End of Arm Replay ==========================================================

            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results



    def execute(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (MG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
            cur_subtask_end_step_MP: list of size 2, the end point of motion planner for two arms

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """
   
        # TODO: This is duplicate code (also there in data_generator.py). Refactor this
        if object_ref["arm_right"] is None:
            ref_object = object_ref["arm_left"]
        elif object_ref["arm_left"] is None:
            ref_object = object_ref["arm_right"]
        else:
            ref_object = object_ref["arm_right"]
        
        if "torso" in ref_object:
            if isinstance(env.robot, Tiago):
                torso_link_name = "torso_lift_link"
            elif isinstance(env.robot, R1):
                torso_link_name = "torso_link4"
            else:
                raise ValueError("Robot type not supported")
            ref_obj = env.env.robots[0].links[torso_link_name]
        else:
            ref_obj = env.env.scene.object_registry("name", ref_object)
        env.primitive._tracking_object = ref_obj
        print("Will track object for this sub-step: ", ref_obj.name)
        robot = env.env.robots[0]
        
        # ================================= Base Navigation ==================================
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            seq = self.waypoint_sequences[0]
            
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            left_mp_last_waypoint = left_mp_waypoints[-1]
            left_waypoints = [left_mp_last_waypoint] + left_replay_waypoints

            left_waypoint_pos = th.vstack([th.tensor(wp.pose[0:3, 3]) for wp in left_waypoints])
            left_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[0:3, 0:3])) for wp in left_waypoints])

            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]
            right_mp_last_waypoint = right_mp_waypoints[-1]
            right_waypoints = [right_mp_last_waypoint] + right_replay_waypoints
 
            right_waypoint_pos = th.vstack([th.tensor(wp.pose[4:7, 3]) for wp in right_waypoints])
            right_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[4:7, 0:3])) for wp in right_waypoints])

            left_waypoint_pos, right_waypoint_pos = self._pad_tensors(left_waypoint_pos, right_waypoint_pos)
            left_waypoint_ori, right_waypoint_ori = self._pad_tensors(left_waypoint_ori, right_waypoint_ori)

            left_waypoint_pos = self._subsample_tensor(left_waypoint_pos)
            left_waypoint_ori = self._subsample_tensor(left_waypoint_ori)
            right_waypoint_pos = self._subsample_tensor(right_waypoint_pos)
            right_waypoint_ori = self._subsample_tensor(right_waypoint_ori)
            
            # left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            # left_waypoint = left_mp_waypoints[-1]
            # left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            # right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            # right_waypoint = right_mp_waypoints[-1]
            # right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))

            eef_pose = {
                "left": (left_waypoint_pos, left_waypoint_ori),
                "right": (right_waypoint_pos, right_waypoint_ori)
            }
            
            if enable_marker_vis:
                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                # env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                # env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)
                env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos[0], orientation=left_waypoint_ori[0])
                env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos[0], orientation=right_waypoint_ori[0])
                for _ in range(10): og.sim.step()
            
            # TODO: Implement this
            check_torso_mode_first = False
            if check_torso_mode_first:
                pass
                # Given the ref obect and the eef poses, check if we can only move the torso to satisfy reachability and visibility
                # 0. attach object
                # 1. call _target_in_reach_of_robot_and_visible(self,
                #                                               eef_pose,
                #                                               initial_joint_pos=env.robot.get_joint_positions(),
                #                                               skip_obstacle_update=True,
                #                                               ik_world_collision_check=False,
                #                                               emb_sel=CuRoboEmbodimentSelection.ARM,
                #                                               attach_obj=False, 
                #                                               eyes_pose=None):
                # So, above will sample eyes pose, check IK solving for (eef_poses, eyes_pose) w/o collision check, for samples that succeed previous Ik check
                # check if setting (current base + IK torso + current arms) is collision-free
                # 2. If yes, use the aforementioned (eyes_pose + eef poses) and do arm mode MP (which is with collision) and overwrite the arm actions to not do anything
                # 3. If above, succeeds, execute it
                # 4. else, continue to base MP

            
            num_tries = 3
            base_mp_trial = 0
            nav_mp_success = False
            while True:
                # Base condition
                if base_mp_trial == num_tries:
                    print("Base MP failed after {} trials. Giving up.".format(num_tries))
                    env.err = env.primitive.mp_err
                    # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
                    # In this case MP failed and phase was not actually executed
                    env.execution_phase_ind += 1
                    # env.valid_env = env.primitive.valid_env
                    return None

                print("Base MP trial: ", base_mp_trial)
                
                enable_visibility_constraint = isinstance(env.robot, R1) and env.hard_visibility_constraint # TODO: make this more general to handle Tiago
                
                # Pass only the eef that has a reference object associated with it (i.e. the arm that is relevant for this sub-step)
                if object_ref["arm_right"] is None:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose={"left": eef_pose["left"]}, visibility_constraint=enable_visibility_constraint)
                elif object_ref["arm_left"] is None:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose={"right": eef_pose["right"]}, visibility_constraint=enable_visibility_constraint)
                else:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose=eef_pose, visibility_constraint=enable_visibility_constraint)
                # action_generator = env.primitive._navigate_to_obj(obj=ref_obj, visibility_constraint=env.hard_visibility_constraint)
                
                init_state = og.sim.dump_state()
                local_env_step = 0
                states = []
                actions = []
                observations = []
                observations_info = []
                datagen_infos = []
                success = {"task": False}
                init_global_env_step = env.global_env_step
                # success = {k: False for k in env.is_success()} # success metrics
                for temp_idx, mp_action in enumerate(action_generator):
                    
                    # This will happen if
                    # 1. base sampling fails
                    # 2. base MP fails.
                    # 3. base execution fails to converge
                    if mp_action is None:
                        print(f"Base MP trial {base_mp_trial} failed. Retrying...")
                        base_mp_trial += 1
                        nav_mp_success = False
                        # This is there to avoid error in nav execution time (which in this case will always be 0)
                        nav_execution_start_time = time.time()
                        break
                    else:
                        nav_mp_success = True
                
                    if temp_idx == 0:
                        print("Time taken for base sampling: ", env.primitive.base_sampling_time)
                        print("Time taken for base MP planning: ", env.primitive.base_mp_planning_time)
                        nav_execution_start_time = time.time()

                    mp_action = mp_action.cpu().numpy()
                    # NOTE: For the MultiFinger gripper controler in binary mode that we use for tiago, we need to ensure that the
                    # gripper actions are correctly set based on whether an object is grasped by that gripper or not 
                    if attached_obj["left"] is not None:
                        mp_action[robot.gripper_action_idx["left"]] = -1
                    if attached_obj["right"] is not None:
                        mp_action[robot.gripper_action_idx["right"]] = -1
                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=mp_action)
                    # print("mp_action[robot.base_action_idx]: ", mp_action[robot.base_action_idx])
                    env.step(mp_action, video_writer)
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(mp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)

                # Save timings to current_phase_logs
                nav_execution_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["base_sampling_time"][base_mp_trial] = env.primitive.base_sampling_time
                phase_logs[env.execution_phase_ind]["base_mp_planning_time"][base_mp_trial] = env.primitive.base_mp_planning_time
                phase_logs[env.execution_phase_ind]["base_mp_execution_time"][base_mp_trial] = round(nav_execution_finish_time - nav_execution_start_time, 2)
                
                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for nav_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
                print(f"Visibility stats for nav_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"])

                if not nav_mp_success:
                    # This will happen if
                    # 1. base sampling fails
                    # 2. base MP fails.
                    # 3. base execution fails to converge

                    # In case #3, we actually step physics in OG, so we need to reset the state
                    if env.primitive.mp_err in ["BaseExecutionBaseTargetNotReached", "BaseExecutionArmTorsoTargetNotReached"]:
                        og.sim.load_state(init_state)
                        for _ in range(5): og.sim.step()
                        
                        # Reset the visibility stats
                        self.reset_visibility_counter(env)

                    continue
                
                env.err = env.primitive.mp_err
                MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
                left_mp_ranges = [init_global_env_step, env.global_env_step]
                right_mp_ranges = [init_global_env_step, env.global_env_step]
                results = dict(
                    states=states,
                    observations=observations,
                    datagen_infos=datagen_infos,
                    actions=np.array(actions),
                    success=bool(success["task"]),
                    mp_end_steps=MP_end_step_local_list,
                    subtask_lengths=local_env_step,
                    left_mp_ranges=left_mp_ranges,
                    right_mp_ranges=right_mp_ranges,
                    retry_nav=False,
                    observations_info=observations_info
                )
                # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
                # In this case MP succeeded and phase was actually executed
                env.execution_phase_ind += 1
                env.phases_completed_wo_mp_err += 1
                return results
        # ============================================== Base Navigation ==============================================

        if phase_type != "navigation":
            # =============================================== Arm MP Planning =============================================
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            # success = {k: False for k in env.is_success()} # success metrics

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # # Temporary: This is just to capture the first image after navigating to the teacup, just for visualization
            # if object_ref["arm_left"] == "teacup" and grasp_init_views_video_writer is not None:
            #     robot_name = env.env.robots[0].name
            #     obs, obs_info = env.get_observation()
            #     ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
            #     # eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            #     # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            #     concatenated_img = hori_concatenate_image([ego_img])
            #     grasp_init_views_video_writer.append_data(concatenated_img)

            
            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

            # # Option 1: If one of the arm does not hav a ref object, set its target pose as the current pose
            # if object_ref["arm_right"] is None:
            #     right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")
            # elif object_ref["arm_left"] is None:
            #     left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            
            # If at least one hand has motion planner waypoints, plan the motion
            if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                target_pos = {
                    robot.eef_link_names["left"]: left_waypoint_pos,
                    robot.eef_link_names["right"]: right_waypoint_pos,
                }
                target_quat = {
                    robot.eef_link_names["left"]: left_waypoint_ori,
                    robot.eef_link_names["right"]: right_waypoint_ori,
                }
                # If both hands have motion planner waypoints, we use the arm + torso embodiment
                # If only one of the hands has motion planner waypoints, we use the arm embodiment only because
                # when we replay the waypoints for the other hand, we assume the torso is fixed.
                emb_sel = CuRoboEmbodimentSelection.ARM if len(left_mp_waypoints) > 0 and len(right_mp_waypoints) > 0 else CuRoboEmbodimentSelection.ARM_NO_TORSO
                
                # To test MP in arm_no_toso mode instead of arm mode, uncomment the line below
                emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                
                # # Option 1: Use template to know attached objects
                # if attached_obj is None:
                #     attached_obj_scale = None
                # else:
                #     attached_obj_new = {}
                #     attached_obj_scale = {}
                #     for arm, obj_name in attached_obj.items():
                #         if obj_name is not None:
                #             attached_obj_new[robot.eef_link_names[arm]] = env.env.scene.object_registry("name", obj_name).root_link
                #             attached_obj_scale[robot.eef_link_names[arm]] = 0.9
                #     attached_obj = attached_obj_new

                # Option 2: Use OG to know attached objects
                retval = self.obtain_attached_object(env, robot)
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]

                # Option 2: If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                if object_ref["arm_right"] is None:
                    del target_pos["right_eef_link"]
                    del target_quat["right_eef_link"]
                elif object_ref["arm_left"] is None:
                    del target_pos["left_eef_link"]
                    del target_quat["left_eef_link"]

                # # Check object visibility at start-of-manip step
                # try:
                #     obs, obs_info = env.get_observation()
                #     seg_instance = obs[f"{env.robot_name}::{env.robot_name}:eyes:Camera:0::seg_instance"]
                #     seg_instance_info = obs_info[f"{env.robot_name}"][f"{env.robot_name}:eyes:Camera:0"]["seg_instance"]
                #     key_of_coffee_cup = next((key for key, value in seg_instance_info.items() if value == "coffee_cup"), None)
                #     if key_of_coffee_cup is None:
                #         count = 0
                #     else:
                #         count = (seg_instance == key_of_coffee_cup).sum().item()
                #     if count > 150:
                #         env.obj_visible_at_start_of_manip = True
                # except Exception as e:

                
                # This is for retract behavior. We are not using this as of now, but let it be 
                initial_left_eef_pose = robot.get_eef_pose("left")
                initial_right_eef_pose = robot.get_eef_pose("right")
                
                print("ARM MP START")
                eyes_target_pos, eyes_target_quat = None, None
                # NOTE: Keep this commented out. We won't be using soft visibility constraint with manipulation for now. As we are using ARM_NO_TORSO mode
                # if env.soft_visibility_constraint:
                #     obj_pose = ref_obj.get_position_orientation()
                #     eyes_target_pos = obj_pose[0]
                #     eyes_target_quat = obj_pose[1]

                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                    env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)

                
                # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                num_tries = 3
                arm_mp_trial = 0
                new_target_pos = copy.deepcopy(target_pos)
                while True:
                    
                    # Base condition 
                    if arm_mp_trial > 0:
                        # # Trying a hacky way to reduce the IK failure. Basically moving the robot base a bit towards the object. 
                        # # This does not ensure collision-free motion
                        # if "IK Fail" in mp_results[0].status.value:
                        #     obj_pos = ref_obj.get_position_orientation()[0][:2]
                        #     robot_base_pose = env.robot.get_position_orientation()
                        #     robot_base_pos = robot_base_pose[0][:2]
                        #     vec = obj_pos - robot_base_pos
                        #     vec = vec / np.linalg.norm(vec)
                        #     for _ in range(10):
                        #         joint_pos = env.robot.get_joint_positions()
                        #         joint_pos[:2] = joint_pos[:2] + (vec * 0.01)
                        #         action = env.robot.q_to_action(joint_pos).cpu().numpy()
                        #         # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        #         if left_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                        #         if right_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                                
                        #         state = env.get_state()["states"]
                        #         obs, obs_info = env.get_obs_IL()
                        #         datagen_info = env_interface.get_datagen_info(action=action)
                        #         env.step(action, video_writer)
                        #         local_env_step += 1
                        #         env.global_env_step += 1
                        #         states.append(state)
                        #         actions.append(action)
                        #         observations.append(obs)
                        #         datagen_infos.append(datagen_info)

                        if ("IK Fail" in mp_results[0].status.value or "TrajOpt Fail" in mp_results[0].status.value) and env.retry_nav_on_arm_mp_failure:
                            results = dict(
                                states=states,
                                observations=observations,
                                datagen_infos=datagen_infos,
                                actions=np.array(actions),
                                success=bool(success["task"]),
                                retry_nav=True,
                                observations_info=observations_info
                            )
                            return results
                        
                        # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                        if env.retry_nav_on_arm_mp_failure:
                            base_condition = arm_mp_trial == num_tries
                        else:
                            base_condition = arm_mp_trial == num_tries or ("IK Fail" in mp_results[0].status.value)
                        
                        if base_condition:
                            print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                            if "TrajOpt Fail" in mp_results[0].status.value:
                                env.err = "ArmMPTrajOptFailed"
                            elif "IK Fail" in mp_results[0].status.value:
                                env.err = "ArmMPIKFailed"
                            else:
                                env.err = "ArmMPOtherFailed"
                            env.valid_env = False 
                            env.execution_phase_ind += 1
                            return None
                                
                    # Aggregate target_pos and target_quat to match batch_size
                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                    new_target_quat = {
                        k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                    }
                    
                    arm_mp_planning_start_time = time.time()
                    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                    mp_results, traj_paths = env.cmg.compute_trajectories(
                        target_pos=new_target_pos,
                        target_quat=new_target_quat,
                        is_local=False,
                        max_attempts=50,
                        timeout=60.0,
                        ik_fail_return=10,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1,
                        return_full_result=True,
                        success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                        attached_obj=attached_obj,
                        attached_obj_scale=attached_obj_scale,
                        emb_sel=emb_sel,
                        eyes_target_pos=eyes_target_pos,
                        eyes_target_quat=eyes_target_quat,
                    )
                    arm_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                    successes = mp_results[0].success 
                    print("Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()
                    
                    if len(success_idx) == 0:
                        print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                        arm_mp_trial += 1
                        # modify target_pos a bit
                        for k in target_pos.keys():
                            new_target_pos[k] = target_pos[k] + th.rand(3) * 0.01 - 0.005
                        continue
                    else:
                        traj_path = traj_paths[success_idx[0]]
                        break
            
                print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                # ========================================================= End of Arm MP Planning ==========================================================
                
                # ========================================================== Arm MP Execution ==========================================================
                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)

                arm_mp_execution_start_time = time.time()

                # These lines are for debugging purposes.
                # successes, traj_paths = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=False, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)
                # full_result = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=True, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)

                # Convert planned joint trajectory to actions
                # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                # If we use curobo joint space planning instead of Cartesian space planning, we need to downsample the trajectory 
                # q_traj = q_traj[::50]
                q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                q_traj = q_traj.cpu()
                mp_actions = []
                for j_pos in q_traj:

                    # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                    if object_ref["arm_left"] is None:
                        j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    elif object_ref["arm_right"] is None:
                        j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                    action = robot.q_to_action(j_pos).cpu().numpy()

                    # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                    if left_gripper_action is not None:
                        action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                    if right_gripper_action is not None:
                        action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                    
                    mp_actions.append(action)

                left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                if len(left_mp_waypoints) == 0:
                    # We need to pad the left hand waypoints to match the length of the MP trajectory
                    if len(left_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                            left_replay_waypoints.append(last_waypoint)

                    left_eef_poses = []
                    # We convert the target pose of the left hand to replay_action
                    # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                        left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_left"]
                        action[action_idx] = replay_action[action_idx]
                        action[env_interface.gripper_action_dim[0]] = left_replay_waypoints[i].gripper_action[0]

                    # We remove the waypoints that have been replayed for the left arm
                    left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                # Same logic as above but for the right hand
                elif len(right_mp_waypoints) == 0:
                    if len(right_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                            right_replay_waypoints.append(last_waypoint)
                    right_eef_poses = []
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                        right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_right"]
                        action[action_idx] = replay_action[action_idx]
                        action[env_interface.gripper_action_dim[1]] = right_replay_waypoints[i].gripper_action[1]

                    right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                init_global_env_step = env.global_env_step
                num_repeat = 1
                for i, mp_action in enumerate(mp_actions):
                    for _ in range(num_repeat):
                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        # TODO: Check if we can use primitive stack execute action here. This will allow for checking convergence errors etc.
                        mp_action = env.primitive._postprocess_action(mp_action)
                        env.step(mp_action, video_writer)
                        if enable_marker_vis:
                            env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                            env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                            env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                            env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                # # If using MP in default mode. Will remove this code later but keeping it for now for debugging purposes  
                # q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                # q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                # q_traj = q_traj.cpu()
                # left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(q_traj)
                # right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(q_traj)
                # num_repeat = 1
                # for i, j_pos in enumerate(q_traj):
                #     for _ in range(num_repeat):
                #         action = robot.q_to_action(j_pos).cpu().numpy()
                #         if left_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                #         if right_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                #         state = env.get_state()["states"]
                #         # obs, obs_info = env.get_obs_IL()
                #         datagen_info = env_interface.get_datagen_info(action=action)
                #         env.step(action)
                #         env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                #         env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                #         env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                #         env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                #         local_env_step += 1
                #         states.append(state)
                #         actions.append(action)
                #         observations.append(obs)
                #         datagen_infos.append(datagen_info)


            # Set the MP ranges to save to hdf5 file
            left_mp_ranges, right_mp_ranges = None, None
            if len(left_mp_waypoints) > 0:
                left_mp_ranges = [init_global_env_step, env.global_env_step]
            if len(right_mp_waypoints) > 0:
                right_mp_ranges = [init_global_env_step, env.global_env_step]
            
            
            MP_end_step_local = copy.deepcopy(local_env_step)
            # left MP points
            if len(left_mp_waypoints) == 0: 
                left_MP_end_step_local = 0
            else: 
                left_MP_end_step_local = MP_end_step_local
            if len(right_mp_waypoints) == 0: 
                right_MP_end_step_local = 0
            else: 
                right_MP_end_step_local = MP_end_step_local

            MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

            arm_mp_execution_finish_time = time.time()
            # Since there is only 1 trial for arm MP execution, we set the 0th index
            phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
            print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])
            
            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
            print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

            # ============================================== End of Arm MP Execution ==========================================================
            
            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)
            
            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()
            
            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None:
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None:
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos
            
            init_global_env_step = env.global_env_step
            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            for left_waypoint, right_waypoint in zip(left_replay_waypoints, right_replay_waypoints):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None:
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None:
                    pose[:4, :] = current_left_ee_pose
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                replay_action[env_interface.gripper_action_dim[0]] = left_waypoint.gripper_action[0]
                replay_action[env_interface.gripper_action_dim[1]] = right_waypoint.gripper_action[1]

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])
            
            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            # =================================================== End of Arm Replay ==========================================================

            # =================================================== Arm/Torso Retract ==========================================================
            if retract_type != "no_retract":
                print("Starting Retract")
                
                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)
                
                retract_torso_only = False
                current_robot_base_pose_wrt_world = robot.get_position_orientation()
                # If we retract the left and right eef to the pose at the start of arm MP
                if retract_type == "retract_to_start_of_arm_mp":
                    if object_ref["arm_right"] is None:
                        arm_side = "left"
                        current_left_eef_pose = robot.get_eef_pose("left")
                        target_pos = {"left_eef_link": initial_left_eef_pose[0]}
                        # target_quat = {"left_eef_link": current_left_eef_pose[1]} # Retain current orientation
                        target_quat = {"left_eef_link": initial_left_eef_pose[1]} # Use initial orientation
                    elif object_ref["arm_left"] is None:
                        arm_side = "right"
                        current_right_eef_pose = robot.get_eef_pose("right")
                        target_pos = {"right_eef_link": initial_right_eef_pose[0]}
                        # target_quat = {"right_eef_link": current_right_eef_pose[1]} # Retain current orientation
                        target_quat = {"right_eef_link": initial_right_eef_pose[1]} # Retain initial orientation
                    # TODO: implement this. Not too important for now as this would never happen. In this case it's a bimanual coordinated and we don't need to retract
                    else:
                        pass

                # If we retract the left and right eef and eyes to a canonical pose
                elif retract_type == "retract_to_canonical_pose":
                    eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                    eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)

                    target_pos = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[0],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[0],
                        "eyes": eyes_reset_pose_wrt_world[0],
                    }
                    target_quat = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[1],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[1],
                        "eyes": eyes_reset_pose_wrt_world[1],
                    }

                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                    eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)
                    current_left_eef_pose = robot.get_eef_pose("left")

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)
                    current_right_eef_pose = robot.get_eef_pose("right")

                    target_pos = {
                        "left_eef_link": left_eef_reset_pose_wrt_robot[0],
                        "right_eef_link": right_eef_reset_pose_wrt_robot[0],
                        "eyes": eyes_reset_pose_wrt_world[0],
                    }
                    target_quat = {
                        "left_eef_link": current_left_eef_pose[1],
                        "right_eef_link": current_right_eef_pose[1],
                        "eyes": eyes_reset_pose_wrt_world[1],
                    }

                else:
                    raise ValueError(f"Invalid retract type: {retract_type}")


                # Aggregate target_pos and target_quat to match batch_size
                new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                new_target_quat = {
                    k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                }
                
                retval = self.obtain_attached_object(env, robot)
                grasp_action = retval["grasp_action"]
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]

                # if enable_marker_vis:
                #     if arm_side == "left":
                #         env.eef_goal_marker_left.set_position_orientation(target_pos["left_eef_link"], target_quat["left_eef_link"])
                #     elif arm_side == "right":
                #         env.eef_goal_marker_right.set_position_orientation(target_pos["right_eef_link"], target_quat["right_eef_link"])
                
                if retract_type == "retract_to_start_of_arm_mp":
                    emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                elif retract_type == "retract_to_canonical_pose":
                    emb_sel = CuRoboEmbodimentSelection.ARM
                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    emb_sel = CuRoboEmbodimentSelection.ARM

                full_retract_mp_planning_start_time = time.time()
                mp_results, traj_paths = env.cmg.compute_trajectories(
                    target_pos=new_target_pos,
                    target_quat=new_target_quat,
                    is_local=False,
                    max_attempts=50,
                    timeout=20.0,
                    ik_fail_return=50,
                    enable_finetune_trajopt=True,
                    finetune_attempts=1,
                    return_full_result=True,
                    success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                    attached_obj=attached_obj,
                    attached_obj_scale=attached_obj_scale,
                    emb_sel=emb_sel,
                )
                full_retract_mp_planning_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0] = round(full_retract_mp_planning_finish_time - full_retract_mp_planning_start_time, 2)
                print("Time taken for full retract MP planning: ", phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0])

                successes = mp_results[0].success 
                print("Retract Arm MP successes: ", successes)
                success_idx = th.where(successes)[0].cpu()

                if len(success_idx) == 0:
                    print(f"Arm retract failed with status {mp_results[0].status}.")
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = mp_results[0].status.value
                    retract_torso_only = True
                else:
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = "None"
                    full_retract_mp_execution_start_time = time.time()
                    traj_path = traj_paths[success_idx[0]]

                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                    q_traj = q_traj.cpu()

                    num_repeat = 1
                    init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                    init_global_env_step = env.global_env_step
                    for j_pos in q_traj:
                        if retract_type == "retract_to_start_of_arm_mp":
                            if arm_side == "left":
                                j_pos[robot.arm_control_idx["right"]] = init_right_arm_pos
                            elif arm_side == "right":
                                j_pos[robot.arm_control_idx["left"]] = init_left_arm_pos

                        mp_action = robot.q_to_action(j_pos).cpu().numpy()
                        mp_action[robot.gripper_action_idx["left"]] = grasp_action["left"]
                        mp_action[robot.gripper_action_idx["right"]] = grasp_action["right"]

                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        env.step(mp_action, video_writer)
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                    full_retract_mp_execution_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["full_retract_mp_execution_time"][0] = round(full_retract_mp_execution_finish_time - full_retract_mp_execution_start_time, 2)

                    num_phase_steps = env.global_env_step - init_global_env_step
                    for sensor_name, sensor in env.robot.sensors.items():
                        if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                            shortened_sensor_name = sensor_name.split(":")[1]
                            if num_phase_steps > 0:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                            else:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"]= 0
                            print(f"Visibility stats for full_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"])
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"]= 0
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_steps"] = num_phase_steps
                    print(f"Visibility stats for full_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"])

                # If full retract failed, try retracting only the torso
                if retract_torso_only and retract_type != "retract_to_start_of_arm_mp":
                    print("Retracting torso only")
                    
                    # reset the visibility counter for each sensor
                    self.reset_visibility_counter(env)
                    
                    target_pos = {"eyes": eyes_reset_pose_wrt_world[0]}
                    target_quat = {"eyes": eyes_reset_pose_wrt_world[1]}

                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                    new_target_quat = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()}

                    torso_retract_mp_planning_start_time = time.time()
                    mp_results, traj_paths = env.cmg.compute_trajectories(
                        target_pos=new_target_pos,
                        target_quat=new_target_quat,
                        is_local=False,
                        max_attempts=50,
                        timeout=20.0,
                        ik_fail_return=50,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1,
                        return_full_result=True,
                        success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                        attached_obj=attached_obj,
                        attached_obj_scale=attached_obj_scale,
                        emb_sel=emb_sel,
                    )
                    torso_retract_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["torso_retract_mp_planning_time"][0] = round(torso_retract_mp_planning_finish_time - torso_retract_mp_planning_start_time, 2)

                    successes = mp_results[0].success 
                    print("Torso-only retract: Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()

                    if len(success_idx) == 0:
                        print(f"Torso retract failed with status {mp_results[0].status}.")
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = mp_results[0].status.value
                    else:
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = "None"
                        torso_retract_mp_execution_start_time = time.time()
                        traj_path = traj_paths[success_idx[0]]

                        q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                        q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                        q_traj = q_traj.cpu()

                        num_repeat = 1
                        init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                        init_global_env_step = env.global_env_step
                        for j_pos in q_traj:
                            mp_action = robot.q_to_action(j_pos).cpu().numpy()
                            mp_action[robot.gripper_action_idx["left"]] = grasp_action["left"]
                            mp_action[robot.gripper_action_idx["right"]] = grasp_action["right"]
                            # Don't want to move the arm relative to the torso
                            mp_action[robot.arm_action_idx["right"]] = init_right_arm_pos
                            mp_action[robot.arm_action_idx["left"]] = init_left_arm_pos

                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            env.step(mp_action, video_writer)
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]
                        
                        torso_retract_mp_execution_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_execution_time"][0] = round(torso_retract_mp_execution_finish_time - torso_retract_mp_execution_start_time, 2)
                        
                        num_phase_steps = env.global_env_step - init_global_env_step
                        for sensor_name, sensor in env.robot.sensors.items():
                            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                                shortened_sensor_name = sensor_name.split(":")[1]
                                if num_phase_steps > 0:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                                else:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"]= 0
                                print(f"Visibility stats for torso_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"])
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"]= 0
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_steps"] = num_phase_steps
                        print(f"Visibility stats for torso_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"])

            # ================================================== End of Arm/Torso Retract ==========================================================
                    
            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results
