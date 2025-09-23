"""
Main data generation script.

Example:

TASK=pick_cup
DR=0
NUM_DEMOS=10
WORKER_ID=0
FOLDER=/path/to/data
python momagen/scripts/generate_dataset.py \
    --config momagen/datasets/configs/demo_src_r1_$TASK\_task_D$DR.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --folder $FOLDER/$TASK/r1_$TASK\_worker_$WORKER_ID \
    --seed $WORKER_ID
"""

import os
import shutil
import json
import time
import argparse
import traceback
import random
import imageio
import numpy as np
import torch as th
import warnings
import logging

# Configure logging and warnings
th.set_printoptions(precision=3, sci_mode=False, linewidth=1000)
warnings.filterwarnings('ignore', module='trimesh')
logging.getLogger('trimesh').setLevel(logging.ERROR)
logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

from robomimic.utils.file_utils import get_env_metadata_from_dataset

import robomimic.utils.env_utils as EnvUtils
import momagen.utils.file_utils as MG_FileUtils
import momagen.utils.robomimic_utils as RobomimicUtils
from momagen.utils.robot_config import configure_tiago_env_meta

from momagen.configs.config import config_factory
from momagen.configs.task_spec import MG_TaskSpec
from momagen.datagen.data_generator import DataGenerator
from momagen.env_interfaces.base import make_interface

import omnigibson as og

from omnigibson.objects.primitive_object import PrimitiveObject

# Disable pyembree for trimesh
os.environ["TRIMESH_NO_PYEMBREE"] = "1"

ROBOT_TYPE = "R1"

def visualize_base_poses(env):
    """Visualize base poses with colored markers (debug function)."""
    sampled_base_poses = env.sampled_base_poses

    # Create failure markers (red)
    _create_pose_markers(
        positions=sampled_base_poses["failure"],
        prefix="base_marker_failure",
        color=th.tensor([1, 0, 0, 1]),
        env=env
    )

    # Create success markers (green)
    _create_pose_markers(
        positions=sampled_base_poses["success"],
        prefix="base_marker_success",
        color=th.tensor([0, 1, 0, 1]),
        env=env
    )

def _create_pose_markers(positions, prefix, color, env):
    """Helper to create visualization markers."""
    base_marker_list = []
    for i in range(len(positions)):
        base_marker = PrimitiveObject(
            relative_prim_path=f"/{prefix}_{i}",
            primitive_type="Cube",
            name=f"{prefix}_{i}",
            size=th.tensor([0.03, 0.03, 0.03]),
            visual_only=True,
            rgba=color
        )
        base_marker_list.append(base_marker)

    if base_marker_list:
        og.sim.batch_add_objects(base_marker_list, [env.env.scene] * len(base_marker_list))
        for i, pos in enumerate(positions):
            base_marker_list[i].set_position_orientation(position=pos)

def get_important_stats(
    new_dataset_folder_path,
    num_success,
    num_failures,
    num_attempts,
    num_problematic,
    ep_lengths,
    start_time=None,
    ep_length_stats=None,
    all_episode_logs=None
):
    """
    Return a summary of important stats to write to json.

    Args:
        new_dataset_folder_path (str): path to folder that will contain generated dataset
        num_success (int): number of successful trajectories generated
        num_failures (int): number of failed trajectories
        num_attempts (int): number of total attempts
        num_problematic (int): number of problematic trajectories that failed due
            to a specific exception that was caught
        start_time (float or None): starting time for this run from time.time()
        ep_length_stats (dict or None): if provided, should have entries that summarize
            the episode length statistics over the successfully generated trajectories

    Returns:
        important_stats (dict): dictionary with useful summary of statistics
    """
    important_stats = dict(
        generation_path=new_dataset_folder_path,
        success_rate=((100. * num_success) / num_attempts),
        failure_rate=((100. * num_failures) / num_attempts),
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        all_episode_logs=all_episode_logs

    )
    if (ep_length_stats is not None):
        important_stats.update(ep_length_stats)
    if start_time is not None:
        # add in time taken
        important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time) / 3600.)
    return important_stats


def generate_dataset(
    mg_config,
    auto_remove_exp=False,
    render=False,
    no_save_video=False,
    video_skip=5,
    render_image_names=None,
    pause_subtask=False,
    bimanual=False,
    enable_marker_vis=False,
    ds_ratio=1,
    no_partial_tasks=False,
    headless=False,
    baseline=None,
):
    """
    Main function to collect a new dataset with MoMaGen.

    Args:
        mg_config (MG_Config instance): MoMaGen config object

        auto_remove_exp (bool): if True, will remove generation folder if it exists, else
            user will be prompted to decide whether to keep existing folder or not

        render (bool): if True, render each data generation attempt on-screen

        no_save_video (bool): if True, don't save video of data generation attempts 

        video_skip (int): skip every nth frame when writing video

        render_image_names (list of str or None): if provided, specify camera names to 
            use during on-screen / off-screen rendering to override defaults

        pause_subtask (bool): if True, pause after every subtask during generation, for
            debugging.

        bimanual (bool): if True, use bimanual robot configuration.

        enable_marker_vis (bool): if True, enable marker visualization.

        ds_ratio (int): downsampling ratio for trajectory.

        no_partial_tasks (bool): if True, don't save partial trajectories.

        headless (bool): if True, run in headless mode.

        baseline (str or None): baseline method to use (e.g., "mimicgen", "skillgen").
    """

    # time this run
    script_start_time = time.time()

    # check some args
    write_video = not no_save_video
    assert not (render and write_video) # either on-screen or video but not both
    if pause_subtask:
        assert render, "should enable on-screen rendering for pausing to be useful"

    if write_video:
        # debug video - use same cameras as observations
        if len(mg_config.obs.camera_names) > 0:
            assert render_image_names is None
            render_image_names = list(mg_config.obs.camera_names)

    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

    # get environment metadata from dataset
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    
    if ROBOT_TYPE == "Tiago":
        env_meta = configure_tiago_env_meta(env_meta)

    # set seed for generation
    random.seed(mg_config.experiment.seed)
    np.random.seed(mg_config.experiment.seed)
    th.manual_seed(mg_config.experiment.seed)

    # create new folder for this data generation run
    base_folder = os.path.expandvars(os.path.expanduser(mg_config.experiment.generation.path))
    new_dataset_folder_name = mg_config.experiment.name
    new_dataset_folder_path = os.path.join(
        base_folder,
        new_dataset_folder_name,
    )
    print("\nData will be generated at: {}".format(new_dataset_folder_path))

    # ensure dataset folder does not exist, and make new folder
    exist_ok = False
    if os.path.exists(new_dataset_folder_path):
        if not auto_remove_exp:
            # ans = input("\nWARNING: dataset folder ({}) already exists! \noverwrite? (y/n)\n".format(new_dataset_folder_path))
            ans = "n"
        else:
            ans = "y"
        if ans == "y":
            print("Removed old results folder at {}".format(new_dataset_folder_path))
            shutil.rmtree(new_dataset_folder_path)
        else:
            print("Keeping old dataset folder. Note that individual files may still be overwritten.")
            exist_ok = True
    os.makedirs(new_dataset_folder_path, exist_ok=exist_ok)

    # log terminal output to text file

    # save config to disk
    MG_FileUtils.write_json(
        json_dic=mg_config,
        json_path=os.path.join(new_dataset_folder_path, "mg_config.json"),
    )

    print("\n============= Config =============")
    print(mg_config)
    print("")

    # some paths that we will create inside our new dataset folder

    # new dataset that will be generated
    new_dataset_path = os.path.join(new_dataset_folder_path, "demo.hdf5")

    # tmp folder that will contain per-episode hdf5s that were successful (they will be merged later)
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    os.makedirs(tmp_dataset_folder_path, exist_ok=exist_ok)

    # folder containing logs
    json_log_path = os.path.join(new_dataset_folder_path, "logs")
    os.makedirs(json_log_path, exist_ok=exist_ok)

    if mg_config.experiment.generation.keep_failed:
        # new dataset for failed trajectories, and tmp folder for per-episode hdf5s that failed
        new_failed_dataset_path = os.path.join(new_dataset_folder_path, "demo_failed.hdf5")
        tmp_dataset_failed_folder_path = os.path.join(new_dataset_folder_path, "tmp_failed")
        os.makedirs(tmp_dataset_failed_folder_path, exist_ok=exist_ok)

    # get list of source demonstration keys from source hdf5
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset_path,
        filter_key=mg_config.experiment.source.filter_key,
        start=mg_config.experiment.source.start,
        n=mg_config.experiment.source.n,
    )

    # prepare args for creating simulation environment

    # auto-fill camera rendering info if not specified
    if (write_video or render) and (render_image_names is None):
        render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)
    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    # env args: cameras to use come from debug camera video to write, or from observation collection
    camera_names = (mg_config.obs.camera_names if not write_video else render_image_names)

    # env args: don't use image obs when writing debug video
    use_image_obs = ((mg_config.obs.collect_obs and (len(mg_config.obs.camera_names) > 0)) if not write_video else False)
    use_depth_obs = False


    # simulation environment
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        camera_names=camera_names,
        camera_height=mg_config.obs.camera_height,
        camera_width=mg_config.obs.camera_width,
        render=render,
        render_offscreen=write_video,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        manipulation_only=False,
        real_robot_mode="pick_cup" in mg_config.experiment.task.name,
        baseline=baseline,
    )
    print("\n==== Real robot mode ====")
    print("pick_cup" in mg_config.experiment.task.name)

    print("\n==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # get information necessary to create env interface
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    # possibly override from config
    if mg_config.experiment.task.interface is not None:
        env_interface_name = mg_config.experiment.task.interface
    if mg_config.experiment.task.interface_type is not None:
        env_interface_type = mg_config.experiment.task.interface_type

    # create environment interface to use during data generation
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))

    # self.arm_command_start_idx {'left': 5, 'right': 12}
    # self.arm_command_end_idx {'left': 11, 'right': 18}

    # make sure we except the same exceptions that we would normally except during policy rollouts
    exceptions_to_except = env.rollout_exceptions

    # get task spec object from config
    task_spec_json_string = mg_config.task.task_spec.dump()
    if bimanual:
        task_spec = MG_TaskSpec.from_json_bimanual(json_string=task_spec_json_string)
    else:
        task_spec = MG_TaskSpec.from_json(json_string=task_spec_json_string)
    
    D2_sign = True if "D2" in mg_config.experiment.task.name else False
    # make data generator object
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
        bimanual=bimanual,
        D2_sign=D2_sign,
    )

    if write_video:
        os.makedirs(f"{new_dataset_folder_path}/videos", exist_ok=True) 

    print("\n==== Created Data Generator ====")
    print(data_generator)
    print("")

    existing_log_jsons = os.listdir(json_log_path)
    if len(existing_log_jsons) > 0:
        # find the last json file
        existing_log_jsons.sort()
        last_json = existing_log_jsons[-1]
        last_json_path = os.path.join(json_log_path, last_json)
        with open(last_json_path, "r") as f:
            last_json_dict = json.load(f)
        num_attempts = last_json_dict["num_attempts"]
        num_success = last_json_dict["num_success"]
        num_failures = last_json_dict["num_failures"]
        num_problematic = last_json_dict["num_problematic"]
        # backward compatibility
        ep_lengths = last_json_dict.get("ep_lengths", [])
        all_episode_logs = last_json_dict["all_episode_logs"]
    else:
        # data generation statistics
        num_attempts = 0
        num_success = 0
        num_failures = 0
        num_problematic = 0
        ep_lengths = [] # episode lengths for successfully generated data
        all_episode_logs = {
            "episode_number": [],
            "err_status": [],
            "time_taken": [],
            "task_success": [],
            "phases_completed": [],
            "phase_logs": [],
        }


    # we will keep generating data until @num_trials successes (if @guarantee_success) else @num_trials attempts
    num_trials = mg_config.experiment.generation.num_trials
    guarantee_success = mg_config.experiment.generation.guarantee
    
    base_mp_failures, arm_mp_ik_failures, arm_mp_trajopt_failures, arm_mp_other_failures, base_sampling_failures, base_mp_ik_failures = 0, 0, 0, 0, 0, 0
    obj_visible_at_start_of_manip = 0

    while True:
        print(f"======================= ATTEMPT {num_attempts} ========================")

        # we might write a video to show the data generation attempts
        video_writer = None
        if write_video:
            video_writer = imageio.get_writer(f"{new_dataset_folder_path}/videos/{num_attempts:04d}.mp4", fps=20)

        # generate trajectory
        try:
            episode_start_time = time.time()
            generated_traj = data_generator.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=mg_config.experiment.generation.select_src_per_subtask,
                transform_first_robot_pose=mg_config.experiment.generation.transform_first_robot_pose,
                interpolate_from_last_target_pose=mg_config.experiment.generation.interpolate_from_last_target_pose,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                camera_names=render_image_names,
                pause_subtask=pause_subtask,
                enable_marker_vis=enable_marker_vis,
                ds_ratio=ds_ratio,
                grasp_init_views_video_writer=None,
                no_partial_tasks=no_partial_tasks,
                baseline=baseline,
            )
            episode_time_taken = time.time() - episode_start_time
            print("==============================")
            print("Time taken for generation: {:.2f} seconds".format(episode_time_taken))
            print("==============================")

            # save episode logs
            all_episode_logs["episode_number"].append(num_attempts+num_problematic)
            all_episode_logs["err_status"].append(env.err)
            all_episode_logs["time_taken"].append(episode_time_taken)
            all_episode_logs["task_success"].append(env.is_success()["task"])
            if generated_traj is not None:
                all_episode_logs["phases_completed"].append(generated_traj["phases_completed"])
                all_episode_logs["phase_logs"].append(generated_traj["phase_logs"])
            else:
                all_episode_logs["phases_completed"].append(-1)
                all_episode_logs["phase_logs"].append(dict())

        except exceptions_to_except as e:
            # problematic trajectory - do not have this count towards our total number of attempts, and re-try
            print("")
            print("*" * 50)
            print("WARNING: got rollout exception {}".format(e))
            print("*" * 50)
            print("")
            
            episode_time_taken = time.time() - episode_start_time
            # save episode logs
            all_episode_logs["episode_number"].append(num_attempts+num_problematic)
            all_episode_logs["err_status"].append("problematic")
            all_episode_logs["time_taken"].append(episode_time_taken)
            all_episode_logs["task_success"].append(False)
            all_episode_logs["phases_completed"].append(-1)
            all_episode_logs["phase_logs"].append(dict())
            
            num_problematic += 1
            continue
        
        num_attempts += 1
        if write_video:
            video_writer.close()
        
        if env.err == "BaseMPFailed":
            base_mp_failures += 1
        elif env.err == "BaseMPIKFailed":
            base_mp_ik_failures += 1
        elif env.err == "ArmMPTrajOptFailed":
            arm_mp_trajopt_failures += 1
        elif env.err == "ArmMPIKFailed":
            arm_mp_ik_failures += 1
        elif env.err == "BaseSamplingFailed":   
            base_sampling_failures += 1
        elif env.err == "ArmMPOtherFailed":
            arm_mp_other_failures += 1
        
        if env.obj_visible_at_start_of_manip:
            obj_visible_at_start_of_manip += 1

        # generated_traj will be None if a) the 0th phase of the trajectory failed due to MP or b) no_partial_tasks is True meaning that any MP failure in any phase
        # is considered a failure and is not saved in either the success or failure hdf5 file.
        invalid_traj = generated_traj is None or len(generated_traj["states"]) == 0
        if invalid_traj:
            success = False
            num_failures += 1
        else:
            success = env.is_success()["task"]
            if success:
                num_success += 1
            else:
                num_failures += 1

        print("")
        print("*" * 50)
        print("trial {} success: {}".format(num_attempts, success))
        print("have {} successes out of {} trials so far".format(num_success, num_attempts))
        print("have {} failures out of {} trials so far".format(num_failures, num_attempts))
        print('have {} Base MP failures, {} Arm MP IK failures, {} Arm MP TrajOpt failures, {} Arm MP other failures, {} Base sampling failures, {} Base MP IK failures'.format(base_mp_failures, arm_mp_ik_failures, arm_mp_trajopt_failures, arm_mp_other_failures, base_sampling_failures, base_mp_ik_failures))
        print("*" * 50)

        if success:
            # store successful demonstration
            ep_lengths.append(generated_traj["actions"].shape[0])
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_dataset_folder_path,
                env=env,
                initial_state=generated_traj["initial_state"],
                states=generated_traj["states"],
                observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                observations_info=generated_traj["observations_info"],
                datagen_info=generated_traj["datagen_infos"],
                actions=generated_traj["actions"],
                src_demo_inds=generated_traj["src_demo_inds"],
                src_demo_labels=generated_traj["src_demo_labels"],
                mp_end_steps=generated_traj["mp_end_steps"],
                subtask_lengths=generated_traj["subtask_lengths"],
                sensor_info=generated_traj["sensor_info"],
                episode_time_taken=episode_time_taken,
                partial=generated_traj["partial"],
                left_mp_ranges=generated_traj["left_mp_ranges"],
                right_mp_ranges=generated_traj["right_mp_ranges"],
            )
        else:
            keep_failed = mg_config.experiment.generation.keep_failed
            less_than_max_failures = (mg_config.experiment.max_num_failures is None) or (num_failures <= mg_config.experiment.max_num_failures)
            # check if this failure should be kept
            if keep_failed and less_than_max_failures and not invalid_traj:
                # save failed trajectory in separate folder
                MG_FileUtils.write_demo_to_hdf5(
                    folder=tmp_dataset_failed_folder_path,
                    env=env,
                    initial_state=generated_traj["initial_state"],
                    states=generated_traj["states"],
                    observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                    observations_info=generated_traj["observations_info"],
                    datagen_info=generated_traj["datagen_infos"],
                    actions=generated_traj["actions"],
                    src_demo_inds=generated_traj["src_demo_inds"],
                    src_demo_labels=generated_traj["src_demo_labels"],
                    mp_end_steps=generated_traj["mp_end_steps"],
                    subtask_lengths=generated_traj["subtask_lengths"],
                    sensor_info=generated_traj["sensor_info"],
                    episode_time_taken=episode_time_taken,
                    partial=generated_traj["partial"],
                    left_mp_ranges=generated_traj["left_mp_ranges"],
                    right_mp_ranges=generated_traj["right_mp_ranges"],
                )

        # regularly log progress to disk every so often
        if (num_attempts % mg_config.experiment.log_every_n_attempts) == 0:
            # get summary stats
            summary_stats = get_important_stats(
                new_dataset_folder_path=new_dataset_folder_path,
                num_success=num_success,
                num_failures=num_failures,
                num_attempts=num_attempts,
                num_problematic=num_problematic,
                ep_lengths=ep_lengths,
                start_time=script_start_time,
                ep_length_stats=None,
                all_episode_logs=all_episode_logs,
            )

            # write stats to disk
            max_digits = len(str(num_trials * 1000)) + 1 # assume we will never have lower than 0.1% data generation SR
            json_file_path = os.path.join(json_log_path, "attempt_{}_succ_{}_rate_{}.json".format(
                str(num_attempts).zfill(max_digits), # pad with leading zeros for ordered list of jsons in directory
                num_success,
                np.round((100. * num_success) / num_attempts, 2),
            ))
            MG_FileUtils.write_json(json_dic=summary_stats, json_path=json_file_path)

        # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
        check_val = num_success if guarantee_success else num_attempts
        if check_val >= num_trials:
            break


    # save episode logs
    with open(os.path.join(new_dataset_folder_path, "episode_logs.json"), "w") as f:
        json.dump(all_episode_logs, f, indent=4)
    
    # merge all new created files
    print("\nFinished data generation. Merging per-episode hdf5s together...\n")
    MG_FileUtils.merge_all_hdf5(
        folder=tmp_dataset_folder_path,
        new_hdf5_path=new_dataset_path,
        delete_folder=True,
    )
    if mg_config.experiment.generation.keep_failed:
        MG_FileUtils.merge_all_hdf5(
            folder=tmp_dataset_failed_folder_path,
            new_hdf5_path=new_failed_dataset_path,
            delete_folder=True,
        )

    # get episode length statistics
    ep_length_stats = None
    if len(ep_lengths) > 0:
        ep_length_mean = float(np.mean(ep_lengths))
        ep_length_std = float(np.std(ep_lengths))
        ep_length_max = int(np.max(ep_lengths))
        ep_length_3std = int(np.ceil(ep_length_mean + 3. * ep_length_std))
        ep_length_stats = dict(
            ep_length_mean=ep_length_mean,
            ep_length_std=ep_length_std,
            ep_length_max=ep_length_max,
            ep_length_3std=ep_length_3std,
        )

    stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        start_time=script_start_time,
        ep_length_stats=ep_length_stats,
    )
    print("\nStats Summary")
    print(json.dumps(stats, indent=4))

    # maybe render videos
    if mg_config.experiment.render_video:
        if (num_success > 0):
            playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}.mp4".format(new_dataset_folder_name))
            num_render = mg_config.experiment.num_demo_to_render
            print("Rendering successful trajectories...")
            RobomimicUtils.make_dataset_video(
                dataset_path=new_dataset_path,
                video_path=playback_video_path,
                num_render=num_render,
            )
        else:
            print("\n" + "*" * 80)
            print("\nWARNING: skipping dataset video creation since no successes")
            print("\n" + "*" * 80 + "\n")
        if mg_config.experiment.generation.keep_failed:
            if (num_failures > 0):
                playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}_failed.mp4".format(new_dataset_folder_name))
                num_render = mg_config.experiment.num_fail_demo_to_render
                print("Rendering failure trajectories...")
                RobomimicUtils.make_dataset_video(
                    dataset_path=new_failed_dataset_path,
                    video_path=playback_video_path,
                    num_render=num_render,
                )
            else:
                print("\n" + "*" * 80)
                print("\nWARNING: skipping dataset video creation since no failures")
                print("\n" + "*" * 80 + "\n")

    # return some summary info
    final_important_stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        start_time=script_start_time,
        ep_length_stats=ep_length_stats,
        all_episode_logs=all_episode_logs,
    )

    # write stats to disk
    json_file_path = os.path.join(new_dataset_folder_path, "important_stats.json")
    MG_FileUtils.write_json(json_dic=final_important_stats, json_path=json_file_path)


    if env_meta["type"] == EnvUtils.EB.EnvType.OG_TYPE:
        og.shutdown()

    return final_important_stats


def main(args):

    # load config object
    with open(args.config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by MoMaGen
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

    # update config with external json - this will throw errors if
    # the external config has keys not present in the base config
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

        # We assume that the external config specifies all subtasks, so
        # delete any subtasks not in the external config.
        source_subtasks = set(mg_config.task.task_spec.keys())
        new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
        for subtask in (source_subtasks - new_subtasks):
            print("deleting subtask {} in original config".format(subtask))
            del mg_config.task.task_spec[subtask]

        # maybe override some settings
        if args.task_name is not None:
            mg_config.experiment.task.name = args.task_name

        if args.source is not None:
            mg_config.experiment.source.dataset_path = args.source

        if args.folder is not None:
            mg_config.experiment.generation.path = args.folder

        if args.num_demos is not None:
            mg_config.experiment.generation.num_trials = args.num_demos

        if args.seed is not None:
            mg_config.experiment.seed = args.seed

        # maybe modify config for debugging purposes
        if args.debug:
            # shrink length of generation to test whether this run is likely to crash
            mg_config.experiment.source.n = 3
            mg_config.experiment.generation.guarantee = False
            mg_config.experiment.generation.num_trials = 2

            # send output to a temporary directory
            mg_config.experiment.generation.path = "/tmp/tmp_momagen"

    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = generate_dataset(
            mg_config=mg_config,
            auto_remove_exp=args.auto_remove_exp,
            render=args.render,
            no_save_video=args.no_video_save,
            video_skip=args.video_skip,
            render_image_names=args.render_image_names,
            pause_subtask=args.pause_subtask,
            bimanual=args.bimanual,
            enable_marker_vis=args.enable_marker_vis,
            ds_ratio=args.ds_ratio,
            no_partial_tasks=args.no_partial_tasks,
            headless=args.headless,
            baseline=args.baseline,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        important_stats = json.dumps(important_stats, indent=4)
        print("\nFinal Data Generation Stats")
        print(important_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to MoMaGen config json",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick generation run for debugging purposes",
    )
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )
    parser.add_argument(
        "--bimanual",
        action='store_true',
        help="force the code to use bimanual setup"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="render each data generation attempt on-screen",
    )
    parser.add_argument(
        "--no_video_save",
        action='store_true',
        help="if provided, don't save video of data generation attempts",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="skip every nth frame when writing video",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    parser.add_argument(
        "--pause_subtask",
        action='store_true',
        help="pause after every subtask during generation for debugging - only useful with render flag",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="path to source dataset, to override the one in the config",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="environment name to use for data generation, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="folder that will be created with new data, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        help="number of demos to generate, or attempt to generate, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--headless",
        action='store_true',
        help="whether to generate data in headless mode",
    )
    parser.add_argument(
        "--enable_marker_vis",
        action='store_true',
        help="enable the marker visualization when generating data, the markers are mainly for vis the eef pose and target pose",
    )
    parser.add_argument(
        "--ds_ratio",
        type=int,
        help="downsample rate for the replay data",
        default=None,
    )
    parser.add_argument(
        "--no_partial_tasks",
        action='store_true',
        help="disable the marker visualization when generating data, the markers are mainly for vis the eef pose and target pose",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="baseline to run. Options: mimicgen or skillgen",
        default=None,
    )

    args = parser.parse_args()
    main(args)
