"""
Script to extract information needed for data generation from low-dimensional simulation states
in a source dataset and add it to the source dataset. Basically a stripped down version of
dataset_states_to_obs.py script in the robomimic codebase, with a handful of modifications.

Example usage:

    # prepare a source dataset collected on robosuite Stack task
    python prepare_src_dataset.py --dataset /path/to/stack.hdf5 --env_interface MG_Stack --env_interface_type robosuite

    # prepare a source dataset collected on robosuite Square task, but only use first 10 demos, and write output to new hdf5
    python prepare_src_dataset.py --dataset /path/to/square.hdf5 --env_interface MG_Square --env_interface_type robosuite --n 10 --output /tmp/square_new.hdf5
"""
import shutil
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

import momagen.utils.file_utils as MG_FileUtils
from momagen.env_interfaces.base import make_interface
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm


def extract_datagen_info_from_trajectory(
    env,
    env_interface,
    initial_state,
    states,
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of robomimic EnvBase): environment

        env_interface (MG_EnvInterface instance): environment interface for some data generation operations

        initial_state (dict): initial simulation state to load

        states (np.array): array of simulation states to load to extract information

        actions (np.array): array of actions

    Returns:
        datagen_infos (dict): the datagen info objects across all timesteps represented as a dictionary of
            numpy arrays, for easy writes to an hdf5
    """
    assert isinstance(env, EnvBase)
    assert len(states) == actions.shape[0]

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    all_datagen_infos = []
    traj_len = len(states)
    for t in range(traj_len):
        # reset to state
        print('timestep:', t)
        env.reset_to({"states": states[t]})

        # extract datagen info as a dictionary
        # datagen_info is a dict with dict_keys(['eef_pose', 'object_poses', 'subtask_term_signals', 'target_pose', 'gripper_action'])
        datagen_info = env_interface.get_datagen_info(action=actions[t]).to_dict()
        all_datagen_infos.append(datagen_info)

    # convert list of dict to dict of list for datagen info dictionaries (for convenient writes to hdf5 dataset)
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)

    for k in all_datagen_infos:
        if k in ["object_poses", "subtask_term_signals"]:
            # convert list of dict to dict of list again
            all_datagen_infos[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos[k])
            # list to numpy array
            for k2 in all_datagen_infos[k]:
                all_datagen_infos[k][k2] = np.array(all_datagen_infos[k][k2])
        else:
            # list to numpy array
            all_datagen_infos[k] = np.array(all_datagen_infos[k])

    return all_datagen_infos


def prepare_src_dataset(
    dataset_path,
    env_interface_name,
    env_interface_type,
    filter_key=None,
    n=None,
    output_path=None,
):
    """
    Adds DatagenInfo object instance for each timestep in each source demonstration trajectory
    and stores it under the "datagen_info" key for each episode. Also store the @env_interface_name
    and @env_interface_type used in the attribute of each key. This information is used during
    MimicGen data generation.

    Args:
        dataset_path (str): path to input hdf5 dataset, which will be modified in-place unless
            @output_path is provided

        env_interface_name (str): name of environment interface class to use for this source dataset

        env_interface_type (str): type of environment interface to use for this source dataset

        filter_key (str or None): name of filter key

        n (int or None): if provided, stop after n trajectories are processed

        output_path (str or None): if provided, write a new hdf5 here instead of modifying the
            original dataset in-place
    """
    # maybe write to new file instead of modifying existing file in-place
    if output_path is not None:
        shutil.copy(dataset_path, output_path)
        dataset_path = output_path

    if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
        FileUtils.preprocess_omnigibson_dataset(dataset_path)

    # create environment that was to collect source demonstrations
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    print("")
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # create environment interface for us to grab relevant information from simulation at each timestep
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))
    print("")
    # some operations are env-type-specific
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    state_key = "state" if env_interface_type == "omnigibson" or env_interface_type == 'omnigibson_bimanual' else "states"
    action_key = "action" if env_interface_type == "omnigibson" or env_interface_type == 'omnigibson_bimanual' else "actions"

    # get list of source demonstration keys from source hdf5
    demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=filter_key,
        start=None,
        n=n,
    )

    print("File that will be modified with datagen info: {}".format(dataset_path))
    print("")

    # open file to modify it
    f = h5py.File(dataset_path, "a")

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]
        ep_grp = f["data/{}".format(ep)]

        # prepare states to reload from
        states = ep_grp[state_key][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = ep_grp.attrs["model_file"]
        # extract datagen info
        actions = ep_grp[action_key][()]
        datagen_info = extract_datagen_info_from_trajectory(
            env=env,
            env_interface=env_interface,
            initial_state=initial_state,
            states=states,
            actions=actions,
        )

        # delete old dategen info if it already exists
        if "datagen_info" in ep_grp:
            del ep_grp["datagen_info"]

        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # handle dict
                for k2 in datagen_info[k]:
                    ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
            else:
                ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

        # remember the env interface used too
        ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
        ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

    print("Modified {} trajectories to include datagen info.".format(len(demos)))
    f.close()

    # Properly shutdown omnigibson if needed
    if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
        import omnigibson as og
        og.shutdown()


def prepare_src_dataset_new(
    dataset_path,
    env_interface_name,
    env_interface_type,
    filter_key=None,
    n=None,
    output_path=None,
    save_dataset=False
):
    """
    Adds DatagenInfo object instance for each timestep in each source demonstration trajectory
    and stores it under the "datagen_info" key for each episode. Also store the @env_interface_name
    and @env_interface_type used in the attribute of each key. This information is used during
    MimicGen data generation.

    Args:
        dataset_path (str): path to input hdf5 dataset, which will be modified in-place unless
            @output_path is provided

        env_interface_name (str): name of environment interface class to use for this source dataset

        env_interface_type (str): type of environment interface to use for this source dataset

        filter_key (str or None): name of filter key

        n (int or None): if provided, stop after n trajectories are processed

        output_path (str or None): if provided, write a new hdf5 here instead of modifying the
            original dataset in-place
    """
    # maybe write to new file instead of modifying existing file in-place
    if output_path is not None:
        shutil.copy(dataset_path, output_path)
        # dataset_path = output_path

    if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
        FileUtils.preprocess_omnigibson_dataset(dataset_path)

    # create environment that was to collect source demonstrations
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    print("==== Using environment with the following metadata ====")
    print(env_meta)

    gm.ENABLE_TRANSITION_RULES = False
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=dataset_path,
        output_path=None,
        robot_obs_modalities=(),
        robot_sensor_config=None,
        external_sensors_config=None,
        n_render_iterations=1,
        only_successes=False,
        replay_state=True,
    )

    # create environment interface for us to grab relevant information from simulation at each timestep
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env,
    )
    print("Created environment interface: {}".format(env_interface))

    # get list of source demonstration keys from source hdf5
    demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=filter_key,
        start=None,
        n=n,
    )

    demo_ids = None
    # Only playback the demos filtered by the filter_key
    if filter_key is not None:
        demo_ids = [int(demo.split("_")[-1]) for demo in demos]

    print("File that will be modified with datagen info: {}".format(dataset_path))

    # # ========================== custom changes to save images for paper ==========================
    # import omnigibson as og
    # import torch as th

    # robot = env.robots[0]    
    # for material in robot.materials:
    #     material.diffuse_color_constant = th.tensor([0.0, 0.0, 0.0])
    
    # # Set camera
    # og.sim.viewer_camera.horizontal_aperture = 35.0
    # # from ipynb
    # # og.sim.viewer_camera.set_position_orientation(position=th.tensor([ 5.473, -2.686,  2.403]),orientation=th.tensor([ 0.364, -0.004, -0.004,  0.932]))
    # # modified
    # og.sim.viewer_camera.set_position_orientation(th.tensor([ 5.6319, -2.6868,  2.4037]), th.tensor([ 0.4077, -0.0031, -0.0013,  0.9131]))

    # # Add/Remove objects
    # obj = env.scene.object_registry("name", "fixed_window_glimdy_0")
    # obj.visible = False
    # for eef_link_name in robot.eef_link_names.values():
    #     robot.links[eef_link_name].visual_meshes["VisualSphere"].visible = False
    # # ============================================================================================

    all_datagen_info = env.playback_dataset(record_data=False, callback=env_interface.get_datagen_info, demo_ids=demo_ids)

    env.input_hdf5.close()
    
    if not save_dataset:
        print("Not saving the dataset. Only used to visualize the collected demo")
        return

    # open file to modify it
    f = h5py.File(output_path, "a")

    for ind in tqdm(range(len(demos))):
        ep = demos[ind]
        ep_grp = f["data/{}".format(ep)]

        datagen_info = all_datagen_info[ind]
        datagen_info = [info.to_dict() for info in datagen_info]

        # convert list of dict to dict of list for datagen info dictionaries (for convenient writes to hdf5 dataset)
        datagen_info = TensorUtils.list_of_flat_dict_to_dict_of_list(datagen_info)

        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # convert list of dict to dict of list again
                datagen_info[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(datagen_info[k])
                # list to numpy array
                for k2 in datagen_info[k]:
                    datagen_info[k][k2] = np.array(datagen_info[k][k2])
            else:
                # list to numpy array
                datagen_info[k] = np.array(datagen_info[k])

        # delete old dategen info if it already exists
        if "datagen_info" in ep_grp:
            del ep_grp["datagen_info"]

        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # handle dict
                for k2 in datagen_info[k]:
                    ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
            else:
                ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

        # remember the env interface used too
        ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
        ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

    print("Modified {} trajectories to include datagen info.".format(len(demos)))
    f.close()

    # Properly shutdown omnigibson if needed
    if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
        import omnigibson as og
        og.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset, which will be modified in-place",
    )
    parser.add_argument(
        "--env_interface",
        type=str,
        required=True,
        help="name of environment interface class to use for this source dataset",
    )
    parser.add_argument(
        "--env_interface_type",
        type=str,
        required=True,
        help="type of environment interface to use for this source dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) name of filter key, to select a subset of demo keys in the source hdf5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) path to output hdf5 dataset, instead of modifying existing dataset in-place",
    )
    parser.add_argument(
        "--save",
        action='store_true',
        help="if not passed, don't save the dataset. Only used to visualize the collected demo",
    )

    args = parser.parse_args()
    prepare_src_dataset_new(
        dataset_path=args.dataset,
        env_interface_name=args.env_interface,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=args.n,
        output_path=args.output,
        save_dataset=args.save,
    )
