"""
A collection of utilities related to files.
"""
import os
import h5py
import json
import time
import datetime
import shutil
import shlex
import tempfile
import numpy as np
import copy

import torch as th
from glob import glob
from tqdm import tqdm

import robomimic
import robomimic.utils.tensor_utils as TensorUtils

from momagen.datagen.datagen_info import DatagenInfo


def write_json(json_dic, json_path):
    """
    Write dictionary to json file.
    """
    with open(json_path, 'w') as f:
        # preserve original key ordering
        json.dump(json_dic, f, sort_keys=False, indent=4)


def get_all_demos_from_dataset(
    dataset_path,
    filter_key=None,
    start=None,
    n=None,
):
    """
    Helper function to get demonstration keys from robomimic hdf5 dataset.

    Args:
        dataset_path (str): path to hdf5 dataset
        filter_key (str or None): name of filter key
        start (int or None): demonstration index to start from
        n (int or None): number of consecutive demonstrations to retrieve

    Returns:
        demo_keys (list): list of demonstration keys
    """
    f = h5py.File(dataset_path, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demo_keys = [demos[i] for i in inds]
    if start is not None:
        demo_keys = demo_keys[start:]
    if n is not None:
        demo_keys = demo_keys[:n]

    f.close()
    return demo_keys


def get_env_interface_info_from_dataset(
    dataset_path,
    demo_keys,
):
    """
    Gets environment interface information from source dataset.

    Args:
        dataset_path (str): path to hdf5 dataset
        demo_keys (list): list of demonstration keys to extract info from

    Returns:
        env_interface_name (str): name of environment interface class
        env_interface_type (str): type of environment interface
    """
    f = h5py.File(dataset_path, "r")
    env_interface_names = []
    env_interface_types = []
    for ep in demo_keys:
        datagen_info_key = "data/{}/datagen_info".format(ep)
        assert datagen_info_key in f, "Could not find MimicGen metadata in dataset {}. Ensure you have run prepare_src_dataset.py on this hdf5".format(dataset_path)
        env_interface_names.append(f[datagen_info_key].attrs["env_interface_name"])
        env_interface_types.append(f[datagen_info_key].attrs["env_interface_type"])
    f.close()

    # ensure all source demos are consistent
    env_interface_name = env_interface_names[0]
    env_interface_type = env_interface_types[0]
    assert all(elem == env_interface_name for elem in env_interface_names)
    assert all(elem == env_interface_type for elem in env_interface_types)
    return env_interface_name, env_interface_type


def parse_source_dataset(
    dataset_path,
    demo_keys,
    task_spec=None,
    subtask_term_signals=None,
    subtask_term_offset_ranges=None,
):
    """
    Parses a source dataset to extract info needed for data generation (DatagenInfo instances) and 
    subtask indices that split each source dataset trajectory into contiguous subtask segments.

    Args:
        dataset_path (str): path to hdf5 dataset
        demo_keys (list): list of demo keys to use from dataset path
        task_spec (MG_TaskSpec instance or None): task spec object, which will be used to
            infer the sequence of subtask termination signals and offset ranges.
        subtask_term_signals (list or None): sequence of subtask termination signals, which 
            should only be provided if not providing @task_spec. Should have an entry per subtask 
            and the last subtask entry should be None, since the final subtask ends when the 
            task ends.
        subtask_term_offset_ranges (list or None): sequence of subtask termination offset ranges, which 
            should only be provided if not providing @task_spec. Should have an entry per subtask 
            and the last subtask entry should be None or (0, 0), since the final subtask ends when the 
            task ends.

    Returns:

        datagen_infos (list): list of DatagenInfo instances, one per source
            demonstration. Each instance has entries with leading dimension [T, ...], 
            the length of the trajectory.

        subtask_indices (np.array): array of shape (N, S, 2) where N is the number of
                demos and S is the number of subtasks for this task. Each entry is
                a pair of integers that represents the index at which a subtask 
                segment starts and where it is completed.

        subtask_term_signals (list): sequence of subtask termination signals

        subtask_term_offset_ranges (list): sequence of subtask termination offset ranges
    """

    # should provide either task_spec or the subtask termination lists, but not both
    assert (task_spec is not None) or ((subtask_term_signals is not None) and (subtask_term_offset_ranges is not None))
    assert (task_spec is None) or ((subtask_term_signals is None) and (subtask_term_offset_ranges is None))

    if task_spec is not None:
        subtask_term_signals = [subtask_spec["subtask_term_signal"] for subtask_spec in task_spec]
        subtask_term_offset_ranges = [subtask_spec["subtask_term_offset_range"] for subtask_spec in task_spec]

    assert len(subtask_term_signals) == len(subtask_term_offset_ranges)
    assert subtask_term_signals[-1] is None, "end of final subtask does not need to be detected"
    assert (subtask_term_offset_ranges[-1] is None) or (subtask_term_offset_ranges[-1] == (0, 0)), "end of final subtask does not need to be detected"
    subtask_term_offset_ranges[-1] = (0, 0)

    f = h5py.File(dataset_path, "r")

    datagen_infos = []
    subtask_indices = []
    for ind in tqdm(range(len(demo_keys))):
        ep = demo_keys[ind]
        ep_grp = f["data/{}".format(ep)]

        # extract datagen info
        ep_datagen_info = ep_grp["datagen_info"]
        ep_datagen_info_obj = DatagenInfo(
            eef_pose=ep_datagen_info["eef_pose"][:],
            object_poses={ k : ep_datagen_info["object_poses"][k][:] for k in ep_datagen_info["object_poses"] },
            subtask_term_signals={ k : ep_datagen_info["subtask_term_signals"][k][:] for k in ep_datagen_info["subtask_term_signals"] },
            target_pose=ep_datagen_info["target_pose"][:],
            gripper_action=ep_datagen_info["gripper_action"][:],
        )
        datagen_infos.append(ep_datagen_info_obj)

        # parse subtask indices using subtask termination signals
        ep_subtask_indices = []
        prev_subtask_term_ind = 0
        for subtask_ind in range(len(subtask_term_signals)):
            subtask_term_signal = subtask_term_signals[subtask_ind]
            if subtask_term_signal is None:
                # final subtask, finishes at end of demo
                # OG uses "action" rather than "actions"
                action = ep_grp["actions"] if "actions" in ep_grp else ep_grp["action"]
                subtask_term_ind = action.shape[0]
            else:
                # trick to detect index where first 0 -> 1 transition occurs - this will be the end of the subtask
                subtask_indicators = ep_datagen_info_obj.subtask_term_signals[subtask_term_signal]
                diffs = subtask_indicators[1:] - subtask_indicators[:-1]
                end_ind = int(diffs.nonzero()[0][0]) + 1
                subtask_term_ind = end_ind + 1 # increment to support indexing like demo[start:end]
            ep_subtask_indices.append([prev_subtask_term_ind, subtask_term_ind])
            prev_subtask_term_ind = subtask_term_ind

        # run sanity check on subtask_term_offset_range in task spec to make sure we can never
        # get an empty subtask in the worst case when sampling subtask bounds:
        #
        #   end index of subtask i + max offset of subtask i < end index of subtask i + 1 + min offset of subtask i + 1
        #
        assert len(ep_subtask_indices) == len(subtask_term_signals), "mismatch in length of extracted subtask info and number of subtasks"
        for i in range(1, len(ep_subtask_indices)):
            prev_max_offset_range = subtask_term_offset_ranges[i - 1][1]
            # TODO: okay right here it is assuming that the different subtasks are sequentially ordered, the signals will have to be ordered in the same way,
            # TODO: the grasp signal is not detecting change, it is detecting when the grasp is active or whether the touch is active, 
            assert ep_subtask_indices[i - 1][1] + prev_max_offset_range < ep_subtask_indices[i][1] + subtask_term_offset_ranges[i][0], \
                "subtask sanity check violation in demo key {} with subtask {} end ind {}, subtask {} max offset {}, subtask {} end ind {}, and subtask {} min offset {}".format(
                    demo_keys[ind], i - 1, ep_subtask_indices[i - 1][1], i - 1, prev_max_offset_range, i, ep_subtask_indices[i][1], i, subtask_term_offset_ranges[i][0])

        subtask_indices.append(ep_subtask_indices)
    f.close()

    # convert list of lists to array for easy indexing
    subtask_indices = np.array(subtask_indices)

    return datagen_infos, subtask_indices, subtask_term_signals, subtask_term_offset_ranges


def parse_source_dataset_bimanual(
    dataset_path,
    demo_keys,
    task_spec=None,
    subtask_term_signals=None,
    subtask_term_offset_ranges=None,
):
    """
    Parses a source dataset to extract info needed for data generation (DatagenInfo instances) and 
    subtask indices that split each source dataset trajectory into contiguous subtask segments.

    Args:
        dataset_path (str): path to hdf5 dataset
        demo_keys (list): list of demo keys to use from dataset path
        task_spec (MG_TaskSpec instance or None): task spec object, which will be used to
            infer the sequence of subtask termination signals and offset ranges.
        subtask_term_signals (list or None): sequence of subtask termination signals, which 
            should only be provided if not providing @task_spec. Should have an entry per subtask 
            and the last subtask entry should be None, since the final subtask ends when the 
            task ends.
        subtask_term_offset_ranges (list or None): sequence of subtask termination offset ranges, which 
            should only be provided if not providing @task_spec. Should have an entry per subtask 
            and the last subtask entry should be None or (0, 0), since the final subtask ends when the 
            task ends.

    Returns:

        datagen_infos (list): list of DatagenInfo instances, one per source
            demonstration. Each instance has entries with leading dimension [T, ...], 
            the length of the trajectory.

        subtask_indices (np.array): array of shape (N, S, 2) where N is the number of
                demos and S is the number of subtasks for this task. Each entry is
                a pair of integers that represents the index at which a subtask 
                segment starts and where it is completed.

        subtask_term_signals (list): sequence of subtask termination signals

        subtask_term_offset_ranges (list): sequence of subtask termination offset ranges
    """
    # get saved data information

    f = h5py.File(dataset_path, "r")

    datagen_infos = []
    subtask_indices = []
    demo_lens = []
    actions = []
    for ind in tqdm(range(len(demo_keys))):
        ep = demo_keys[ind]
        ep_grp = f["data/{}".format(ep)]

        # extract datagen info
        # Only record eef_pose that are actually achieved, not the target_pose
        ep_datagen_info = ep_grp["datagen_info"]
        ep_datagen_info_obj = DatagenInfo(
            eef_pose=ep_datagen_info["eef_pose"][:],
            object_poses={ k : ep_datagen_info["object_poses"][k][:] for k in ep_datagen_info["object_poses"] },
            subtask_term_signals=None if "subtask_term_signals" not in ep_datagen_info else { k : ep_datagen_info["subtask_term_signals"][k][:] for k in ep_datagen_info["subtask_term_signals"] },
            # subtask_term_signals={ k : ep_datagen_info["subtask_term_signals"][k][:] for k in ep_datagen_info["subtask_term_signals"] },
            # target_pose=ep_datagen_info["target_pose"][:],
            gripper_action=ep_datagen_info["gripper_action"][:],
        )
        datagen_infos.append(ep_datagen_info_obj)
        # OG uses "action" rather than "actions"
        action = ep_grp["actions"] if "actions" in ep_grp else ep_grp["action"]
        actions.append(np.array(action))
        num_steps = action.shape[0]
        demo_lens.append(num_steps)

    f.close()

    # get subtask info
    task_spec_all = copy.deepcopy(task_spec)

    def get_arm_spec_info(task_spec, demo_keys, demo_lens, prev_subtask_term_ind):
        # checking for each arm 

        subtask_term_signals = [subtask_spec["subtask_term_signal"] for subtask_spec in task_spec]
        subtask_term_offset_ranges = [subtask_spec["subtask_term_offset_range"] for subtask_spec in task_spec]

        assert len(subtask_term_signals) == len(subtask_term_offset_ranges)
        assert (subtask_term_offset_ranges[-1] is None) or (subtask_term_offset_ranges[-1] == (0, 0)), "end of final subtask does not need to be detected"
        subtask_term_offset_ranges[-1] = (0, 0)

        subtask_indices = []
        for ind in tqdm(range(len(demo_keys))):

            # parse subtask indices using subtask termination signals
            ep_subtask_indices = []
            for subtask_ind in range(len(subtask_term_signals)):
                subtask_term_step = task_spec[subtask_ind]["subtask_term_step"]

                if subtask_term_step is None:
                    # final subtask, finishes at end of demo
                    # OG uses "action" rather than "actions"
                    subtask_term_ind = demo_lens[ind]
                else:
                    subtask_term_ind = subtask_term_step
                ep_subtask_indices.append([prev_subtask_term_ind, subtask_term_ind])
                prev_subtask_term_ind = subtask_term_ind

            # run sanity check on subtask_term_offset_range in task spec to make sure we can never
            # get an empty subtask in the worst case when sampling subtask bounds:
            #
            #   end index of subtask i + max offset of subtask i < end index of subtask i + 1 + min offset of subtask i + 1
            #
            assert len(ep_subtask_indices) == len(subtask_term_signals), "mismatch in length of extracted subtask info and number of subtasks"
            for i in range(1, len(ep_subtask_indices)):
                prev_max_offset_range = subtask_term_offset_ranges[i - 1][1]
                # TODO: okay right here it is assuming that the different subtasks are sequentially ordered, the signals will have to be ordered in the same way,
                # TODO: the grasp signal is not detecting change, it is detecting when the grasp is active or whether the touch is active, 
                assert ep_subtask_indices[i - 1][1] + prev_max_offset_range < ep_subtask_indices[i][1] + subtask_term_offset_ranges[i][0], \
                    "subtask sanity check violation in demo key {} with subtask {} end ind {}, subtask {} max offset {}, subtask {} end ind {}, and subtask {} min offset {}".format(
                        demo_keys[ind], i - 1, ep_subtask_indices[i - 1][1], i - 1, prev_max_offset_range, i, ep_subtask_indices[i][1], i, subtask_term_offset_ranges[i][0])

            subtask_indices.append(ep_subtask_indices)

        # convert list of lists to array for easy indexing
        subtask_indices = np.array(subtask_indices)

        print('before enfing of parsing the dataset')
        return subtask_indices, subtask_term_signals, subtask_term_offset_ranges, prev_subtask_term_ind
    
    subtask_indices = []
    subtask_term_signals = []
    subtask_term_offset_ranges = []
    prev_subtask_term_ind = 0
    num_phases = len(task_spec_all)
    for phase_index in range(num_phases):
        task_spec = task_spec_all[phase_index]
        
        subtask_indices_l, subtask_term_signals_l, subtask_term_offset_ranges_l, prev_subtask_term_ind_l = get_arm_spec_info(task_spec[0], demo_keys, demo_lens, prev_subtask_term_ind)
        subtask_indices_r, subtask_term_signals_r, subtask_term_offset_ranges_r, prev_subtask_term_ind_r= get_arm_spec_info(task_spec[1], demo_keys, demo_lens, prev_subtask_term_ind)

        assert prev_subtask_term_ind_l == prev_subtask_term_ind_r # the end point of phase for both arms should be the same
        prev_subtask_term_ind = prev_subtask_term_ind_l

        subtask_indices.append([])
        subtask_indices[-1].append(subtask_indices_l)
        subtask_indices[-1].append(subtask_indices_r)

        subtask_term_signals.append([])
        subtask_term_signals[-1].append(subtask_term_signals_l)
        subtask_term_signals[-1].append(subtask_term_signals_r)

        subtask_term_offset_ranges.append([])
        subtask_term_offset_ranges[-1].append(subtask_term_offset_ranges_l)
        subtask_term_offset_ranges[-1].append(subtask_term_offset_ranges_r)
    
    return datagen_infos, subtask_indices, subtask_term_signals, subtask_term_offset_ranges, actions


def write_demo_to_hdf5(
    folder,
    env,
    initial_state,
    states,
    observations,
    observations_info,
    datagen_info,
    actions,
    src_demo_inds=None,
    src_demo_labels=None,
    mp_end_steps=None,
    subtask_lengths=None,
    sensor_info=None,
    episode_time_taken=None,
    partial=False,
    left_mp_ranges=None,
    right_mp_ranges=None,
):
    """
    Helper function to write demonstration to an hdf5 file (robomimic format) in a folder. It will be 
    named using a timestamp.

    Args:
        folder (str): folder to write hdf5 to 
        env (robomimic EnvBase instance): simulation environment
        initial_state (dict): dictionary corresponding to initial simulator state (see robomimic dataset structure for more information)
        states (list): list of simulator states
        observations (list): list of observation dictionaries
        datagen_info (list): list of DatagenInfo instances
        actions (np.array): actions per timestep
        src_demo_inds (list or None): if provided, list of selected source demonstration indices for each subtask
        src_demo_labels (np.array or None): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
    """

    # name hdf5 based on timestamp
    timestamp = time.time()
    time_str = datetime.datetime.fromtimestamp(timestamp).strftime('date_%m_%d_%Y_time_%H_%M_%S')
    dataset_path = os.path.join(folder, "{}.hdf5".format(time_str))
    data_writer = h5py.File(dataset_path, "w")
    data_grp = data_writer.create_group("data")
    data_grp.attrs["timestamp"] = timestamp
    data_grp.attrs["readable_timestamp"] = time_str

    # single episode
    ep_data_grp = data_grp.create_group("demo_0")

    # write actions
    ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write simulator states
    if isinstance(states[0], dict):
        states = TensorUtils.list_of_flat_dict_to_dict_of_list(states)
        for k in states:
            ep_data_grp.create_dataset("states/{}".format(k), data=np.stack(states[k]))
    else:
        lens_states = [len(states[i]) for i in range(len(states))]
        # Pad the states to the same size in case they are not
        states_std = np.std(lens_states)
        if states_std > 0:
            max_state_size = max(lens_states)
            for i, state in enumerate(states):
                padded_state = th.zeros(max_state_size, dtype=th.float32)
                padded_state[: len(state)] = state
                states[i] = padded_state

        ep_data_grp.create_dataset("states", data=np.stack(states))

    # write observations
    if observations is not None:
        obs = TensorUtils.list_of_flat_dict_to_dict_of_list(observations)
        # TODO: check if we don't write seg_instance how much space do we save
        ignore_keys = ["robot_r1::robot_r1:left_eef_link:Camera:0::seg_instance", "robot_r1::robot_r1:right_eef_link:Camera:0::seg_instance", "robot_r1::robot_r1:eyes:Camera:0::seg_instance"]
        for k in obs:
            # Uncomment in case we don't want to write seg_instance
            # if k in ignore_keys:
            #     # ignore seg_instance
            #     continue
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.stack(obs[k]), compression="gzip")

    if observations_info is not None:
        # write observations info
        dt = h5py.string_dtype(encoding='utf-8')
        ep_data_grp.create_dataset("obs_info", data=np.array(observations_info, dtype=dt))

    # write datagen info
    if datagen_info is not None:
        datagen_info = TensorUtils.list_of_flat_dict_to_dict_of_list([x.to_dict() for x in datagen_info])
        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # convert list of dict to dict of list again
                datagen_info[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(datagen_info[k])
                for k2 in datagen_info[k]:
                    datagen_info[k][k2] = np.array(datagen_info[k][k2])
                    ep_data_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
            else:
                ep_data_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

    # maybe write which source demonstrations generated this episode
    if src_demo_inds is not None:
        ep_data_grp.create_dataset("src_demo_inds", data=np.array(src_demo_inds))
    if src_demo_labels is not None:
        ep_data_grp.create_dataset("src_demo_labels", data=np.array(src_demo_labels))
    if mp_end_steps is not None:
        ep_data_grp.create_dataset("mp_end_steps", data=np.array(mp_end_steps))
    if left_mp_ranges is not None:
        ep_data_grp.create_dataset("left_mp_ranges", data=np.array(left_mp_ranges))
    if right_mp_ranges is not None:
        ep_data_grp.create_dataset("right_mp_ranges", data=np.array(right_mp_ranges))
    if subtask_lengths is not None:
        ep_data_grp.create_dataset("subtask_lengths", data=np.array(subtask_lengths))
    if sensor_info is not None:
        for sensor in sensor_info:
        #     ep_data_grp.create_group(sensor)
            for k2 in sensor_info[sensor]:            
                ep_data_grp.create_dataset(f"sensor_info/{sensor}/{k2}", data=np.array(sensor_info[sensor][k2]))
    
    # todo: has bug in it
    # if external_sensor_info is not None:
    #     for k in external_sensor_info:
    #         ep_data_grp.create_dataset("external_sensor_info/{}".format(k), data=np.array(external_sensor_info[k]))

    # episode metadata
    if ("model" in initial_state) and (initial_state["model"] is not None):
        # only for robosuite envs
        ep_data_grp.attrs["model_file"] = initial_state["model"] # model xml for this episode
    ep_data_grp.attrs["num_samples"] = actions.shape[0] # number of transitions in this episode
    if episode_time_taken is not None:
        ep_data_grp.attrs["episode_time_taken"] = episode_time_taken # time taken to complete this episode
    if partial is not None:
        ep_data_grp.attrs["partial"] = partial # whether this task was partially completed

    # global metadata
    data_grp.attrs["total"] = actions.shape[0]
    if env is not None:
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    data_writer.close()


def merge_all_hdf5(
    folder,
    new_hdf5_path,
    delete_folder=False,
    dry_run=False,
    return_horizons=False,
):
    """
    Helper function to take all hdf5s in @folder and merge them into a single one.
    Returns the number of hdf5s that were merged.
    """
    source_hdf5s = glob(os.path.join(folder, "*.hdf5"))

    # print(source_hdf5s)
    print('len source hdf5s', len(source_hdf5s))


    # get all timestamps and sort files from lowest to highest
    timestamps = []
    filtered_source_hdf5s = []
    index = 0
    for source_hdf5_path in source_hdf5s:
        index += 1
        print('index', index)
        try:
            f = h5py.File(source_hdf5_path, "r")
        except Exception as e:
            print("WARNING: problem with file {}".format(source_hdf5_path))
            print("Exception: {}".format(e))
            continue
        try:
            # check if timestamp in file
            timestamps.append(f["data"].attrs["timestamp"])
            f.close()
        except Exception as e:
            print("WARNING: file {} does not have timestamp attribute".format(source_hdf5_path))
            continue
        filtered_source_hdf5s.append(source_hdf5_path)
        print("len filtered out one", len(filtered_source_hdf5s))

    assert len(timestamps) == len(filtered_source_hdf5s)
    inds = np.argsort(timestamps)
    sorted_hdf5s = [filtered_source_hdf5s[i] for i in inds]

    if dry_run:
        if return_horizons:
            horizons = []
            for source_hdf5_path in sorted_hdf5s:
                with h5py.File(source_hdf5_path, "r") as f:
                    horizons.append(f["data"].attrs["total"])
            return len(sorted_hdf5s), horizons
        return len(sorted_hdf5s)

    # write demos in order to new file
    f_new = h5py.File(new_hdf5_path, "w")
    f_new_grp = f_new.create_group("data")

    env_meta_str = None
    total = 0
    if return_horizons:
        horizons = []
    for i, source_hdf5_path in enumerate(sorted_hdf5s):
        with h5py.File(source_hdf5_path, "r") as f:
            # copy this episode over under a different name
            demo_str = "demo_{}".format(i)
            f.copy("data/demo_0", f_new_grp, name=demo_str)
            if return_horizons:
                horizons.append(f["data"].attrs["total"])
            total += f["data"].attrs["total"]
            if env_meta_str is None:
                env_meta_str = f["data"].attrs["env_args"]

    f_new["data"].attrs["total"] = total
    f_new["data"].attrs["env_args"] = env_meta_str if env_meta_str is not None else ""
    f_new.close()

    if delete_folder:
        print("removing folder at path {}".format(folder))
        shutil.rmtree(folder)

    if return_horizons:
        return len(sorted_hdf5s), horizons
    return len(sorted_hdf5s)




def config_generator_to_script_lines(generator, config_dir):
    """
    Takes a robomimic ConfigGenerator and uses it to
    generate a set of training configs, and a set of bash command lines 
    that correspond to each training run (one per config). Note that
    the generator's script_file will be overridden to be a temporary file that
    will be removed from disk.

    Args:
        generator (ConfigGenerator instance or list): generator(s)
            to use for generating configs and training runs

        config_dir (str): path to directory where configs will be generated

    Returns:
        config_files (list): a list of config files that were generated

        run_lines (list): a list of strings that are training commands, one per config
    """

    # make sure config dir exists
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # support one or more config generators
    if not isinstance(generator, list):
        generator = [generator]

    all_run_lines = []
    for gen in generator:

        # set new config directory by copying base config file from old location to new directory
        base_config_file = gen.base_config_file
        config_name = os.path.basename(base_config_file)
        new_base_config_file = os.path.join(config_dir, config_name)
        shutil.copyfile(
            base_config_file,
            new_base_config_file,
        )
        gen.base_config_file = new_base_config_file

        # we'll write script file to a temp dir and parse it from there to get the training commands
        with tempfile.TemporaryDirectory() as td:
            gen.script_file = os.path.join(td, "tmp.sh")

            # generate configs
            gen.generate()

            # collect training commands
            with open(gen.script_file, "r") as f:
                f_lines = f.readlines()
                run_lines = [line for line in f_lines if line.startswith("python")]
                all_run_lines += run_lines

        os.remove(gen.base_config_file)

    # get list of generated configs too
    config_files = []
    config_file_dict = dict()
    for line in all_run_lines:
        cmd = shlex.split(line)
        config_file_name = cmd[cmd.index("--config") + 1]
        config_files.append(config_file_name)
        assert config_file_name not in config_file_dict, "got duplicate config name {}".format(config_file_name)
        config_file_dict[config_file_name] = 1

    return config_files, all_run_lines
