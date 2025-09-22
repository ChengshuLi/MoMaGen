"""
A collection of utilities for working with config generators. These generators 
are re-used from robomimic (https://robomimic.github.io/docs/tutorials/hyperparam_scan.html)
"""
import json
from collections.abc import Iterable


def set_debug_settings(
    generator,
    group,
):
    """
    Sets config generator parameters for a quick debug run.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
    """
    generator.add_param(
        key="experiment.generation.guarantee",
        name="", 
        group=group, 
        values=[False],
    )
    generator.add_param(
        key="experiment.generation.num_trials",
        name="", 
        group=group, 
        values=[2],
    )
    return generator


def set_basic_settings(
    generator,
    group,
    source_dataset_path,
    source_dataset_name,
    generation_path,
    guarantee,
    num_traj,
    num_src_demos=None,
    max_num_failures=25,
    num_demo_to_render=10,
    num_fail_demo_to_render=25,
    render_video=True,
    verbose=False,
):
    """
    Sets config generator parameters for some basic data generation settings.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        source_dataset_path (str): path to source dataset
        source_dataset_name (str): name to give source dataset in experiment name
        generation_path (str): folder for generated data
        guarantee (bool): whether to ensure @num_traj successes
        num_traj (int): number of trajectories for generation
        num_src_demos (int or None): number of source demos to take from @source_dataset_path
        max_num_failures (int): max failures to keep
        num_demo_to_render (int): max demos to render to video
        num_fail_demo_to_render (int): max fail demos to render to video
        render_video (bool): whether to render videos
        verbose (bool): if True, make experiment name verbose using the passed settings
    """

    # set source dataset
    generator.add_param(
        key="experiment.source.dataset_path",
        name="src" if source_dataset_name is not None else "",
        group=group,
        values=[source_dataset_path],
        value_names=[source_dataset_name],
    )

    # set number of demos to use from source dataset
    generator.add_param(
        key="experiment.source.n",
        name="n_src" if verbose else "",
        group=group,
        values=[num_src_demos],
    )

    # set generation settings
    generator.add_param(
        key="experiment.generation.path",
        name="", 
        group=group, 
        values=[generation_path],
    )
    generator.add_param(
        key="experiment.generation.guarantee",
        name="gt" if verbose else "", 
        group=group, 
        values=[guarantee],
        value_names=["t" if guarantee else "f"],
    )
    generator.add_param(
        key="experiment.generation.num_trials",
        name="nt" if verbose else "", 
        group=group, 
        values=[num_traj],
    )
    generator.add_param(
        key="experiment.max_num_failures",
        name="", 
        group=group, 
        values=[max_num_failures],
    )
    generator.add_param(
        key="experiment.num_demo_to_render",
        name="", 
        group=group, 
        values=[num_demo_to_render],
    )
    generator.add_param(
        key="experiment.num_fail_demo_to_render",
        name="", 
        group=group, 
        values=[num_fail_demo_to_render],
    )
    generator.add_param(
        key="experiment.render_video",
        name="",
        group=group,
        values=[render_video],
    )

    return generator


def set_obs_settings(
    generator,
    group,
    collect_obs,
    camera_names,
    camera_height,
    camera_width,
):
    """
    Sets config generator parameters for collecting observations.
    """
    generator.add_param(
        key="obs.collect_obs",
        name="", 
        group=group, 
        values=[collect_obs],
    )
    generator.add_param(
        key="obs.camera_names",
        name="", 
        group=group, 
        values=[camera_names],
    )
    generator.add_param(
        key="obs.camera_height",
        name="", 
        group=group, 
        values=[camera_height],
    )
    generator.add_param(
        key="obs.camera_width",
        name="", 
        group=group, 
        values=[camera_width],
    )
    return generator


def set_subtask_settings(
    generator,
    group,
    base_config_file,
    select_src_per_subtask,
    subtask_term_offset_range=None,
    selection_strategy=None,
    selection_strategy_kwargs=None,
    action_noise=None,
    num_interpolation_steps=None,
    num_fixed_steps=None,
    verbose=False,
):
    """
    Sets config generator parameters for each subtask.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        base_config_file (str): path to base config file being used for generating configs
        select_src_per_subtask (bool): whether to select src demo for each subtask
        subtask_term_offset_range (list or None): if provided, should be list of 2-tuples, one
            entry per subtask, with the last entry being None
        selection_strategy (str or None): src demo selection strategy
        selection_strategy_kwargs (dict or None): kwargs for selection strategy
        action_noise (float or list or None): action noise for all subtasks
        num_interpolation_steps (int or list or None): interpolation steps for all subtasks
        num_fixed_steps (int or list or None): interpolation steps for all subtasks
        verbose (bool): if True, make experiment name verbose using the passed settings
    """

    # get number of subtasks
    with open(base_config_file, 'r') as f:
        config = json.load(f)
        num_subtasks = len(config["task"]["task_spec"])

    # whether to select a different source demonstration for each subtask
    generator.add_param(
        key="experiment.generation.select_src_per_subtask",
        name="select_src_per_subtask" if verbose else "",
        group=group,
        values=[select_src_per_subtask],
        value_names=["t" if select_src_per_subtask else "f"],
    )

    # settings for each subtask

    # offset range
    if subtask_term_offset_range is not None:
        assert len(subtask_term_offset_range) == num_subtasks
        for i in range(num_subtasks):
            if (i == num_subtasks - 1):
                assert subtask_term_offset_range[i] is None
            else:
                assert (subtask_term_offset_range[i] is None) or (len(subtask_term_offset_range[i]) == 2)
            generator.add_param(
                key="task.task_spec.subtask_{}.subtask_term_offset_range".format(i + 1),
                name="offset" if (verbose and (i == 0)) else "", 
                group=group,
                values=[subtask_term_offset_range[i]],
            )

    # selection strategy
    if selection_strategy is not None:
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.selection_strategy".format(i + 1),
                name="ss" if (verbose and (i == 0)) else "", 
                group=group,
                values=[selection_strategy],
            )

    # selection kwargs
    if selection_strategy_kwargs is not None:
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.selection_strategy_kwargs".format(i + 1),
                name="", 
                group=group,
                values=[selection_strategy_kwargs],
            )

    # action noise
    if action_noise is not None:
        if not isinstance(action_noise, Iterable):
            action_noise = [action_noise for _ in range(num_subtasks)]
        assert len(action_noise) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.action_noise".format(i + 1),
                name="noise" if (verbose and (i == 0)) else "", 
                group=group,
                values=[action_noise[i]],
            )

    # interpolation
    if num_interpolation_steps is not None:
        if not isinstance(num_interpolation_steps, Iterable):
            num_interpolation_steps = [num_interpolation_steps for _ in range(num_subtasks)]
        assert len(num_interpolation_steps) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.num_interpolation_steps".format(i + 1),
                name="ni" if (verbose and (i == 0)) else "", 
                group=group,
                values=[num_interpolation_steps[i]],
            )
    if num_fixed_steps is not None:
        if not isinstance(num_fixed_steps, Iterable):
            num_fixed_steps = [num_fixed_steps for _ in range(num_subtasks)]
        assert len(num_fixed_steps) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.num_fixed_steps".format(i + 1),
                name="ni" if (verbose and (i == 0)) else "", 
                group=group,
                values=[num_fixed_steps[i]],
            )




def set_subtask_settings_bimanual(
    generator,
    group,
    base_config_file,
    select_src_per_subtask,
    verbose=False,
):
    # TODO: the generate core config script may not be necesasry, why need to load multiple times?
    """
    Sets config generator parameters for each subtask.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        base_config_file (str): path to base config file being used for generating configs
        select_src_per_subtask (bool): whether to select src demo for each subtask
        subtask_term_offset_range (list or None): if provided, should be list of 2-tuples, one
            entry per subtask, with the last entry being None
        selection_strategy (str or None): src demo selection strategy
        selection_strategy_kwargs (dict or None): kwargs for selection strategy
        action_noise (float or list or None): action noise for all subtasks
        num_interpolation_steps (int or list or None): interpolation steps for all subtasks
        num_fixed_steps (int or list or None): interpolation steps for all subtasks
        verbose (bool): if True, make experiment name verbose using the passed settings
    """
    # currently matching v3 config, with phase in the subtask setting
    # config architecture
    # - phase_1
    #   - arm_left
    #     - subtask_1
    #     - subtask_2
    #     - ...
    #   - arm_right
    #     - subtask_1
    #     - subtask_2
    #     - ...

    # get number of subtasks
    with open(base_config_file, 'r') as f:
        config = json.load(f)
        num_subtasks = len(config["task"]["task_spec"])

    # whether to select a different source demonstration for each subtask
    generator.add_param(
        key="experiment.generation.select_src_per_subtask",
        name="select_src_per_subtask" if verbose else "",
        group=group,
        values=[select_src_per_subtask],
        value_names=["t" if select_src_per_subtask else "f"],
    )

    num_phases = len(config["task"]["task_spec"])
    # settings for each subtask

    for phase_index in range(num_phases):
        left_subtask_configs = config["task"]["task_spec"][f"phase_{phase_index+1}"]["arm_left"]
        num_subtasks_left = len(left_subtask_configs)

        right_subtask_configs = config["task"]["task_spec"][f"phase_{phase_index+1}"]["arm_right"]
        num_subtasks_right = len(right_subtask_configs)


        # reading randomization range offset from the config file 
        for i in range(num_subtasks_left):

            cur_range_l = left_subtask_configs[f"subtask_{int(i+1)}"]["subtask_term_offset_range"]
            print(i, cur_range_l)
            if (i == num_subtasks_left - 1):
                    assert cur_range_l is None
            else:
                assert (cur_range_l is None) or (len(cur_range_l) == 2)
            generator.add_param(
                    key="task.task_spec.phase_{}.arm_left.subtask_{}.subtask_term_offset_range".format(phase_index+1, i+1),
                    name="offset" if (verbose and (i == 0)) else "", 
                    group=group,
                    values=[cur_range_l],
                )
            
            name_mapping = {
                "selection_strategy": "selection_strategy",
                "selection_strategy_kwargs": "selection_strategy_kwargs",
                "action_noise": "noise",
                "num_interpolation_steps":"ni",
                "num_fixed_steps": "ni",
                "subtask_term_step": "subtask_term_step",
                "MP_end_step": "MP_end_step",
                "attached_obj": "attached_obj",
            }
            for kwargs in name_mapping.keys():
                generator.add_param(
                    key="task.task_spec.phase_{}.arm_left.subtask_{}.{}".format(phase_index+1, i+1, kwargs),
                    name=name_mapping[f"{kwargs}"] if (verbose and (i == 0)) else "", 
                    group=group,
                    values=[left_subtask_configs[f"subtask_{int(i+1)}"][kwargs]],
                )
            
        for i in range(num_subtasks_right):
            cur_range_r = right_subtask_configs[f"subtask_{int(i+1)}"]["subtask_term_offset_range"]
            print(i, cur_range_r)
            if (i == num_subtasks_right - 1):
                    assert cur_range_r is None
            else:
                assert (cur_range_r is None) or (len(cur_range_r) == 2)
            generator.add_param(
                key="task.task_spec.phase_{}.arm_right.subtask_{}.subtask_term_offset_range".format(phase_index+1,i+1),
                name="offset" if (verbose and (i == 0)) else "", 
                group=group,
                values=[cur_range_r],
            )

            name_mapping = {
                "selection_strategy": "selection_strategy",
                "selection_strategy_kwargs": "selection_strategy_kwargs",
                "action_noise": "noise",
                "num_interpolation_steps":"ni",
                "num_fixed_steps": "ni",
            }
            for kwargs in name_mapping.keys():
                generator.add_param(
                    key="task.task_spec.phase_{}.arm_right.subtask_{}.{}".format(phase_index+1,i+1, kwargs),
                    name=name_mapping[f"{kwargs}"] if (verbose and (i == 0)) else "", 
                    group=group,
                    values=[right_subtask_configs[f"subtask_{int(i+1)}"][kwargs]],
                )

    return generator


