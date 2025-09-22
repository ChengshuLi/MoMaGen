"""
We utilize robomimic's config generator class to easily generate data generation configs for our
core set of tasks in the paper. It can be modified easily to generate other configs.

The global variables at the top of the file should be configured manually.

See https://robomimic.github.io/docs/tutorials/hyperparam_scan.html for more info.
"""
import os
import json
import shutil

import robomimic
from robomimic.utils.hyperparam_utils import ConfigGenerator

import momagen
import momagen.utils.config_utils as ConfigUtils
from momagen.utils.file_utils import config_generator_to_script_lines


# set path to folder containing src datasets
SRC_DATA_DIR = os.path.join(momagen.__path__[0], "./datasets/source_og")

# set base folder for where to copy each base config and generate new config files for data generation
# CONFIG_DIR = "/tmp/core_configs_og"
CONFIG_DIR = os.path.join(momagen.__path__[0], "./datasets/configs") 

# set base folder for newly generated datasets
# OUTPUT_FOLDER = "/tmp/core_datasets_og"
OUTPUT_FOLDER = os.path.join(momagen.__path__[0], "./datasets/generated_datasets")

# number of trajectories to generate (or attempt to generate)
NUM_TRAJ = 2

# whether to guarantee that many successful trajectories (e.g. keep running until that many successes, or stop at that many attempts)
GUARANTEE = True

# whether to run a quick debug run instead of full generation
DEBUG = False

# camera settings for collecting observations
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
CAMERA_SIZE = (84, 84)

BASE_BASE_CONFIG_PATH = os.path.join(momagen.__path__[0], "./datasets/base_configs")
BASE_CONFIGS = [
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_pick_cup.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_pick_cup_mimicgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_pick_cup_skillgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_tidy_table.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_tidy_table_mimicgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_tidy_table_skillgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_dishes_away.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_dishes_away_mimicgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_dishes_away_skillgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_clean_pan.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_clean_pan_mimicgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_clean_pan_skillgen.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "r1_bringing_water.json"),
]

def make_generators(base_configs):
    """
    An easy way to make multiple config generators by using different
    settings for each.
    """
    all_settings = [
        # MoMaGen Pick Cup
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_pick_cup.hdf5"),
            dataset_name="r1_pick_cup",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_pick_cup".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_pick_cup_D0", "r1_pick_cup_D1", "r1_pick_cup_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MimicGen Pick Cup
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_pick_cup.hdf5"),
            dataset_name="r1_pick_cup_mimicgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_pick_cup_mimicgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_pick_cup_D0", "r1_pick_cup_D1", "r1_pick_cup_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # SkillGen Pick Cup
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_pick_cup.hdf5"),
            dataset_name="r1_pick_cup_skillgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_pick_cup_skillgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_pick_cup_D0", "r1_pick_cup_D1", "r1_pick_cup_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MoMaGen Tidy Table
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_tidy_table.hdf5"),
            dataset_name="r1_tidy_table",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_tidy_table".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_tidy_table_D0", "r1_tidy_table_D1", "r1_tidy_table_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MimicGen Tidy Table
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_tidy_table.hdf5"),
            dataset_name="r1_tidy_table_mimicgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_tidy_table_mimicgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_tidy_table_D0", "r1_tidy_table_D1", "r1_tidy_table_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # SkillGen Tidy Table
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_tidy_table.hdf5"),
            dataset_name="r1_tidy_table_skillgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_tidy_table_skillgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_tidy_table_D0", "r1_tidy_table_D1", "r1_tidy_table_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MoMaGen Dishes Away
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_dishes_away.hdf5"),
            dataset_name="r1_dishes_away",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_dishes_away".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_dishes_away_D0", "r1_dishes_away_D1", "r1_dishes_away_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MimicGen Dishes Away
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_dishes_away.hdf5"),
            dataset_name="r1_dishes_away_mimicgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_dishes_away_mimicgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_dishes_away_D0", "r1_dishes_away_D1", "r1_dishes_away_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # SkillGen Dishes Away
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_dishes_away.hdf5"),
            dataset_name="r1_dishes_away_skillgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_dishes_away_skillgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_dishes_away_D0", "r1_dishes_away_D1", "r1_dishes_away_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MoMaGen Clean Pan
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_clean_pan.hdf5"),
            dataset_name="r1_clean_pan",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_clean_pan".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_clean_pan_D0", "r1_clean_pan_D1", "r1_clean_pan_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MimicGen Clean Pan
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_clean_pan.hdf5"),
            dataset_name="r1_clean_pan_mimicgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_clean_pan_mimicgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_clean_pan_D0", "r1_clean_pan_D1", "r1_clean_pan_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # SkillGen Clean Pan
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_clean_pan.hdf5"),
            dataset_name="r1_clean_pan_skillgen",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_clean_pan_skillgen".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_clean_pan_D0", "r1_clean_pan_D1", "r1_clean_pan_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
        # MoMaGen Bringing Water
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "r1_bringing_water.hdf5"),
            dataset_name="r1_bringing_water",   # this will dictate the name of the config file in core_configs_og
            generation_path="{}/r1_bringing_water".format(OUTPUT_FOLDER), # this is where the MimicGen generated data will be stored inside {path}/core_datasets_og
            tasks=["r1_bringing_water_D0", "r1_bringing_water_D1", "r1_bringing_water_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
    ]

    assert len(base_configs) == len(all_settings)
    ret = []
    for conf, setting in zip(base_configs, all_settings):
        ret.append(make_generator(os.path.expanduser(conf), setting))
    return ret


def make_generator(config_file, settings):
    """
    Implement this function to setup your own hyperparameter scan.
    Each config generator is created using a base config file (@config_file)
    and a @settings dictionary that can be used to modify which parameters
    are set.
    """
    generator = ConfigGenerator(
        base_config_file=config_file,
        script_file="", # will be overriden in next step
    )

    # set basic settings
    ConfigUtils.set_basic_settings(
        generator=generator,
        group=0,
        source_dataset_path=settings["dataset_path"],
        source_dataset_name=settings["dataset_name"],
        generation_path=settings["generation_path"],
        guarantee=GUARANTEE,
        num_traj=NUM_TRAJ,
        num_src_demos=10,
        max_num_failures=None,
        num_demo_to_render=10,
        num_fail_demo_to_render=25,
        render_video=False,
        verbose=False,
    )

    # set settings for subtasks
    bimanual=True
    if bimanual:
        # now all the configs are from the configuraiton file 
        ConfigUtils.set_subtask_settings_bimanual(
            generator=generator,
            group=0,
            base_config_file=config_file,
            select_src_per_subtask=settings["select_src_per_subtask"],
            verbose=False,
        )
    else:
        ConfigUtils.set_subtask_settings(
            generator=generator,
            group=0,
            base_config_file=config_file,
            select_src_per_subtask=settings["select_src_per_subtask"],
            subtask_term_offset_range=settings["subtask_term_offset_range"],
            selection_strategy=settings.get("selection_strategy", None),
            selection_strategy_kwargs=settings.get("selection_strategy_kwargs", None),
            # default settings: action noise 0.05, with 5 interpolation steps
            # Disable any action noise for now
            # action_noise=0.05,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            verbose=False,
        )

    # optionally set env interface to use, and type
    # generator.add_param(
    #     key="experiment.task.interface",
    #     name="",
    #     group=0,
    #     values=[settings["task_interface"]],
    # )
    # generator.add_param(
    #     key="experiment.task.interface_type",
    #     name="",
    #     group=0,
    #     values=["robosuite"],
    # )

    # set task to generate data on
    generator.add_param(
        key="experiment.task.name",
        name="task",
        group=1,
        values=settings["tasks"],
        value_names=settings["task_names"],
    )

    # optionally set robot and gripper that will be used for data generation (robosuite-only)
    if settings.get("robots", None) is not None:
        generator.add_param(
            key="experiment.task.robot",
            name="r",
            group=2,
            values=settings["robots"],
        )
    if settings.get("grippers", None) is not None:
        generator.add_param(
            key="experiment.task.gripper",
            name="g",
            group=2,
            values=settings["grippers"],
        )

    # set observation collection settings
    ConfigUtils.set_obs_settings(
        generator=generator,
        group=-1,
        collect_obs=True,
        camera_names=CAMERA_NAMES,
        camera_height=CAMERA_SIZE[0],
        camera_width=CAMERA_SIZE[1],
    )

    if DEBUG:
        # set debug settings
        ConfigUtils.set_debug_settings(
            generator=generator,
            group=-1,
        )

    # seed
    generator.add_param(
        key="experiment.seed",
        name="",
        group=1000000,
        values=[1],
    )

    return generator


def main():

    # make config generators
    generators = make_generators(base_configs=BASE_CONFIGS)

    # # maybe remove existing config directory
    config_dir = CONFIG_DIR
    # if os.path.exists(config_dir):
    #     ans = input("Non-empty dir at {} will be removed.\nContinue (y / n)? \n".format(config_dir))
    #     if ans != "y":
    #         exit()
    #     shutil.rmtree(config_dir)

    all_json_files, run_lines = config_generator_to_script_lines(generators, config_dir=config_dir)

    real_run_lines = []
    for line in run_lines:
        line = line.strip().replace("train.py", "generate_dataset.py")
        line += " --auto-remove-exp"
        real_run_lines.append(line)
    run_lines = real_run_lines

    print("configs")
    print(json.dumps(all_json_files, indent=4))
    print("runs")
    print(json.dumps(run_lines, indent=4))


if __name__ == "__main__":
    main()
