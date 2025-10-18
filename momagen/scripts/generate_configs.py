"""
We utilize robomimic's config generator class to easily generate data generation configs for our
core set of tasks in the paper. It can be modified easily to generate other configs.

The global variables at the top of the file should be configured manually.

See https://robomimic.github.io/docs/tutorials/hyperparam_scan.html for more info.
"""
import os
import json

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

# task names that support baseline variations (momagen, mimicgen, skillgen)
TASK_NAMES_WITH_BASELINES = ["pick_cup", "tidy_table", "dishes_away", "clean_pan"]

# task names that only have momagen configuration
TASK_NAMES_MOMAGEN_ONLY = ["bringing_water"]

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
    # Add new base configs here
    
]

def make_generators(base_configs):
    """
    Create multiple config generators for different tasks.

    Args:
        base_configs (list): List of base config file paths

    Returns:
        list: List of ConfigGenerator instances
    """
    # Common task configuration template
    def create_task_config(task_name, baseline_suffix=""):
        full_name = f"r1_{task_name}{baseline_suffix}"
        return dict(
            dataset_path=os.path.join(SRC_DATA_DIR, f"r1_{task_name}.hdf5"),
            dataset_name=full_name,
            generation_path=f"{OUTPUT_FOLDER}/{full_name}",
            tasks=[f"{full_name}_D0", f"{full_name}_D1", f"{full_name}_D2"],
            task_names=["D0", "D1", "D2"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        )

    all_settings = []

    # Add configurations for tasks with baseline variations
    for task_name in TASK_NAMES_WITH_BASELINES:
        # MoMaGen version
        all_settings.append(create_task_config(task_name))
        # MimicGen baseline
        all_settings.append(create_task_config(task_name, "_mimicgen"))
        # SkillGen baseline
        all_settings.append(create_task_config(task_name, "_skillgen"))

    # Add configurations for tasks with only self version
    for task_name in TASK_NAMES_MOMAGEN_ONLY:
        # MoMaGen version only
        all_settings.append(create_task_config(task_name))

    assert len(base_configs) == len(all_settings)
    ret = []
    for conf, setting in zip(base_configs, all_settings):
        ret.append(make_generator(os.path.expanduser(conf), setting))
    return ret


def make_generator(config_file, settings):
    """
    Create a config generator for hyperparameter scanning.

    Args:
        config_file (str): Path to base config file
        settings (dict): Dictionary of settings for this generator

    Returns:
        ConfigGenerator: Configured generator instance
    """
    generator = ConfigGenerator(
        base_config_file=config_file,
        script_file="",  # will be overridden in next step
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
    bimanual = True
    if bimanual:
        # Use configuration from the config file
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
            action_noise=0.0,  # Disabled for now
            num_interpolation_steps=5,
            num_fixed_steps=0,
            verbose=False,
        )


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
    """Generate configuration files and run scripts for MoMaGen tasks."""

    # Create config generators
    generators = make_generators(base_configs=BASE_CONFIGS)

    # Generate config files and run lines
    all_json_files, run_lines = config_generator_to_script_lines(
        generators, config_dir=CONFIG_DIR
    )

    # Customize run lines for data generation
    modified_run_lines = []
    for line in run_lines:
        line = line.strip().replace("train.py", "generate_dataset.py")
        line += " --auto-remove-exp"
        modified_run_lines.append(line)

    # Output results
    print("Generated configs:")
    print(json.dumps(all_json_files, indent=4))
    print("\nGenerated run commands:")
    print(json.dumps(modified_run_lines, indent=4))


if __name__ == "__main__":
    main()
