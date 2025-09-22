"""
Task configs for robosuite.

See @Coffee_Config below for an explanation of each parameter.
"""
from momagen.configs.config import MG_Config

class TestPenBook_Config(MG_Config):
    """
    Corresponds to OG TestPenBook and variants.
    """
    NAME = "test_pen_book"
    TYPE = "omnigibson"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="eraser",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 10),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="book",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()


class TestCabinet(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "test_cabinet"
    TYPE = "omnigibson"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="cabinet",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="touch",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 10),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="cabinet",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()


class TestTiagoGiftbox(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "test_tiago_giftbox"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="gift_box",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="touch",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 10),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="gift_box",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

# TODO: need to delete
class TestTiagoCup(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "test_tiago_cup"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class TestR1Cup(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "test_r1_cup"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class TestTiagoSingleArmCup(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "test_tiago_single_arm_cup"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class R1PutAwayCup(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_put_away_cup"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class R1TidyTable(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_tidy_table"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class R1PickCup(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_pick_cup"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()


class R1DishesAway(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_dishes_away"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()


class R1CleanPan(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_clean_pan"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()

class R1BringingWater(MG_Config):
    """
    Corresponds to OG TestCabinet and variants.
    """
    NAME = "r1_bringing_water"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config,
        which has settings for each object-centric subtask in a task. Each
        dictionary should have kwargs for the @add_subtask method in the
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1_left = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="paper_cup",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp_let",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 6),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        # TODO: need to modify the following based on the predefined configs 
        self.task.task_spec.subtask_2 = dict(
            # Each subtask involves manipulation with respect to a single object frame.
            # This string should specify the object for this subtask. The name should be
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="notebook",
            # The "datagen_info" from the environment and dataset includes binary indicators
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(0, 1),
            # specifies how the source subtask segment should be selected during data generation
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.0,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="breakfast_table",
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()