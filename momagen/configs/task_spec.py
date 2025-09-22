"""
Defines task specification objects, which are used to store task-specific settings
for data generation.
"""
import json
import pdb

from momagen.datagen.selection_strategy import assert_selection_strategy_exists

class MG_TaskSpec:
    """
    Stores task-specific settings for data generation. Each task is a sequence of
    object-centric subtasks, and each subtask stores relevant settings used during
    the data generation process.
    """
    def __init__(self):
        self.spec = []

    def add_subtask(
        self, 
        object_ref,
        subtask_term_signal,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        arm='left',
    ):
        """
        Add subtask to this task spec.

        Args:
            object_ref (str): each subtask involves manipulation with 
                respect to a single object frame. This string should
                specify the object for this subtask. The name
                should be consistent with the "datagen_info" from the
                environment interface and dataset.

            subtask_term_signal (str or None): the "datagen_info" from the environment
                and dataset includes binary indicators for each subtask
                of the task at each timestep. This key should correspond
                to the key in "datagen_info" that should be used to
                infer when this subtask is finished (e.g. on a 0 to 1
                edge of the binary indicator). Should provide None for the final 
                subtask.

            subtask_term_offset_range (2-tuple): if provided, specifies time offsets to 
                be used during data generation when splitting a trajectory into 
                subtask segments. On each data generation attempt, an offset is sampled
                and added to the boundary defined by @subtask_term_signal.

            selection_strategy (str): specifies how the source subtask segment should be
                selected during data generation from the set of source human demos

            selection_strategy_kwargs (dict or None): optional keyword arguments for the selection
                strategy function used

            action_noise (float): amount of action noise to apply during this subtask

            num_interpolation_steps (int): number of interpolation steps to bridge previous subtask segment 
                to this one

            num_fixed_steps (int): number of additional steps (with constant target pose of beginning of 
                this subtask segment) to add to give the robot time to reach the pose needed to carry 
                out this subtask segment

            apply_noise_during_interpolation (bool): if True, apply action noise during interpolation phase 
                leading up to this subtask, as well as during the execution of this subtask

            arm (str): arm role for this subtask. Should be either 'left' or 'right'
        """
        if subtask_term_offset_range is None:
            # corresponds to no offset
            subtask_term_offset_range = (0, 0)
        assert isinstance(subtask_term_offset_range, tuple)
        assert len(subtask_term_offset_range) == 2
        assert subtask_term_offset_range[0] <= subtask_term_offset_range[1]
        assert_selection_strategy_exists(selection_strategy)
        self.spec.append(dict(
            object_ref=object_ref,
            subtask_term_signal=subtask_term_signal,
            subtask_term_offset_range=subtask_term_offset_range,
            selection_strategy=selection_strategy,
            selection_strategy_kwargs=selection_strategy_kwargs,
            action_noise=action_noise,
            num_interpolation_steps=num_interpolation_steps,
            num_fixed_steps=num_fixed_steps,
            apply_noise_during_interpolation=apply_noise_during_interpolation,
            arm=arm,
        ))

    @classmethod
    def from_json(cls, json_string=None, json_dict=None):
        """
        Instantiate a TaskSpec object from a json string. This should
        be consistent with the output of @serialize.

        Args:
            json_string (str): top-level of json has a key per subtask in-order (e.g.
                "subtask_1", "subtask_2", "subtask_3") and under each subtask, there should
                be an entry for each argument of @add_subtask

            json_dict (dict): optionally directly pass json dict
        """
        if json_dict is None:
            json_dict = json.loads(json_string)
        task_spec = cls()
        for subtask_name in json_dict:
            if json_dict[subtask_name]["subtask_term_offset_range"] is not None:
                json_dict[subtask_name]["subtask_term_offset_range"] = tuple(json_dict[subtask_name]["subtask_term_offset_range"])  
            task_spec.add_subtask(**json_dict[subtask_name])
        return task_spec

    @classmethod
    def from_json_bimanual_v2(cls, json_string=None, json_dict=None):
        """
        The bimanual cusomization of the from_json method

        TODO: now the config is not compatible with @serialize, since it is not under the index of subtask_1, subtask_2, ... 
        Instantiate a TaskSpec object from a json string. This should
        be consistent with the output of @serialize.

        Args:
            json_string (str): top-level of json is based on 'arm_left' and 'arm_right' keys
            Under each arm key, there should be a key per subtask in-order (e.g.
                "subtask_1", "subtask_2", "subtask_3") and under each subtask, there should
                be an entry for each argumesubtask_term_signalnt of @add_subtask

            json_dict (dict): optionally directly pass json dict
        """

        # TODO: need to add phase to the config
        # currently matching v2 config
        # config architecture
        # - arm_left
        #   - subtask_1
        #   - subtask_2
        #   - ...
        # - arm_right
        #   - subtask_1
        #   - subtask_2
        #   - ...

        if json_dict is None:
            json_dict = json.loads(json_string)

        left_json_dict = json_dict['arm_left']
        right_json_dict = json_dict['arm_right']
        task_spec = cls()
        for json_dict in [left_json_dict, right_json_dict]:
            task_spec.spec.append([])
            for subtask_name in json_dict:
                if json_dict[subtask_name]["subtask_term_offset_range"] is not None:
                    json_dict[subtask_name]["subtask_term_offset_range"] = tuple(json_dict[subtask_name]["subtask_term_offset_range"])  
                task_spec.add_bimanual_subtask(**json_dict[subtask_name])
        return task_spec

    @classmethod
    def from_json_bimanual(cls, json_string=None, json_dict=None):
        """
        The bimanual cusomization of the from_json method

        TODO: now the config is not compatible with @serialize, since it is not under the index of subtask_1, subtask_2, ... 
        Instantiate a TaskSpec object from a json string. This should
        be consistent with the output of @serialize.

        Args:
            json_string (str): top-level of json is based on 'arm_left' and 'arm_right' keys
            Under each arm key, there should be a key per subtask in-order (e.g.
                "subtask_1", "subtask_2", "subtask_3") and under each subtask, there should
                be an entry for each argumesubtask_term_signalnt of @add_subtask

            json_dict (dict): optionally directly pass json dict
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

        # TODO: this class does not seem necessary 
        if json_dict is None:
            json_dict = json.loads(json_string)
        
        task_spec = cls()
        num_phases = len(json_dict)
        for phase_index in range(num_phases):
            phase_json_dict = json_dict['phase_{}'.format(phase_index+1)]
            task_spec.spec.append([]) # for each phase
            
            for json_dict_arm in [phase_json_dict['arm_left'], phase_json_dict['arm_right']]:
                task_spec.spec[-1].append([])
                for subtask_name in json_dict_arm:
                    if json_dict_arm[subtask_name]["subtask_term_offset_range"] is not None:
                        json_dict_arm[subtask_name]["subtask_term_offset_range"] = tuple(json_dict_arm[subtask_name]["subtask_term_offset_range"])  
                    task_spec.add_bimanual_subtask(phase_type=phase_json_dict["type"], **json_dict_arm[subtask_name])

        return task_spec
    
    def add_bimanual_subtask(self,
        phase_type,
        object_ref,
        subtask_term_signal,
        subtask_term_step=None,
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        arm='left',
        MP_end_step=None,
        attached_obj=None,
        retract_type=None,
        symmetric_object=False,
    ):
        """
        Add subtask to this task spec.

        Args:
            object_ref (str): each subtask involves manipulation with 
                respect to a single object frame. This string should
                specify the object for this subtask. The name
                should be consistent with the "datagen_info" from the
                environment interface and dataset.

            subtask_term_signal (str or None): the "datagen_info" from the environment
                and dataset includes binary indicators for each subtask
                of the task at each timestep. This key should correspond
                to the key in "datagen_info" that should be used to
                infer when this subtask is finished (e.g. on a 0 to 1
                edge of the binary indicator). Should provide None for the final 
                subtask.

            subtask_term_step (int or None): the termination step for the current subtask if it is not None.
            If it is None, the termination step is the last step of the episode.

            subtask_term_offset_range (2-tuple): if provided, specifies time offsets to 
                be used during data generation when splitting a trajectory into 
                subtask segments. On each data generation attempt, an offset is sampled
                and added to the boundary defined by @subtask_term_signal.

            selection_strategy (str): specifies how the source subtask segment should be
                selected during data generation from the set of source human demos

            selection_strategy_kwargs (dict or None): optional keyword arguments for the selection
                strategy function used

            action_noise (float): amount of action noise to apply during this subtask

            num_interpolation_steps (int): number of interpolation steps to bridge previous subtask segment 
                to this one

            num_fixed_steps (int): number of additional steps (with constant target pose of beginning of 
                this subtask segment) to add to give the robot time to reach the pose needed to carry 
                out this subtask segment

            apply_noise_during_interpolation (bool): if True, apply action noise during interpolation phase 
                leading up to this subtask, as well as during the execution of this subtask

            arm (str): arm role for this subtask. Should be either 'left' or 'right'

            MP_end_step (int or None): the end step of the motion planner. If it is None, the end step is the last step of the episode.
        """
        if subtask_term_offset_range is None:
            # corresponds to no offset
            subtask_term_offset_range = (0, 0)
        assert isinstance(subtask_term_offset_range, tuple)
        assert len(subtask_term_offset_range) == 2
        assert subtask_term_offset_range[0] <= subtask_term_offset_range[1]
        assert_selection_strategy_exists(selection_strategy)
        # TODO: now it is only compatible when phase exist; if phase not exist, change to self.spec[-1].append()
        self.spec[-1][-1].append(dict(
            phase_type=phase_type,
            object_ref=object_ref,
            subtask_term_signal=subtask_term_signal,
            subtask_term_step=subtask_term_step,
            subtask_term_offset_range=subtask_term_offset_range,
            selection_strategy=selection_strategy,
            selection_strategy_kwargs=selection_strategy_kwargs,
            action_noise=action_noise,
            num_interpolation_steps=num_interpolation_steps,
            num_fixed_steps=num_fixed_steps,
            apply_noise_during_interpolation=apply_noise_during_interpolation,
            arm=arm,
            MP_end_step=MP_end_step,
            attached_obj=attached_obj,
            retract_type=retract_type,
            symmetric_object=symmetric_object,
        ))
    
    def serialize(self):
        """
        Return a json string corresponding to this task spec object. Compatible with
        @from_json classmethod.
        """
        # TODO: the serialize may not work for bimanual configs
        import pdb; pdb.set_trace()
        json_dict = dict()
        for i, elem in enumerate(self.spec):
            json_dict["subtask_{}".format(i + 1)] = elem
        return json.dumps(json_dict, indent=4)

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, ind):
        """Support list-like indexing"""
        return self.spec[ind]

    def __iter__(self):
        """Support list-like iteration."""
        return iter(self.spec)

    def __repr__(self):
        return json.dumps(self.spec, indent=4)
