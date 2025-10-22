# Creating Custom Tasks

Learn how to define and generate demonstrations for your own custom mobile manipulation tasks.

## Step 1: Define a New BEHAVIOR Task with BDDL

First, review how BEHAVIOR tasks are defined here:

👉 [BEHAVIOR Task Definition](https://behavior.stanford.edu/behavior_components/behavior_tasks)

At the bottom of that page, you’ll find a section on [“Creating Custom Tasks”](https://behavior.stanford.edu/behavior_components/behavior_tasks#creating-custom-tasks).

You can define a custom task for MoMaGen in the same way you define a new BEHAVIOR task.

In this guide, we’ll walk through an example of creating a task called `datagen_picking_up_trash`, a simplified version of the existing `picking_up_trash` task. (You can read more about the original task [here](https://behavior.stanford.edu/challenge/tasks/00_turning_on_radio).)

As described in the “Creating Custom Tasks” section, we’ll follow these steps:

Create a task directory:

```bash
mkdir BEHAVIOR-1K/bddl/bddl/activity_definitions/datagen_picking_up_trash
```

Create a task definition file:

```bash
touch BEHAVIOR-1K/bddl/bddl/activity_definitions/datagen_picking_up_trash/problem0.bddl
```

Define the task:

```
(define (problem datagen_picking_up_trash-0)
    (:domain omnigibson)

    (:objects
        ashcan.n.01_1 - ashcan.n.01
        can__of__soda.n.01_1 - can__of__soda.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop ashcan.n.01_1 floor.n.01_1) 
        (ontop can__of__soda.n.01_1 table.n.02_1) 
        (inroom table.n.02_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (ontop agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?can__of__soda.n.01_1 ?ashcan.n.01_1)
        )
    )
)
```

To verify that the task is correctly defined, load it and enable online object sampling:

```python
import omnigibson as og
cfg = {
    # ... standard configuration ...
    "task": {
        "type": "BehaviorTask",
        "activity_name": "datagen_picking_up_trash",  # Your custom task
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "online_object_sampling": True,
    },
}

env = og.Environment(configs=cfg)
```

---

## Step 2: Sample a Task Instance

Open `BEHAVIOR-1K/OmniGibson/omnigibson/sampling/sample_b1k_tasks.py` and locate the `TASK_CUSTOM_LISTS` dictionary. Add an entry for your task, for example:

```
    ("datagen_picking_up_trash", "house_double_floor_lower"): {
        "whitelist": {
            "can__of__soda.n.01": {
                "can_of_soda": {
                    "itolcg": None,
                    "lugwcz": None,
                    "opivig": None
                }
            },
            "ashcan.n.01": {
                "trash_can": {
                    "wkxtxh": None
                }
            }
        },
        "blacklist": None,
    },
```

This dictionary specifies which objects to include in your task.

* **Whitelist** lets you specify exactly which categories and models should be used. For `can__of__soda.n.01`, we choose the `can_of_soda` category and whitelist three models (`itolcg`, `lugwcz`, `opivig`) to sample from. For `ashcan.n.01`, we choose the `trash_can` category and whitelist a single model (`wkxtxh`).
* **Blacklist** can be used to exclude specific categories or models.

To find valid category and model names, use the [Behavior Knowledge Base](https://behavior.stanford.edu/knowledgebase/index.html).

Once that’s set up, you can sample your task:

```bash
python -m omnigibson.sampling.sample_b1k_tasks --scene_model house_double_floor_lower --activities datagen_picking_up_trash
```

Whether sampling succeeds or fails, you’ll be dropped into a PDB breakpoint where you can inspect the sampled task instance.

If sampling succeeds, you can visualize it in the simulator:

```python
for _ in range(1000): og.sim.render()
```

This lets you move the camera and inspect object placements.

If sampling fails, adjust your whitelist/blacklist or the initial conditions in your BDDL task. Common failure cases include:

* No valid object instances found for one or more objects.
* No valid placement found, e.g., the object doesn’t fit inside or can’t stably rest on another object.

Once you’re satisfied with the sampled instance, press `c` in PDB to continue execution and save it to disk. By default, the instance will be saved to:

```
BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/house_double_floor_lower/json/house_double_floor_lower_task_datagen_picking_up_trash_0_0_template.json
```

✅ **Congrats — you’ve successfully defined and sampled a custom task!**

---

## Step 3: Collect Source Demonstrations

To collect source demonstrations, you can use one of the following input devices:

* **JoyLo** (used for MoMaGen source demos and the BEHAVIOR Challenge dataset)
* **SpaceMouse**

---

### Using JoyLo

You can learn more about JoyLo here:
👉 [JoyLo Overview](https://behavior.stanford.edu/behavior_components/joylo)

#### 1. Hardware Setup

Follow this guide to set up the hardware:
👉 [JoyLo Hardware Guide](https://behavior-robot-suite.github.io/docs/sections/joylo/overview.html)

#### 2. Software Setup

After setting up the hardware, continue with the software instructions on this page:
👉 [JoyLo Software Guide](https://behavior.stanford.edu/behavior_components/joylo#software-setup)

When you reach **Step 5** of the JoyLo guide, you can launch the recording process.

In one terminal, run:

```bash
python BEHAVIOR-1K/joylo/experiments/launch_nodes.py \
  --recording_path /PATH_TO_MOMAGEN/momagen/datasets/source_og/r1_picking_up_trash.hdf5 \
  --task_name datagen_picking_up_trash
```

In another terminal, run:

```bash
python joylo/experiments/run_joylo.py \
  --gello_model r1 \
  --joint_config_file joint_config_{YOUR_GELLO_SET_NAME}.yaml
```

#### 3. Saving the Recording

Once you finish collecting your source demonstration:

1. Focus your cursor on the GUI window.
2. Press `ESC`.
3. The recording will be saved automatically, and the program will exit.

---


## Step 4: Annotate Source Demonstrations 

In this step, you will **annotate the source demonstration** and **generate the configuration files** that contain the information required for MoMaGen data generation. 

Concretely, you need to **create a new json file** in `MoMaGen/momagen/datasets/base_configs`. You can start by **copying the contents of an existing JSON file** from the `base_configs` folder and then **modify it to suit your new task**. Note that **only the manipulation phases need to be annotated**.

#### 1. Set Basic Fields
Modify the `name` to your task name and set `filter_key` to `null` (unless you have multiple episodes in your HDF5 file and would like to specify which episode to use for data generation).

#### 2. Define Task Phases
A phase represents a semantic step in a multi-step, long-horizon task. For instance, a phase could be picking a cup, putting an apple in a bowl, pouring water in a mug, opening a drawer.

#### 3. Specify Phase Type
For each phase, specify its `type`. This can be either `uncoordinated` or `coordinated`.

#### 4. Define Subtasks for Each Phase
Next, define the **subtasks** for each phase and for each arm (left and right).

A subtask represents the portion of a phase that can be broken down into:
- A **free-space motion** part, and  
- A **contact-rich** part.

**Example:**
For the *open drawer* task:
- The free-space motion involves moving the gripper from its starting pose to near the handle.
- The contact-rich part involves grasping the handle and performing the motion to open the drawer.

In all existing MoMaGen tasks, each phase typically has **one subtask**.  
You may define multiple subtasks if desired.  
Each subtask must contain **at most one free-space** and **at most one contact-rich** segment.

5. For each subtask, mention the `object_ref` and the `attached_obj` for each arm. `object_ref` is the reference object for this arm and `attached_obj` is the object that the gripper is holding (if any). 
6. Create env_interface class (search `Add new class here for new tasks`) and task_config (search `Add new task configs here`) for the new custom task in `momagen/env_interfaces/omnigibson.py`
7. Now comes the heavylifting, for each subtask, we need to mention the `MP_end_step` and `subtask_term_step`. The simulation step from `MP_end_step` to the `subtask_end_step` is considered the contact-rich part of the subtask and will be "replayed". To obtain these values, you can replay the source demonstration using
```bash
python momagen/scripts/prepare_src_dataset.py --dataset momagen/datasets/source_og/{hdf5_name} --env_interface {env_interface e.g. MG_R1PickCup} --env_interface_type omnigibson_bimanual --replay_for_annotation 
```
and note down the simulation step where you would like the contact-rich part to begin (`MP_end_step`) and end (`subtask_term_step`).
    <video width="640" controls>
        <source src="/MoMaGen/assets/momagen_annotation_example.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
8. Now, we can generate the hdf5 with the `datagen_info` key which is used in the MoMaGen data generation pipeline.
```bash
python momagen/scripts/prepare_src_dataset.py --dataset momagen/datasets/source_og/{hdf5_name} --env_interface {env_interface e.g. MG_R1PickCup} --env_interface_type omnigibson_bimanual --generate_processed_hdf5
```
9. Add your base config json to `BASE_CONFIGS` and task name to `TASK_NAMES_MOMAGEN_ONLY` in `momagen/scripts/generate_configs.py` and run
```bash
python momagen/scripts/generate_configs.py
```
10. Create a new class for the new task in `momagen/configs/omnigibson.py` 

## Step 5: Run Data Genration

Specify the task and parameters, remember to set your own data path:

```bash
# Set the task name (choose from available tasks above)
TASK=picking_up_trash # your custom task name
DR=0 # can be {0, 1, 2}
NUM_DEMOS=1
WORKER_ID=0
FOLDER=/path/to/data # SPECIFY YOUR OWN PATH HERE
```

Then run the generation script:

```bash
python momagen/scripts/generate_dataset.py \
    --config momagen/datasets/configs/demo_src_r1_$TASK\_task_D$DR.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --folder $FOLDER/$TASK/r1_$TASK\_worker_$WORKER_ID \
    --seed $WORKER_ID
```


