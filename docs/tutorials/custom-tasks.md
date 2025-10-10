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
python BEHAVIOR-1K/OmniGibson/omnigibson/sampling/sample_b1k_tasks.py --scene_model house_double_floor_lower --activities datagen_picking_up_trash
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

## Step 4: Annotate Source Demonstrations

## Step 5: Define MoMaGen Task Configuration

