# Generating Data

This tutorial will walk you through generating demonstration data using MoMaGen.

## Setup

Before starting, ensure you have completed the [Installation](../installation.md) guide.

Then, verify your installation is working correctly:

```bash
# Activate your environment
conda activate momagen

# Test imports
python -c "import momagen; import omnigibson; import robomimic; print('✓ All imports successful')"
```

Next, copy the MoMaGen-specific task instances into the BEHAVIOR-1K dataset:

```bash
# Create directories
mkdir -p BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/Rs_int/json

# Copy scene instances
cp momagen/scene_instances/Rs_int/* BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/Rs_int/json/
cp momagen/scene_instances/house_single_floor/* BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/house_single_floor/json
```

## Generating Demonstrations

Let's generate a simple `pick_cup` demonstration:

### 1. Generate Configuration

```bash
python momagen/scripts/generate_configs.py
```

This creates configuration files for all available tasks in `momagen/datasets/configs/`.

### 2. Generate a Single Demo

Specify the task and parameters, remember to set your own data path:

```bash
# Set the task name (choose from available tasks above)
TASK=pick_cup  # Options: pick_cup, tidy_table, dishes_away, clean_pan
DR=0
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

This will:

- Load the pick_cup task configuration
- Initialize the simulation environment
- Generate ten demonstration trajectory
- Save the result to the specified folder
