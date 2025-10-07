# Data Generation

This guide walks through the process of generating demonstration data for MoMaGen tasks.

## Overview

The data generation pipeline consists of three main steps:

1. Generate task configurations
2. Copy scene instances
3. Generate demonstration datasets

## Step 1: Generate Configurations

Generate configuration files for all tasks:

```bash
python momagen/scripts/generate_configs.py
```

This creates JSON configuration files in `momagen/datasets/configs/` for each task.

## Step 2: Copy Scene Instances

Set up the scene instances for the BEHAVIOR-1K dataset:

```bash
# Create required directories
mkdir -p BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/Rs_int/json
mkdir -p BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/house_single_floor/json

# Copy scene instances
cp momagen/scene_instances/Rs_int/* BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/Rs_int/json
cp momagen/scene_instances/house_single_floor/* BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/house_single_floor/json
```

## Step 3: Generate Demonstrations

### Available Tasks

- `pick_cup` - Pick and place a cup
- `tidy_table` - Organize objects on a table
- `dishes_away` - Put dishes into storage
- `clean_pan` - Clean and store a pan

### Basic Usage

Generate demonstrations for a specific task:

```bash
TASK=pick_cup
DR=0
NUM_DEMOS=10
WORKER_ID=0
FOLDER=/path/to/data  # Specify your desired output path

python momagen/scripts/generate_dataset.py \
    --config momagen/datasets/configs/demo_src_r1_${TASK}_task_D${DR}.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --folder $FOLDER/$TASK/r1_${TASK}_worker_$WORKER_ID \
    --seed $WORKER_ID
```

### Parameters

- `TASK`: Task name (see available tasks above)
- `DR`: Domain randomization level (default: 0)
- `NUM_DEMOS`: Number of demonstrations to generate
- `WORKER_ID`: Worker ID for parallel generation (used as random seed)
- `FOLDER`: Output directory for generated data

### Example: Generate Multiple Tasks

```bash
# Set common parameters
DR=0
NUM_DEMOS=10
WORKER_ID=0
FOLDER=./data

# Generate data for all tasks
for TASK in pick_cup tidy_table dishes_away clean_pan; do
    python momagen/scripts/generate_dataset.py \
        --config momagen/datasets/configs/demo_src_r1_${TASK}_task_D${DR}.json \
        --num_demos $NUM_DEMOS \
        --bimanual \
        --folder $FOLDER/$TASK/r1_${TASK}_worker_$WORKER_ID \
        --seed $WORKER_ID
done
```

## Output Format

Generated demonstrations are saved in HDF5 format with the following structure:

- **Observations**: RGB images, depth, robot state
- **Actions**: Joint positions, gripper actions
- **Rewards**: Task completion signals
- **Metadata**: Scene information, task parameters

## Parallel Generation

For faster data generation, run multiple workers in parallel:

```bash
# Launch 4 workers in parallel
for WORKER_ID in 0 1 2 3; do
    python momagen/scripts/generate_dataset.py \
        --config momagen/datasets/configs/demo_src_r1_pick_cup_task_D0.json \
        --num_demos 25 \
        --bimanual \
        --folder ./data/pick_cup/r1_pick_cup_worker_$WORKER_ID \
        --seed $WORKER_ID &
done

# Wait for all workers to complete
wait
```

## Additional Scripts

### Prepare Source Dataset

To prepare source datasets from existing demonstrations:

```bash
python momagen/scripts/prepare_src_dataset.py \
    --input_folder /path/to/raw/data \
    --output_folder /path/to/processed/data
```

## Troubleshooting

**Issue**: Configuration files not found
- Make sure you ran `generate_configs.py` first

**Issue**: Scene instances missing
- Verify scene instances were copied to the correct BEHAVIOR-1K directories

**Issue**: Generation fails or hangs
- Check GPU/CUDA availability
- Reduce `NUM_DEMOS` for testing
- Check logs for specific error messages
