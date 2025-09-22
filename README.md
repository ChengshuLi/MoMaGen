<h1 align="center">MomaGen</h1>
<h3 align="center">Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation</h3>

![momagen_teaser](https://github.com/user-attachments/assets/278f7c1c-1d73-47c0-942c-e70115875eb4)


## 🛠️ Installation

### Clone the repository with submodules
```
git clone --recurse-submodules https://github.com/ChengshuLi/MoMaGen.git
```

### Set up conda environment
```
conda create -n momagen python=3.10
conda activate momagen
```

### Install dependencies
```
cd MoMaGen && pip install -e .
cd BEHAVIOR-1K && . ./setup.sh --omnigibson --bddl --teleop --dataset --primitives && cd ..
cd robomimic && pip install -e . && cd ..
```

## 📊 Data Generation

### Generate configs
```
python momagen/scripts/generate_configs.py
```

### Copy scene instances
```
cp momagen/scene_instances/Rs_int/* BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/scenes/Rs_int/json
cp momagen/scene_instances/house_single_floor/* BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/scenes/house_single_floor/json
```

### Generate data
```
TASK=pick_cup
DR=0
NUM_DEMOS=10
WORKER_ID=0
FOLDER=/path/to/data
python momagen/scripts/generate_dataset.py \
    --config momagen/datasets/configs/demo_src_r1_$TASK\_task_D$DR.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --folder $FOLDER/$TASK/r1_$TASK\_worker_$WORKER_ID \
    --seed $WORKER_ID
```
