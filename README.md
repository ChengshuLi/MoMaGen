# MoMaGen

## Installation

### Set up conda environment
```
conda create -n momagen python=3.10
conda activate momagen
```

### Install all required repositories and dependencies (at $PROJECT_DIR)
```
PROJECT_DIR=/path/to/momagen
mkdir $PROJECT_DIR

cd $PROJECT_DIR && git clone git@github.com:ChengshuLi/MoMaGen.git && cd MoMaGen && git checkout && pip install -e .

cd $PROJECT_DIR && git clone git@github.com:ChengshuLi/mimicgen.git && cd mimicgen && git checkout b1k-mimicgen && pip install -e .

cd $PROJECT_DIR && git clone git@github.com:ChengshuLi/robomimic.git && cd robomimic && git checkout b1k-mimicgen && pip install -e .

cd $PROJECT_DIR && git clone git@github.com:StanfordVL/bddl.git && cd bddl && git checkout b1k-mimicgen && pip install -e .

cd $PROJECT_DIR && git clone git@github.com:StanfordVL/curobo.git && cd curobo && pip install -e . --no-build-isolation

cd $PROJECT_DIR && git clone git@github.com:StanfordVL/OmniGibson.git && cd OmniGibson && git checkout b1k-mimicgen && pip install -e .
python -m omnigibson.install
```

## Data Generation

### Generate configs
```
cd $PROJECT_DIR
python mimicgen/scripts/generate_core_configs_og.py
```

### Copy scene instances
```
cd $PROJECT_DIR
cp MoMaGen/momagen/scene_instances/Rs_int/* OmniGibson/omnigibson/data/og_dataset/scenes/Rs_int/json
cp MoMaGen/momagen/scene_instances/house_single_floor/* OmniGibson/omnigibson/data/og_dataset/scenes/house_single_floor/json
```

### Generate data
```
cd $PROJECT_DIR/mimicgen
TASK=pick_cup
DR=0
NUM_DEMOS=10
WORKER_ID=0
FOLDER=/path/to/data
python mimicgen/scripts/generate_dataset.py \
    --config datasets/generated_data_mimicgen_format/core_configs_og/demo_src_r1_$TASK\_task_D$DR.json \
    --num_demos $NUM_DEMOS \
    --bimanual \
    --folder $FOLDER/$TASK/r1_$TASK\_worker_$WORKER_ID \
    --seed $WORKER_ID
```