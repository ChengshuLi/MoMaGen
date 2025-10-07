# Installation

!!! info "Prerequisites"
    Before starting, ensure you have:

    - Python 3.10+
    - Conda (recommended for environment management)

## Step 1: Clone Repository

Clone the repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/ChengshuLi/MoMaGen.git
cd MoMaGen
```

## Step 2: Create Conda Environment

Set up a new conda environment:

```bash
conda create -n momagen python=3.10
conda activate momagen
```

## Step 3: Install Dependencies

### Install MoMaGen

```bash
pip install -e .
```

### Install BEHAVIOR-1K and Dependencies

```bash
cd BEHAVIOR-1K
./setup.sh --omnigibson --bddl --joylo --dataset --primitives
cd ..
```

### Install Robomimic

```bash
cd robomimic
pip install -e .
cd ..
```

## Verification

!!! success "Verify Installation"
    To verify the installation, check that the required packages are available:

    ```bash
    python -c "import momagen; import omnigibson; import robomimic; print('Installation successful')"
    ```

    If successful, you should see: `Installation successful`

## Troubleshooting

!!! warning "Common Issues"

    **Submodules not cloned**

    ```bash
    git submodule update --init --recursive
    ```

## Next Steps

Once installation is complete, proceed to the [data generation](tutorials/generating-data.md) tutorial.
