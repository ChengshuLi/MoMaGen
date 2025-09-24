from setuptools import setup, find_packages

setup(
    name='momagen',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'cvxpy',
        'open3d',
        'wandb',
        'einops',
        'fpsample',
        'gym',
        'ninja',
    ],
    author='Chengshu Li',
    description='MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation',
    python_requires='>=3.10',
)

