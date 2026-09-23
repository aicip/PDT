from setuptools import find_packages, setup

# Runtime dependencies of the code in this snapshot. The DMD implementation used by
# PDT is contained in kcl/lib/dmd; no external DMD package is required.
requirements = [
    "torch",
    "torchvision",
    "numpy",
    "tqdm",
    "wandb",
    "tensorboardX",
    "torch_optimizer",
    "datasets",
    "scikit-learn",
    "pillow",
]

setup(
    name="kcl",
    version="1.0",
    install_requires=requirements,
    packages=find_packages(exclude=["scripts", "historical"]),
)
