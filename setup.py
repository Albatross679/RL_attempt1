from setuptools import setup, find_packages

setup(
    name="rl_attempt1",
    version="0.1.0",
    description="RL Development Environment with DisMech and ALF",
    author="",
    python_requires=">=3.10,<3.13",
    packages=find_packages(exclude=["tests", "experiments"]),
    install_requires=[
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "alf",
        "dismech-python",
        "pyvista",
        "pyelastica==0.2.4",
        "tensorboard",
        "huggingface-hub",
    ],
)
