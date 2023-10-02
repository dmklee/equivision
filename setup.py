from setuptools import setup

setup(
    name="equivision",
    version="0.1",
    description="SE(2) equivariant vision models with pretrained weights.",
    python_requires=">3.8.0",
    packages=["equivision"],
    install_requires=[
        "escnn==1.0.11",
        "torch>=2.0",
        "torchvision>=0.15",
    ],
)
