from setuptools import find_packages, setup

setup(
    name="mr-nav",
    version="1.0.0",
    author="Yinghan Sun",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="yinghansun2@gmail.com",
    description="",
    install_requires=[
        "numpy", 
        "opencv-python", 
        "matplotlib", 
        "tensorboard",
        "pyyaml",
        "tqdm",
    ],
)