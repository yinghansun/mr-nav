# Ultra-fast Learning for Microrobot Navigation
Yinghan Sun, Aoji Zhu, Xiang Ji, Yamei Li, Jiachi Zhao, Yun Wang, Li Zhang, Huijun Gao, Lidong Yang.


![fig:framework](https://github.com/yinghansun/mr-nav/blob/main/assets/framework.png?raw=true)
In the proposed framework, we develop a fully vectorized simulator with more than 10,000 artificial vascular environments, parallelizing dynamics, LiDAR-inspired perception, and feasibility checks across over 8,000 environments to achieve roughly 100,000 transitions per second. 
To achieve effectiveness in the fast training, we propose a task–shaping–regularization (TSR) reward design. TSR can accelerate convergence, improve final performance, and enhance trajectory smoothness and obstacle safety.

## Update
- 2026-04-28：Add more instructions and in-line comments.
- 2026-01-29: Clean the code and add additional instructions in the README.
- 2026-01-12: Release the code of the paper "Ultra-fast Learning for Microrobot Navigation".

## Repository Structure
```
.
├── alg               # RL algorithms
├── cfg               # Configuration files
├── dataset           # Dataset for establish the simulation
├── log               # Log files generated during training
├── saved_model       # Saved models for evaluation and visualization
├── scripts           # Training and evaluation scripts
├── utils             # Utility functions for this project
└── ...
```

## Requirement
To achieve a high parallelism and fast training, we recommand user to train the policy on a GPU with at least 8 GB memory. However, users who only want to play with the trained policy can run the code on a CPU.

This repository has been tested on the following environments:
~~~
python==3.12.8
numpy==2.4.2
cv2==4.12.0
matplotlib==3.10.8
tensorboard==2.20.0
pyyaml==6.0.3
tqdm==4.67.3
~~~

## Installation
1. Clone this repository: `git clone https://github.com/yinghansun/mr-nav.git`
2. Create a virtual environment. Below is an example using `virtualenv`.
    ~~~
    $ pip install virtualenv
    $ cd $path-to-mr-nav$
    ~~~
    For windows users:
    ~~~
    $ virtualenv --python $path-to-python.exe$ mr-nav-env
    $ $path-to-mr-nav-env$\Scripts\activate
    ~~~
    For Linux users:
    ~~~
    $ virtualenv --python $path-to-python$ mr-nav-env
    $ source mr-nav-env/bin/activate
    ~~~
3. Install dependencies.
    - Install `PyTorch` based on your platform. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) for more details.
    - Install other dependencies and mr-nav (This takes 1 - 5 mins to install).
        ~~~
        $ pip install -e .
        ~~~
4. Prepare the dataset. Users can either play with the dataset used in the paper or prepare their own dataset.
     - Download the expert dataset from [here](https://figshare.com/articles/dataset/Autonomous_environment-adaptive_microrobot_swarm_navigation_enabled_by_deep_learning-based_real-time_distribution_planning_dataset_/19149779/1?file=34023026) and put it in the `dataset` folder.
     - Process the dataset.
        ~~~
        $ python scripts/process_dataset.py
        ~~~

## Usage
1. Train a policy:
    ~~~
    $ python ./scripts/train.py --env vessel_env --num_run 1 --num_epoch 500
    ~~~
    Note:
    - The trained policy will be saved in `./log/<env_type>/<date>_<batch_size>_<num_envs>_<idx>/`, where 
        - `<date>` is the date when the training is started.
        - `<batch_size>` is the batch size used in training.
        - `<num_envs>` is the number of parallel environments used in training.
        - `<idx>` is the index of the run with the same `<date>`, `<batch_size>` and `<num_envs>`.
    - Users can monitor the training process using `TensorBoard`.
        ~~~
        $ tensorboard --logdir ./log/<env_type>/
        ~~~
    - During training, we do not render the environment by default to speed up the training process. Users can set `--vis True` to visualize the environment during training if needed.
    - Users can set `--render_idx` to visualize the environment with the specified environment index during training. If not set, the environment with index 0 will be visualized.
    - The overall training process takes about 8 - 10 minutes on a computer with a NVIDIA RTX 3060 GPU.
2. Play with the trained policy:
    - Freespace environment:
        ~~~
        $ python ./scripts/play.py --env freespace_env --model_path ./saved_model/FreeSpace/model_499.pt
        ~~~
    - Vascular environment:
        ~~~
        $ python ./scripts/play.py --env vessel_env --model_path ./saved_model/VesselEnv/model_499.pt
        ~~~

## Acknowledgements
Our implementation of RL algorithms leverages the code from [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We thank the authors for their open-source code.