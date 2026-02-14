# Ultra-fast Learning for Microrobot Navigation
Yinghan Sun, Aoji Zhu, Xiang Ji, Yamei Li, Jiachi Zhao, Yun Wang, Li Zhang, Huijun Gao, Lidong Yang.


![fig:framework](https://github.com/yinghansun/mr-nav/blob/main/assets/framework.png?raw=true)
In the proposed framework, we develop a fully vectorized simulator with more than 10,000 artificial vascular environments, parallelizing dynamics, LiDAR-inspired perception, and feasibility checks across over 8,000 environments to achieve roughly 100,000 transitions per second. 
To achieve effectiveness in the fast training, we propose a task–shaping–regularization (TSR) reward design. TSR can accelerate convergence, improve final performance, and enhance trajectory smoothness and obstacle safety.

## Update
- 2026-01-29: Clean the code and add additional instructions in the README.
- 2026-01-12: Release the code of the paper "Ultra-fast Learning for Microrobot Navigation".

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
    - Install other dependencies and mr-nav.
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
    $ python ./scripts/train.py --env vessel_env --num_run 1 --num_epoch 1000
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
2. Play with the trained policy:
    ~~~
    $ python ./scripts/play.py --env vessel_env --model_path ./saved_model/VesselEnv/model_499.pt
    ~~~

## Acknowledgements
Our implementation of RL algorithms leverages the code from [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We thank the authors for their open-source code.