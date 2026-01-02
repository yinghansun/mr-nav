# Ultra-fast Learning for Microrobot Navigation

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
     - Download the expert dataset from [here](https://mycuhk-my.sharepoint.com/personal/1155135830_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155135830%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fdataset&ga=1) and put it in the `dataset` folder.
     - Process the dataset.
        ~~~
        $ python scripts/process_dataset.py
        ~~~