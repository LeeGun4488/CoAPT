# CoAPT
CoAPT

----

# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

Setup conda environment (recommended).

**Create a conda environment**

```
conda create -y -n dept python=3.8
conda activate dept
```

**Install torch (requires version >= 1.8.1) and torchvision**

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**Install dassl**

```
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
```

**Install DePT**

```
cd ..

git clone https://github.com/somebody/DePT.git
cd DePT/

pip install -r requirements.txt
pip install setuptools==59.5.0
```

----

# Data preparation

Please follow the instructions at [DATASETS.md](datasets/DATASETS.md) to prepare all datasets.

----

# Training and Evaluation

We provide parallel running script `parallel_runner.py` for each prompting variant including CoOp (w/ DePT), CoCoOp (w/ DePT), KgCoOp (w/ DePT), MaPLe (w/ DePT). Make sure to configure the dataset paths in environment variable DATA and run the commands from the main directory.

**Base to New Generalization**

```
# Running CoOp (w/ DePT)
python parallel_runner.py --cfg coop
python parallel_runner.py --cfg coop_dept

# Running CoCoOp (w/ DePT)
python parallel_runner.py --cfg cocoop
python parallel_runner.py --cfg cocoop_dept

# Running KgCoOp (w/ DePT)
python parallel_runner.py --cfg kgcoop
python parallel_runner.py --cfg kgcoop_dept

# Running MaPLe (w/ DePT)
python parallel_runner.py --cfg maple
python parallel_runner.py --cfg maple_dept
```

After running, the output will be in the `outputs/` directory, the results will be tallied in the `results/` directory as csv, and a mail will be sent to email address.

If you want to add your own models, you'll need to write your models in the `trainers/` directory and register them in dassl, then configure the settings in the `configs/` directory and `train.py` file, and add your new tasks to the `configs.py` file. Then you can run `python parallel_runner.py --cfg your_model` to run our own model.

----

# Acknowledgements

Our code is based on [CoOp, CoCoOp](https://github.com/KaiyangZhou/CoOp), [KgCoOp](https://github.com/htyao89/KgCoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [DePT](https://github.com/Koorye/DePT) repositories. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
