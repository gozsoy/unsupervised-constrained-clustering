# Unsupervised Constrained Clustering with Contrastive Learning

This project is built on DC-GMM idea, hence borrows code from https://github.com/lauramanduchi/DC-GMM.

## Motivation

In this project, we work on making DC-GMM fully unsupervised, which can automatically understand similarity notions hidden in the given dataset without any external help. The preceding VaDE paper, which is a special case of DC-GMM without any constraints is also fully unsupervised. However, our method differs from it by having constraints as in DC-GMM, but generating them in a self-supervised way using contrastive learning.

## Data Download

Download STL data (Matlab files) from https://cs.stanford.edu/~acoates/stl10/. Change line 106 of utils.py to the save directory.


## Reproducing results on ETH Euler cluster

## Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

## Setup Conda Environment, Load Modules, Activate Conda Environment

```
conda env create -f environment.yml
module load gcc/6.3.0 cuda/10.1.243 cudnn/7.6.4 python_gpu/3.8.5 eth_proxy
conda activate semester_project_env
```

## Run Tasks (please refer to report)
First
```
cd src/
```

For task 1, change experiment settings from config_task1.yml. Then
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python main.py --config ../config_task1.yml
```

For task 2, change experiment settings from config_task2.yml. Then
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python main.py --config ../config_task2.yml
```

For task 3, change experiment settings from config_task3.yml. Then
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python main_contrastive.py --config ../config_task3.yml
```

