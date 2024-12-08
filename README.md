[TOC]

# GDSG: Graph Diffusion-based Solution Generation for Optimization Problems in MEC Networks

## Setup

```bash
conda env create -f environment.yml
conda activate gdsg
```

## Codebase Structure

`diffusion/co_datasets`, dataset wrappers. 

`diffusion/models`, neural networks. 

`diffusion/tools`, executable shell generation tools. 

`diffusion/utils`, task-specific utilities.

`diffusion/pl_meta_model.py`, foundation lightning model. 

`diffusion/pl_msco_model.py`, lightning model for multi-server multi-user computation offloading with diffusion. 

`diffusion/msco_train.py`, executor for pl_msco_model. 

`diffusion/pl_gnn_model.py`, lightning model for multi-server multi-user computation offloading with DiGNN. 

`diffusion/gnn_run.py`, executor for pl_gnn_model. 

`diffusion/mtfnn_run.py`, executor for baseline MTFNN for multi-server multi-user computation offloading. 

`scripts/`, examples of shell scripts for train and test. (It is recommended to use the python script in `diffusion/tools` to generate shells from scratch and run)

`data/`, dataset generation tools and introductions. 

## Reproduction

After completing the environment configuration, follow the instructions in README in `utils/cython_merge` to build related dependencies. 

Please read the shell script generation tool in `diffusion/tools`, set the dataset and checkpoint paths, and then generate the target shell to run. 

## Data & Pretrained Checkpoints

Please download the datasets and pretrained model from here http://ieee-dataport.org/documents/dataset-multi-server-multi-user-computation-offloading. Or you can contact the author directly through GitHub's email information. 

## Gradient acquisition process

When you need to analyze the training process of a certain model, add the  ***grad_calculate***  parameter in the training script. The gradient vectors in the two task directions will be automatically collected during the training process and stored in a *.txt* file with the suffix  ***_grad***  in the same folder. Note that this gradient information file may be large, so it is recommended to set fewer epoch processes when collecting. After obtaining the original .txt file that saves the gradient information, we need to do some processing and analysis. Adjust the parameters of  *grads_angle_op.py* , including model type (diffusion or DiGNN), training data size, whether it is a true value training set, and whether to save the original gradient information (for subsequent analysis). Then run the script and the results will be saved in  *result_storage_path/angle_param* .

### Gradient information file description

In the .txt obtained after training, each two lines are the gradient vectors of the same parameter at the same training step (one line is discrete diffusion and the other line is continuous diffusion), in the format of pytorch tensor. After *grads_angle_op* is processed, the *result_storage_path/reconstructed_grad* folder contains multiple epochs. Each epoch folder contains tensor vectors of multiple parameters at several training steps, still in groups of two lines. The final result is that each parameter in the *result_storage_path/angle_param* file under each epoch contains an angle value in degrees.

## Acknowledgement

This work refers to the work of [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) and [T2T](https://github.com/Thinklab-SJTU/T2TCO), and thanks them for their open source. 