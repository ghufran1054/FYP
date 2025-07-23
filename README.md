### Code Repository for all Experiments for the Project Video Models for Action recognition on edge devices

#### Folder Introduction
1. MiniROAD Folders contains the code for the GRU model and includes all the modification and scripts inside it for training and testing.


2. onnx Folder contains the onnx files for the final used models.

3. IMFE Folder contains the code for the IMFE model with its implementation.

4. TX2 scripts contains the script to run the pipeline on TX2. Its just a simple python script.
5. helping_python_scripts contains any cleaning, transformation and checking scripts for THUMOS dataset.


#### Setup Instructions
1. All the experiments are done using PyTorch for the python environment setup you can follow the instructions in the repo of [MiniROAD](https://github.com/jbistanbul/MiniROAD)


2. Dataset used for all Experiments is THUMOS dataset and the setup instructions can be found in the repo above.

3. If you want to experiment with the ResNet50 backbones you can goto mmaction repository using the [mmaction](https://github.com/open-mmlab/mmaction2/tree/54bc294bff7f742777f5cbb195705c1040a72ebe/configs/recognition/tsn) . From here you can download the weights for RGB models. Setup the mmaction repository to use these ResNet50 weights.

#### Checkpoints and onnx files
All the onnx files and checkpoints will be updated here with a drive link later.

#### Acknowledgement
The code for the GRU model is from the [MiniROAD](https://github.com/jbistanbul/MiniROAD) repository and the insipiration for the implementation of IMFE model is from [RT-Hare](https://arxiv.org/pdf/2409.05662).