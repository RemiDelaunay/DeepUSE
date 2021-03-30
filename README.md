# DeepUSE : Learning-based ultrasound elastography

This repository contains fine-tuning code and pre-trained models for the learning-based quasi-static ultrasound elastography method presented in "An unsupervised learning approach to ultrasound elastography with spatio-temporal consistency" [[1]][paper-link]. Please check our paper for more details and, should you be making use of this work, please cite the paper accordingly.

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/network_architecture.jpg "Overview of the method")

# Setup

To use the DeepUSE library, please clone this repository and install the requirements listed in requirements.txt.
    
    pip -r install requirements.txt

To use PyTorch with CUDA, please check their [official website](www.pytorch.org).

# Network architecture

Two types of network architectures are available. USENet is a feedforward convolutional neural network derived from the U-Net and can be trained to estimate the displacement between a pair of ultrasound radio-frequency (RF) frames. ReUSENet is a recurrent neural network which make use of decoding convLSTM units to process time series ultrasound RF data.

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/network_architecture.jpg "Networks architecture")

# Usage

If you would like to train a model on your own data, you will likely need to customize your own dataloader in [DeepUSE/dataset/][dataset-dir] and configuration file in [DeepUSE/config/][config-dir]. You can then train your model using the following command:

```python
python train.py ./config/yourConfigFile.json
```

During training the model is saved every n epochs by specifying the "model_update_freq" in your configuration file. At inference, you can use the same configuration file and specify which model to use by specifying the number of epoch it has been trained for by using:

```python
python validate.py ./config/yourConfigFile.json --epoch 300
```

Pre-trained models of ReUSENet and USENet trained on both [numerical simulation][] and [in vivo data][] are also available in [DeepUSE/results/][results-dir].

For real-time inference, please have a look at our [3D-slicer extension][slicer-module].

![alt text](https://github.com/YipengHu/example-data/raw/master/label-reg-demo/media/network_architecture.jpg "Inference example")

# References

```
[1]  @article{Delaunay2021,
  title={An unsupervised learning approach to ultrasound elastography with spatio-temporal consistency},
  author={Delaunay, R{\'e}mi and Hu, Yipeng and Vercauteren, Tom},
  journal={},
  volume={},
  number={},
  pages={},
  year={2021},
  publisher={}
}
```
This PyTorch framework was inspired from: https://github.com/branislav1991/PyTorchProjectFramework

# Contact
For any problems or questions please [open an issue][issue] or send us an [email](mailto:remi.delaunay.17@ucl.ac.uk).




[paper-link]: not-available-yet
[numerical simulation]: https://users.encs.concordia.ca/~impact/ultrasound-elastography-simulation-database/
[in vivo data]: https://www.synapse.org/InVivoDataForUSE
[results-dir]: not-available-yet
[config-dir]: not-available-yet
[dataset-dir]: not-available-yet
[issue]: not-available-yet
[slicer-module]:not-available-yet
