# DeepUSE : Learning-based ultrasound elastography

This repository contains fine-tuning code and pre-trained models for the learning-based quasi-static ultrasound elastography method presented in "An unsupervised learning approach to ultrasound elastography with spatio-temporal consistency" [[1]][paper-link]. Please check our paper for more details and, should you be making use of this work, please cite the paper accordingly.

![alt text](https://github.com/RemiDelaunay/Media-example/raw/main/DeepUSE/method_overview.png "Method overview")

# Setup

To use the DeepUSE library, please clone this repository and install the requirements listed in requirements.txt.
    
    pip -r install requirements.txt

To use PyTorch with CUDA, please check their [official website](www.pytorch.org).

# Network architecture

Two types of network architectures are available. USENet is a feedforward convolutional neural network derived from the U-Net and can be trained to estimate the displacement between a pair of ultrasound radio-frequency (RF) frames. ReUSENet is a recurrent neural network which make use of decoding convLSTM units to process time series ultrasound RF data.

![alt text](https://github.com/RemiDelaunay/Media-example/raw/main/DeepUSE/network_architecture.png "Network architecture")

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

![alt text](https://github.com/RemiDelaunay/Media-example/raw/main/DeepUSE/SlicerDeepUSE.gif "Inference example")

# References

```
[1]  @article{10.1088/1361-6560/ac176a,
	author={Delaunay, Rémi and Hu, Yipeng and Vercauteren, Tom},
	title={An unsupervised learning approach to ultrasound strain elastography with spatio-temporal consistency},
	journal={Physics in Medicine & Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ac176a},
	year={2021},
	abstract={Quasi-static ultrasound elastography is an imaging modality that measures deformation (i.e. strain) of tissue in response to an applied mechanical force. In ultrasound elastography, the strain modulus is traditionally obtained by deriving the displacement field estimated between a pair of radio-frequency data. In this work we propose a recurrent network architecture with convolutional Long-Short-Term Memory (convLSTM) decoder blocks to improve displacement estimation and spatio-temporal continuity between time series ultrasound frames. The network is trained in an unsupervised way, by optimising a similarity metric between the reference and compressed image. Our training loss is also composed of a regularisation term that preserves displacement continuity by directly optimising the strain smoothness, and a temporal continuity term that enforces consistency between successive strain predictions. In addition, we propose an open access in vivo database for quasi-static ultrasound elastography, which consists of radio-frequency data sequences captured on the arm of a human volunteer. Our results from numerical simulation and in vivo data suggest that our recurrent neural network can account for larger deformations, as compared with two other feed-forward neural networks. In all experiments, our recurrent network outperformed the state-of-the-art for both learning-based and optimisation-based methods, in terms of elastographic signal-to-noise ratio (SNRe), strain consistency, and image similarity. Finally, our open source code provides a 3D-slicer visualisation module that can be used to process ultrasound RF frames in real-time, at a rate of up to 20 frames per second, using a standard GPU.}
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
