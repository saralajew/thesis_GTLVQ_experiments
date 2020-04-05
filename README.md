# Package with the experiments of my PhD thesis of Section 3.4
[![License](https://img.shields.io/pypi/l/Django.svg)](https://github.com/saralajew/thesis_GTLVQ_experiments/blob/master/LICENSE)

In Section 3.4 I studied the accuracy of 

* Generalized Learning Vector Quantization (GLVQ), 
* Generalized Matrix Learning Vector Quantization (GMLVQ), and
* Generalized Tangent Learning Vector Quantization (GTLVQ)

with different numbers of prototypes on the following datasets:

* Circle;
* Spiral;
* MNIST;
* Cifar10;
* Indian Pine.

The objective of these experiments was to study the interpretability and 
accuracy performances and to compare the methods. The thesis 
can be found here: **will be updated as soon as the thesis published**.

The experiments are based on the internal **anysma** package. The provided 
snippet of the package is mostly undocumented. To get an idea how it works 
and how the package can be used to create LVQ models that can be trained and
evaluated on a GPU see the available files in the `./examples` directory.

![anysma logo](anysma_logo.PNG)

## Installation
To run the scripts you have to install anysma. Clone the script and go to 
the directory by calling:

```
git clone https://github.com/saralajew/thesis_GTLVQ_experiments.git
cd thesis_GTLVQ_experiments
```

Before you install the package it is recommended to create a virtual 
environment via pipenv or to use a docker container. Since I have not tested
the package with other versions of the required package **I made the version 
requirements strict**. Important notes related to this are:

* use Python **3.6**;
* if you wanna use Tensorflow-GPU, install **CUDA Version 9.0.** and **CUDNN 
  7.1.4**;

Internally, we use anysma with different Tensorflow **1** versions without 
any problems. Therefore, feel free to try other Tensorflow **1** versions. 
In this case, you can use other Python and CUDA versions.

If the above requirements are met install the package by:

```
pip install -e .
```

Now, you can start a script from the `./experiments` directory by simpling 
executing:

```
python ./experiments/<path_to_the_script>
```

See the experiments folder for further descriptions related to the scripts.
