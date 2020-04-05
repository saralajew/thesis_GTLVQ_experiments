# The scripts to reproduce the results of Section 3.4
For each experiment (an exception is the `subspace_dimension_estimation.py`), 
I uploaded one `*.h5` file with the weights of a trained model. These trained 
model weights correspond to the weights used to create the images in the 
thesis. The experiments are grouped into two categories:
* toy datasets and
* real-world datasets.

The toy datasets are the **spiral** and the **circle** dataset. For the 
real-world datasets I used **MNIST**, **Cifar10**, and **Indian Pine**. 

To start a script (e.g., the GLVQ model on MNIST) execute (make sure you are
in the `./experiments` directory):

```
python ./MNIST/GLVQ.py
```

The training procedure that is automatically started is the same as used in 
the thesis. After the training an evaluation procedure is triggered that 
generates the evaluation images used in the thesis.

The scripts are parametrized and some parameters are equivalent for all 
scripts (an exception is the `subspace_dimension_estimation.py`). These 
parameters are:

| parameter | argument | description |
|---|---|---|
| `-w` or `--weights` | `<path_to_the_h5_file>` | load an available weight file |
| `--save_dir` | `<path_to_the_output_directory>`; default: `./output` | specify an output folder |
| `--gpu` | `<number_of_GPU>`; default: `0` | device number of the GPU used for the calculations|
| `--eval` | none | skip model training and start evaluation |

For example call 
```
python ./MNIST/GLVQ.py -w ./MNIST/trained_model_GLVQ.h5 --eval
```
to execute the evaluation procedure on MNIST of the GLVQ model with one 
prototype per class.

In the following, I describe the content of the single folders, special 
parameters, and preparations to be made in order to call the scripts.

## Circle

```
cd ./circle
```

A GLVQ, GTLVQ, and GMLVQ model with different number of prototypes. The 
decision boundary that has to be learned is a circle. The dataset is 
provided in the `./data` directory by two numpy files:
* `./data/labels.npy` the labels;
* `./data/points.npy` the corresponding data points.

By executing the file `generate_circle_dataset.py` another instance of the 
dataset can be generated. If the dataset is already available, the output is
a plot of the dataset. Credits for this implementation go to:

https://github.com/hyounesy/TFPlaygroundPSA/blob/master/src/dataset.py

The following pre-trained models are available:
* `trained_model_glvq.h5` the weights for the GLVQ model;
* `trained_model_gmlvq.h5` the weights for the GMLVQ model;
* `trained_model_gtlvq.h5` the weights for the GTLVQ model;
* `trained_model_gtlvq_init.h5` the weights for the GTLVQ model after the 
  initialization.
  
To call the GLVQ model training and evaluation use the following:
```
python GLVQ_circle.py
```

By setting the parameter `--mode` to `GLVQ`, `GMLVQ`, or `GTLVQ` another LVQ
 model type can be called. For example,
 ```
python GLVQ_circle.py --mode GTLVQ -w trained_model_gtlvq.h5 --eval
```
executes the evaluation of the GTLVQ model with the available weights.

## Spiral

```
cd ./spiral
```

The available files for the Spiral dataset are similar to the Circle dataset. 
There are three different models available: GLVQ, GMLVQ, and GTLVQ. I 
provide pre-trained weights and the instance of the dataset I used during 
the evaluations. However, other versions can be generated. The 
classification task is binary and consists of classifying two Archimedean 
spiral arms.

For further notes see the **Circle** dataset.

## MNIST: Subspace dimension estimation

```
cd ./subspace_dimension_estimation
```

The execution of the available file starts the experiment to estimate the 
subspace dimension of GTLVQ on MNIST.
```
python subspace_dimension_estimation.py
```
This command starts the entire experiment I used in the thesis. There are no
additional parameters available and no weights are provided.

After evaluating all the models several times, the plot of my thesis is 
generated.

## MNIST: Accuracy and interpretability

```
cd ./MNIST
```

This directory contains the MNIST experiments with GLVQ, GMLVQ, and GTLVQ 
with several configurations of the number of prototypes. The number of 
prototypes can be selected by the additional parameter `-m`. For example,
```
python GLVQ.py 
```
starts the training and evaluation of the GLVQ model with one prototype per 
class. The command 
```
python GLVQ.py -m 
```
starts the training and evaluation of the GLVQ model in the 1M parameter 
setting, which is in this case 128 prototypes per class. This parameter is 
available for all the three LVQ implementations. After the training, the 
models generate different visualizations of the learned prototypes. 

Additionally, I provide pre-trained weight files for all models and 
configurations:
* `trained_model_GLVQ.h5` the weights for the GLVQ model with one prototype 
  per class;
* `trained_model_GLVQ_1M.h5` the weights for the GLVQ model with 128 
  prototypes per class; 
* `trained_model_GMLVQ.h5` the weights for the GMLVQ model with one prototype 
  per class;
* `trained_model_GMLVQ_1M.h5` the weights for the GMLVQ model with 49 
  prototypes per class; 
* `trained_model_GTLVQ.h5` the weights for the GTLVQ model with one prototype 
  per class;
* `trained_model_GTLVQ_1M.h5` the weights for the GTLVQ model with 10 
  prototypes per class.
  
  
## Indian Pine

```
cd ./indian_pine
```

Training or evaluation of a GTLVQ model on the Indian Pine dataset. The 
dataset must be downloaded from 
* http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat 
  (corrected spectral signals);
* http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat (the labels).

The `*.mat` files must be saved into a directory `./data` so that we have 
the following directory structure: `indian_pine/data/*.mat`.

By the command 
```
python GTLVQ_indian_pine.py
```
we start the training and evaluation of a GTLVQ model. The file 
`trained_model.h5` provides weights of a trained model.

## Cifar10

```
cd ./Cifar10
```

GTLVQ model on the Cifar10 dataset with one prototype per class and a 
subspace dimension of 12. Available files are:
* `GTLVQ.py`: Script to train or evaluate the GTLVQ model;
* `trained_model.h5`: Weights of a trained GTLVQ model.
