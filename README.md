# Tensorflow Estimator Codebase (MNIST)
This repository is made as a starting point for an ML project. This is an starting point for using the Estimator API in Tensorflow. In this example I wanted to make the use of the Estimator API so clear as possible. I did not choice to use the fastest datainput instead I opted to use an method that can be easily adopted to use for any other goal.

For other tuturials please check https://github.com/GoogleCloudPlatform/tf-estimator-tutorials. 

## Requirements and installation

Required packages:

- Tensorflow

You can download the repository via `git clone https://github.com/takotab/-estimator_mnist_example` and run by using ` python train.py`. There are settings you can change in [train.py](train.py) or in [config.py](config.py).

## Models

There are 4 models:

- [Baseline](https://www.tensorflow.org/api_docs/python/tf/estimator/BaselineClassifier)
- [Simple Neural Network](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)
- [Deep & Wide](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)
- (Custom)[model.py]

Three are premade Estimators by Tensorflow. And one is a custom model. In the current code it easy to change the these Estimators for other (premade) Estimators. It is highly adviced to try to fit your problem in one of the premade Estimators. Building a custum estimator is much more work. 

## Data

I used the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to showcase the different models. In this dataset the goal is classify a handwritten digit. To showcase some additional features I also added custom (random) features. The number of these extra wide features can be changed with `params.extra_wide_features` in [train.py](train.py). You can edit their creation in [data.py](data.py).

## Results

I have run the models and found some intresseting results. First the resulting accuracy:

| Model                         | Accurracy (test set) |
| ----------------------------- | -------------------- |
| Baseline                      | 11.25%               |
| Custom                        | 98.54%               |
| Deep & Wide                   | 77.50%               |
| Deep Neural Network [300 300] | 89.38%               |

The Custom model with Convolutional parts out preforms the others. However the the Deep & Wide does preform worse compared with the DNN. The DNN parts of both are the same and since the extra info the Deep & Wide gets is not relevant you would hypothocis they should preform equal. Espessaily when the weights accociated with the wide features are approximate 0 (as they should be).

Here you see the log of the cost during training against the the number of iteration:
![loss.png][loss.png]

You see the Custom model preforms much better. It is however good to also see the same graph against time:
![loss_time.png][loss_time.png]

The run time of the custum model is circa 15 times longer than the other models. This shows the relevancy of the premade simple models. This runtime difference is also visible in the graph showing the number of global steps per second over the training:

![global_step_time.png][global_step_sec.png]


