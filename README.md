# Optimal Image Classification Model

## Introduction
This is a TensorFlow implementation for image classification based on pre-trained models.
This repository classifies the input images into 8 classes.

![image](https://github.com/byunghyun23/Image-Classification/blob/main/acc.png)

## Requirements
* Linux or Windows
* Python 3
* Tensorflow 2.4

## Dataset
For training the model, you need to download the dataset [Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images). Then, move the downloaded images(train data) to
```
data/train
```
For test data, move some of the training data or other images to
```
data/test
```
The training data path is
```
data/train/airplane
data/train/car
data/train/cat
data/train/dog
data/train/flower
data/train/fruit
data/train/motorbike
data/train/person
```
This repository contains training and test data. (No data download required.)

## Models
Pre-trained models
* DenseNet121
* DenseNet201
* MobileNetV2
* MobileNetV3Large
* EfficientNetB0
* EfficientNetB1
* Xception
* InceptionV3
* ResNet50
* ResNet50V2
* ResNet152V2
* VGG16
* VGG19
Comparison of models
| model            | val\_accuracy | accuracy | Training time (sec) |
| ---------------- | ------------- | -------- | ------------------- |
| InceptionV3      | 1             | 0.9967   | 16.12               |
| ResNet152V2      | 0.9982        | 1        | 33.2                |
| DenseNet201      | 0.9982        | 0.9984   | 37.78               |
| DenseNet121      | 0.9982        | 0.9967   | 18.56               |
| ResNet50V2       | 0.9964        | 1        | 11.97               |
| Xception         | 0.9964        | 0.9934   | 12.88               |
| MobileNetV2      | 0.9964        | 0.9885   | 13.45               |
| VGG16            | 0.9872        | 0.9836   | 10.28               |
| VGG19            | 0.9763        | 0.9902   | 19.19               |
| ResNet50         | 0.75          | 0.7902   | 22.56               |
| MobileNetV3Large | 0.3066        | 0.3246   | 12.05               |
| EfficientNetB0   | 0.1569        | 0.1459   | 12.81               |
| EfficientNetB1   | 0.1442        | 0.1918   | 17.11               |

## Training
Run
```
python train.py
```

## Testing
Run
```
python test.py
```
