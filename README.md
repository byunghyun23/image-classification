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
```
*DenseNet121*
*DenseNet201*
*MobileNetV2*
*MobileNetV3Large*
*EfficientNetB0*
*EfficientNetB1*
*Xception*
*InceptionV3*
*ResNet50*
*ResNet50V2*
*ResNet152V2*
*VGG16*
*VGG19*
```

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
