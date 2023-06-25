## Imbalanced Image Classification

## How to use

To clone and run this application, you'll need [Git](https://git-scm.com/downloads) installed on your computer. From your command line:

## Clone this repository

```$bash
$ git clone https://github.com/hungsvdut2k2/Imbalanced-Image-Classification.git
```

## Go into the repository

```$bash
$ cd Imbalanced-Balanced-Classification
```

## Install packages

```bash
$ pip install -r requirements.txt
```

## Set up your dataset

Structure of these folders.

```
train/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

```
validation/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

## Train your model by running this command line

```bash
$ python train.py --epochs ${epochs} --num-classes ${num_classes}
```
