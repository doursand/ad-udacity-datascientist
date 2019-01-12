# AI Programming with Python Project

## Overview
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

##

In my version of the project, I created two classes :
- one including all functions to build a pytorch neural network (deeplearning.py)
- one including all functions to trigger inferences (deeplearninginfer.py)

There are two associated files to instantiate objects from each of the two classes and to do the job
- train.py
- predict.py

```shell
usage: train.py [-h] [--gpu] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--version]
                data_directory
```

```shell
usage: predict.py [-h] [--gpu] [--top_k TOP_K]
                  [--category_names CATEGORY_NAMES] [--version]
                  image_path checkpoint_path
```
