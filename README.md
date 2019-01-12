# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In my version of this project, I created two classes, one including all functions to build a pytorch neural network (deeplearning.py), and another one including all functions to trigger inferences (deeplearninginfer.py)
There are then two associated files (respectively train.py & predict.py) to instantiate objects from each of the two classes and to do the job

usage: train.py [-h] [--gpu] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--version]
                data_directory

 
usage: predict.py [-h] [--gpu] [--top_k TOP_K]
                  [--category_names CATEGORY_NAMES] [--version]
                  image_path checkpoint_path
