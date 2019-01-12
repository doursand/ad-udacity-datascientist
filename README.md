# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In my version of this project, I created two classes, one including all functions to build a pytorch neural network (deeplearning.py), and another one including all functions to trigger inferences (deeplearninginfer.py)
There are then two associated files (respectively train.py & predict.py) to instantiate objects from each of the two classes and to do the job

usage: train.py [-h] [--gpu] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--version]
                data_directory

train neural network

positional arguments:
  data_directory        provide the directory where all images are stored,
                        split by train, valid and test folders

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 set this option to enable the gpu support. By default
                        training is performed on cpu
  --save_dir SAVE_DIR   set the save directory for the checkpoint. If not
                        specified then local directory
  --arch ARCH           provide the selected pretrained model architecture. If
                        not specified then vgg16
  --learning_rate LEARNING_RATE
                        provide the learning rate for the training. If not
                        specified then 0.001
  --hidden_units HIDDEN_UNITS
                        provide the amount of neurons in the hidden layer. If
                        not specified then 4096
  --epochs EPOCHS       provide number of epochs for the training. If not
                        specified then 10
  --version             show program's version number and exit
  
  
  usage: predict.py [-h] [--gpu] [--top_k TOP_K]
                  [--category_names CATEGORY_NAMES] [--version]
                  image_path checkpoint_path

make inference using pre saved neural network

positional arguments:
  image_path            provide the path to the image for which we want the
                        inference
  checkpoint_path       provide the path to the checkpoint of the saved
                        trained model

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 set this option to enable the gpu support. By default
                        training is performed on cpu
  --top_k TOP_K         provide the top k probabilities returned by the model.
                        If not specified then 3
  --category_names CATEGORY_NAMES
                        provide the mapping of categories to real names. If
                        not specified then set to cat_to_name.json
  --version             show program's version number and exit
