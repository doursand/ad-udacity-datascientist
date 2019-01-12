from deeplearning import deep_learning_model
import argparse

if __name__ == '__main__':
    
    #parser
    parser = argparse.ArgumentParser(description='train neural network')
    parser.add_argument('data_directory', help='provide the directory where all images are stored, split by train, valid and test folders',type=str)
    parser.add_argument('--gpu',help='set this option to enable the gpu support. By default training is performed on cpu',action='store_true',default=False)
    parser.add_argument('--save_dir',help='set the save directory for the checkpoint. If not specified then local directory',action='store',default='.',type=str)
    parser.add_argument('--arch', help='provide the selected pretrained model architecture. If not specified then vgg16',action='store',default='vgg16',type=str)
    parser.add_argument('--learning_rate',help='provide the learning rate for the training. If not specified then 0.001',action='store', type=float,default=0.001)
    parser.add_argument('--hidden_units',help='provide the amount of neurons in the hidden layer. If not specified then 4096',action='store',type=int,default=4096)
    parser.add_argument('--epochs',help='provide number of epochs for the training. If not specified then 10',action='store',type=int,default=10)
    parser.add_argument('--version', action='version',version='%(prog)s 1.0')
    parser.parse_args()
    args = parser.parse_args()

    #0. instantiate object to build the model    
    nn_model=deep_learning_model(args.data_directory, args.learning_rate,args.hidden_units,args.epochs,args.save_dir,args.arch,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225],args.gpu)
    
    #1. prepare data for training
    nn_model.prepare_images()
    
    #2. select architecture, freeze gradient, create classifier
    nn_model.prepare_model()
    
    #3. start training and report loss, acc on training set and acc on valid set
    nn_model.train_model()
    
    #4. save checkpoint to use the model without training afterwards
    nn_model.save_model()
        
        
        
