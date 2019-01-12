from deeplearninginfer import deep_learning_inference
import argparse

if __name__ == '__main__':
    
    #parser
    parser = argparse.ArgumentParser(description='make inference using pre saved neural network')
    parser.add_argument('image_path', help='provide the path to the image for which we want the inference',type=str)
    parser.add_argument('checkpoint_path', help='provide the path to the checkpoint of the saved trained model',type=str)
    parser.add_argument('--gpu',help='set this option to enable the gpu support. By default training is performed on cpu',action='store_true',default=False)
    parser.add_argument('--top_k',help='provide the top k probabilities returned by the model. If not specified then 3',action='store',default=3,type=int)
    parser.add_argument('--category_names', help='provide the mapping of categories to real names. If not specified then set to cat_to_name.json',action='store',default='cat_to_name.json',type=str)
    parser.add_argument('--version', action='version',version='%(prog)s 1.0')
    parser.parse_args()
    args = parser.parse_args()

    #0. instantiate object to build the model
    inf_model = deep_learning_inference(args.image_path,args.checkpoint_path,args.top_k,args.gpu,args.category_names)
    
    #1. load previously saved model
    inf_model.load_model()
    
    #2. prediction
    inf_model.predict()
    
