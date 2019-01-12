import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
from workspace_utils import active_session
import time
import argparse
import json

#class including all functions to do inference using pre saved model
class deep_learning_inference:
    def __init__(self,image_path,checkpoint_path,top_k=3,gpu=False,category_names='cat_to_name.json'):
        self.image_path=image_path
        self.checkpoint_path=checkpoint_path
        self.top_k=top_k
        self.category_names=category_names
        self.gpu=gpu
        with open(category_names, 'r') as f:
            self.cat_to_name = json.load(f)
    
    # function to load pre saved model, including model architecture and number of hidden units
    def load_model(self):
        checkpoint_path=self.checkpoint_path
        if self.gpu:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        #recreate a vggXX pretrained network        
        if checkpoint['arch']=='vgg16':
            nmodel = models.vgg16(pretrained=True)
        elif checkpoint['arch']=='vgg13':
            nmodel = models.vgg13(pretrained=True)
        elif checkpoint['arch']=='vgg19':
            nmodel = models.vgg19(pretrained=True)
        
        #freeze all parameters as it is pretrained
        for param in nmodel.parameters():
            param.requires_grad = False
     
        #recreate the classifier and assign it to nmodel
        classifier = nn.Sequential(OrderedDict(
                        [
                          ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]
                          ))
                        
        nmodel.classifier = classifier
        nmodel.class_to_idx = checkpoint['class_to_idx']
        nmodel.load_state_dict(checkpoint['state_dict'])
        self.model=nmodel
       
    # function for image preprocessing
    def process_image(self,image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
                returns an Numpy array
            '''
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        w= image.size[0]
        h= image.size[1]

        if w < h:
            w=256              
        else:
            h=256
        image.thumbnail((w,h))

        a=(image.size[0]-224)/2
        b=(image.size[1]-224)/2
        c= a + 224
        d= b + 224

        image = image.crop((a,b,c,d))

        np_image = np.array(image)/255
        np_image = (np_image-mean)/std
        np_image = np_image.transpose((2,0,1))
        return np_image

    def invert_dict(self,d):
        inverse = {v: k for k, v in d.items()}
        return inverse
    
    def return_flowers(self,flower_class):
        model=self.model
        cat_to_name=self.cat_to_name
        returned_list = []
        #invert class to idx to ease mapping with outcome of predict function
        idx_to_class = self.invert_dict(model.class_to_idx)
        for x in np.nditer(flower_class):
            returned_list.append(cat_to_name.get(idx_to_class.get(x.item())))
        return returned_list
    
    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        model=self.model
        top_k=self.top_k
        
        image_path=self.image_path
        
        im = Image.open(image_path)
    
        #process image to make it usable by the network
        im_np = self.process_image(im)
    
        im_ts = torch.from_numpy(im_np).type(torch.FloatTensor)
        
        # add batch dimension as it is required at the beginning of the image tensor
        im_ts.unsqueeze_(0)

        # disable gradient calc to speed up inference
        with torch.no_grad():
            prediction = model.forward(im_ts)
        # use exponential as the probabilities are expressed in log
        probs, classes = torch.exp(prediction).topk(top_k)
        #print(list(np.array(probs[0])), list(np.array(classes[0])), self.return_flowers(classes))
        #return list(np.array(probs[0])), list(np.array(classes[0])), self.return_flowers(classes)
        print("\n=======================================================================")
        print("Top {} most likely flower names :".format(top_k))
        print("=======================================================================\n")
        for i in (range(top_k)):
            print("Flower name : {} ... ".format(self.return_flowers(classes)[i]),
                "Probability : {} ... ".format(np.array(probs[0])[i])
                 )
      
    