import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import time


#class including all functions to prepare images, prepare model, train model and finally save checkpoint
class deep_learning_model:
    def __init__(self, data_directory, learning_rate,hidden_units,epochs,save_directory,arch='vgg16',dataset_mean=[0.485, 0.456, 0.406],dataset_std=[0.229, 0.224, 0.225],gpu=False):
        self.data_directory = data_directory 
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units=hidden_units
        self.epochs=epochs
        self.save_directory=save_directory
        self.mean=dataset_mean
        self.std=dataset_std
        self.gpu=gpu
    
    # function to prepare images
    def prepare_images(self):
        set_mean=self.mean
        set_std=self.std
        data_dir=self.data_directory
        train_dir=data_dir + '/train'
        valid_dir=data_dir + '/valid'
        test_dir=data_dir + '/test'
        data_transforms = {'training' : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(set_mean, set_std)]),                   
                            'valid' : transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(set_mean, set_std)]),
                  
                            'test' : transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(set_mean, set_std)])
                            }

        image_datasets = {'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
                  'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])    
                        }

        dataloaders = {'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
              'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
              'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
                        }
        self.data_transforms=data_transforms
        self.image_datasets=image_datasets
        self.dataloaders=dataloaders
           
    # function to select model, freeze gradient on pretrained model, create and assign classifier
    def prepare_model(self):
        arch=self.arch
        hidden=self.hidden_units
        print('arch : ' + arch)
        
        if arch=='vgg16':
            model = models.vgg16(pretrained=True)
        elif arch=='vgg13':
            model = models.vgg13(pretrained=True)
        elif arch=='vgg19':
            model = models.vgg19(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict(
                        [
                          ('fc1', nn.Linear(25088, hidden)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                        ]
                          ))
        model.classifier = classifier
        self.model=model
       
    # function to trigger the deep learning
    def do_deep_learning(self, model, trainloader, epochs, criterion, optimizer, device):
        dataloaders=self.dataloaders
        steps = 0

        # change to cuda or cpu
        model.to(device)

        for e in range(epochs):
            running_loss = 0
            running_corrects = 0
            total = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                _,predicted = torch.max(outputs, 1)
            
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_corrects  += (predicted == labels).sum().item()
                total += labels.size(0)
        
            # for each epoch, print loss & accuracy on training set and accuracy on valid set 
            print("Epoch: {}/{}... ".format(e+1, epochs), 
              "Loss: {:.4f}... ".format(running_loss/total),
              "training set Acc: {:.2f} %... ".format(100 * running_corrects/total),
              "valid set Acc: {:.2f} %... ".format(self.calculate_acc(dataloaders['valid'],device)
             )
             )
    # function to calculate the accuracy
    def calculate_acc(self,testloader, device='cuda'):
        model=self.model
        testloader=self.dataloaders["valid"]
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return (100 * correct / total)
    
    #  train model 
    def train_model(self):
        epochs=self.epochs
        learning_rate=self.learning_rate
        model=self.model
        trainloader = self.dataloaders["training"]
        if self.gpu:
            device='cuda'
        else:
            device='cpu'
        print("device : " + device)
        # set optimizer and criterion. Optimizer set to only look at classifier part of the model. Criterion is set to use NLLLoss          function
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
        self.optimizer=optimizer
        
        #start training
        model.train()
        start = time.time()
        print("training started ...")
        with active_session():
            self.do_deep_learning(model, trainloader, epochs, criterion, optimizer, device)
        end = time.time()
        print("training completed ... took (seconds) : {:.2f}".format(end-start)) 
    
    # save model        
    def save_model(self):
        model=self.model
        image_datasets=self.image_datasets
        epochs=self.epochs
        learning_rate=self.learning_rate
        optimizer=self.optimizer
        save_dir=self.save_directory
        arch=self.arch
        hidden_units=self.hidden_units
        
        model.class_to_idx = image_datasets['training'].class_to_idx
        #switch to cpu mode before saving as otherwise gpu will be required each time we are using the trained model
        model.cpu()

        # TODO: Save the checkpoint 

        checkpoint= {
            'epochs' : epochs,
            'learning_rate' : learning_rate,
            'optimizer_state_dict' : optimizer.state_dict(),
            'state_dict' : model.state_dict(),
            'class_to_idx' : model.class_to_idx,
            'arch' : arch,
            'hidden_units' : hidden_units
    
            }

        torch.save(checkpoint, save_dir + 'cmdline_trained_model.pth')
        print('model successfully saved to ' + save_dir +  'cmdline_trained_model.pth')     
        
        
        
        
        
        
        
        
    
    

