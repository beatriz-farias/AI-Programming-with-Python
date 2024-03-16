import torch # version: 2.1.0
import json

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

from PIL import Image

# Functions for train.py

def load_data(data_dir):
    '''Loads the data from the data directory provided
        Argument
            data_dir: path of the data directory
        Returns
            train_data: Transformed data used for training
            trainloader: Dataloader of the train_data
            validloader: Dataloader of the validation data
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Data Transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(p= 0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Image datasets
    train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform= valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_transforms)
    
   # Dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, trainloader, validloader

def setup_model(arch, learning_rate, hidden_units, dropout, device): 

    '''Sets up a model
        Arguments
            arch: Architecture of the model (vgg13, vgg16 or vgg19)
            learning_rate
            hidden_units
            dropout
            device: cpu or gpu(cuda)
        Returns
            model
            criterion
            optimizer
    '''
    
    # Define the architecture
    if (arch == 'VGG'):
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif (arch == 'Densenet'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif (arch == 'Alexnet'):
        model = models.alexnet(pretrained=True)
        input_size = 4096
    else:
        raise ValueError("Invalid architecture. 'arch' must be 'VGG', 'Densenet', or 'Alexnet'.")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p= dropout)),
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 84)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(84, 96)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(96, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    model.to(device)
    
    return model, criterion, optimizer

def train_model(epochs, model, device, trainloader, validloader, criterion, optimizer):
    '''Trains a model
        Arguments
            epochs
            model
            device: cpu or gpu(cuda)
            trainloader
            validloader
            criterion
            optimizer
        No Returns
    '''

    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.4f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.4f}")
                running_loss = 0
                model.train()

def save_model(train_data, model, save_dir, arch, hidden_units, dropout, learning_rate):
    '''Saves a trained model
        Arguments
            train_data
            model
            save_dir: Directory in which the checkpoint is saved
            arch: Architecture of the model (vgg13, vgg16 or vgg19)
            learning_rate
            hidden_units
            dropout
        No Returns
    '''

    model.class_to_idx = train_data.class_to_idx
    path = f'{save_dir}/checkpoint.pt'

    torch.save({'architecture': arch,
                'hidden_units': hidden_units,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()
                }, path)
    
# Functions for predict.py

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        Argument: image_path
        Returns: tensor
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    # Turn PIL image into pytorch Tensor
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    
    tensor = transform(image)
    
    return tensor

def load_model(path, device):
    '''Loads a model from the checkpoint file
        Arguments
            path: checkpoint path
            device: cpu or gpu(cuda)
        Returns
            model
    '''

    checkpoint = torch.load(path)

    arch = checkpoint['architecture']
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']

    model,_,_ = setup_model(arch, learning_rate, hidden_units, dropout, device)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Arguments
            image: image whose species will be predicted 
            model
            topk: k highest probabilities of species
            device: cpu or gpu(cuda)
    '''
    # TODO: Implement the code to predict the class from an image file 
    image = process_image(image)
    
    model.to(device)
    model.eval()
    
    image.to(device)
    image = image.unsqueeze(0)
    image = image.float()
    with torch.no_grad():
        if (device == 'cuda'):
            logps = model.forward(image.cuda())
        else:
            logps = model.forward(image)
        ps = torch.exp(logps)
        probs, indexes = ps.topk(topk, dim=1)
        
        # Reshaping probs and indices to return 1d numpy arrays
    probs = probs.tolist()[0]
    indexes = indexes.tolist()[0]

    idx_to_classes = {index: label for label, index in model.class_to_idx.items()}

    classes = [idx_to_classes[i] for i in indexes]

    return probs, classes

def output_predictions(probs, classes, category_names):
    '''
        Outputs the classes names along with the respectives probabilities
        Arguments
            probs: Probability
            indices: Indice of a class
            category_names: Fie with the dictionary that maps indices to class names
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[i] for i in classes]
    for i in range(len(classes)):
        print(f'The probability that this flower a {classes[i]} is {probs[i]*100:.2f}%')