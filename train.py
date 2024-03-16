import argparse
import torch
import utils

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', default= 'flowers')
parser.add_argument('--save_dir', action='store', dest= 'save_dir', default= 'checkpoint.pt')
parser.add_argument('--arch', action='store', dest= 'arch', choices= ['VGG', 'Densenet', 'Alexnet'], default= 'VGG')
parser.add_argument('--learning_rate', action='store', dest= 'learning_rate', type=float, default= 0.001)
parser.add_argument('--hidden_units', action='store', dest= 'hidden_units', type=int, default= 100)
parser.add_argument('--dropout', action='store', dest= 'dropout', type=float, default= 0.5)
parser.add_argument('--epochs', action='store', dest= 'epochs', type=int, default= 20)
parser.add_argument('--gpu', action='store_true', default= False)

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
dropout = args.dropout
epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == True else 'cpu')

print('Loading the dataset..')
train_data, trainloader, validloader = utils.load_data(data_dir)

print('Setting up the model..')
model, criterion, optimizer = utils.setup_model(arch, learning_rate, hidden_units, dropout, device)

print('Training the model..')
utils.train_model(epochs, model, device, trainloader, validloader, criterion, optimizer)

if (save_dir != None):
    print('Saving the checkpoint..')
    utils.save_model(train_data, model, save_dir, arch, hidden_units, dropout, learning_rate)

print('You successfully trained your model!')