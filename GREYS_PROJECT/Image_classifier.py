import matplotlib

import os
import matplotlib.pyplot as plt
import matplotlib as mpl


import seaborn as sb
import cv2
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import collections
from collections import OrderedDict

def code(bd_name, json_name, nb_epoch = 8, route = "", train = False, analyze = False, use_model="vgg16"):
    data_dir = 'data/{}'.format(bd_name)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/val'
    test_dir = data_dir + '/test'


    # Define transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder It depends ou our bd_name
    
    
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    Batch_sizes={'oct':64,'BD_melanome':64, 'flowers':64, 'brain_tumor':64, 'blood_cells':64} 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=Batch_sizes[bd_name], shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=Batch_sizes[bd_name])
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=Batch_sizes[bd_name])
     #A VARIABLE FOR THE DENSTE TRAIN FONCTION 
    dataloaders={'train':train_loader,'val':validate_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is : {}".format(device))
    
    #Lecture de JSon
    import json

    with open('json/{}.json'.format(json_name), 'r') as f:
        json_to_name = json.load(f)
        
    print(len(json_to_name))
    print("on arrve a lire le bon json ")
    nb_classes = len(json_to_name)
    print(json_to_name)

    # Build and train your network
    # Transfer Learning 

    # Freeze pretrained model parameters to avoid backpropogating through them
    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False
        print("set parameter required grad succes")

    def set_optimizer(use_model, model): 
        if (use_model=="vgg16"):
            criterion = nn.NLLLoss()
        elif (use_model=="densenet") :
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        print("set optimizer succes")
        return criterion, optimizer

    
    def test_accuracy(model, test_loader):
        # Do validation on the test set
        model.eval()
        model.to(device)

        with torch.no_grad():

            accuracy = 0

            for images, labels in iter(test_loader):

                images, labels = images.to(device), labels.to(device)

                output = model.forward(images)

                probabilities = torch.exp(output)
            
                equality = (labels.data == probabilities.max(dim=1)[1])
            
                accuracy += equality.type(torch.FloatTensor).mean()
            
            print("Test Accuracy: {}".format(accuracy/len(test_loader))) 

    
    def save_checkpoint(model):
        if(use_model=="vgg16"):


            model.class_to_idx = training_dataset.class_to_idx

            checkpoint = {'arch': "vgg16",
                    'class_to_idx': model.class_to_idx,
                    'model_state_dict': model.state_dict()
                }
            torch.save(checkpoint,'checkpoints/checkpoint_{}.pth'.format(bd_name))
            
        else:
       
            model.class_to_idx = training_dataset.class_to_idx

            checkpoint = {'arch': "densnet169",
                        'class_to_idx': model.class_to_idx,
                        'model_state_dict': model.state_dict(),}
            torch.save(checkpoint,'checkpoints/checkpoint_{}.pth'.format(bd_name))


    def load_checkpoint(filepath):    

        checkpoint = torch.load(filepath)

        if checkpoint['arch'] == 'vgg16': 

            model = models.vgg16(pretrained=True)
            set_parameter_requires_grad(model)
            model.class_to_idx = checkpoint['class_to_idx']
            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                                    ('relu', nn.ReLU()),
                                                    ('drop', nn.Dropout(p=0.5)),
                                                    ('fc2', nn.Linear(5000, nb_classes)),
                                                    ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier

        elif checkpoint['arch'] == 'densnet169': 

            model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet169', pretrained=True)
            set_parameter_requires_grad(model)
            model.class_to_idx = checkpoint['class_to_idx']
            #model.classifier = nn.Linear(model.classifier.in_features, 7)
            model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, nb_classes), nn.LogSoftmax(dim=1))

        else: 
            print("Architecture not recognized.")


        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model

       
    def  initialize_model(use_model, nb_classes, use_pretrained=True):
        if (use_model=="vgg16"):
            model = models.vgg16(pretrained=True)
            model.to(device)
            set_parameter_requires_grad(model)
            # Build custom classifier
            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                                    ('relu', nn.ReLU()),
                                                    ('drop', nn.Dropout(p=0.5)),
                                                    ('fc3', nn.Linear(5000, nb_classes)), 
                                                    ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier
             # Loss function and gradient descent
            
        elif (use_model == "densenet"):
            model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet169', use_pretrained)
            set_parameter_requires_grad(model)
            #model.classifier = nn.Linear(model.classifier.in_features,nb_classes)
            model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, nb_classes), nn.LogSoftmax(dim=1))
            
        else:
            print("Invalid model name, exiting...")
            exit()
        
        criterion, optimizer = set_optimizer(use_model, model)
        print("Initialize model OK !")
        return model

        # Function for the validation pass
    def validation(model, validateloader, criterion):

        val_loss = 0
        accuracy = 0
        
        for images, labels in iter(validateloader):

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            val_loss += criterion(output, labels).item()

            probabilities = torch.exp(output)
            
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return val_loss, accuracy
        
    if(use_model=="densenet"):
        import copy
        import time
        def train_model(model, dataloaders, criterion, optimizer, num_epochs=3, is_inception=False):
            since = time.time()

            val_acc_history = []

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            
            model.cuda()

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
 
                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if is_inception and phase == 'train':
                                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                print(type(model(inputs)))
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = model(inputs).to(device)
                                loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        val_acc_history.append(epoch_acc)
                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
            model.load_state_dict(best_model_wts)
            save_checkpoint(model)
            return best_acc
    

    # load best model weights
     
    #return model, val_acc_history

    #Definition de la fonction de l'entrainement pour densne net 


                   

    def train_classifier(model):
        print('First step 1 on the train ')
        criterion, optimizer = set_optimizer(use_model, model)
        if(use_model=="vgg16"):
            epochs = nb_epoch
            steps = 0
            print_every = 40
            print('step2')
            model.to(device)
            print('step3')
            for e in range(epochs):
            
                model.train()
                running_loss = 0

                for images, labels in iter(train_loader):
            
                    steps += 1
            
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
            
                    output = model.forward(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            
                    running_loss += loss.item()
            
                    if steps % print_every == 0:
                    
                        model.eval()
                    
                        # Turn off gradients for validation, saves memory and computations
                        with torch.no_grad():
                            validation_loss, accuracy = validation(model, validate_loader, criterion)
                
                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                                "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                                "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                                "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
                
                        running_loss = 0
                        model.train()
        else:
            train_model(model,dataloaders, criterion, optimizer, num_epochs=nb_epoch, is_inception=False)

    
    if train == True : 
        if os.path.isfile('checkpoints/checkpoint_{}.pth'.format(bd_name)) :
            print("GO to use : checkpoint_{}.pth".format(bd_name))
            model = load_checkpoint('checkpoints/checkpoint_{}.pth'.format(bd_name))
            criterion, optimizer = set_optimizer(use_model, model)
        else : 
            print("checkpoint_{}.pth doesn't existe... GO to create checkpoint".format(bd_name))
            model = initialize_model(use_model, nb_classes, use_pretrained=True)
        train_classifier(model)
        print("train_classifier OK!!")
        test_accuracy(model, test_loader)
        save_checkpoint(model)

    if analyze == True :
        print("GO to use : checkpoint_{}.pth".format(bd_name))
        model = load_checkpoint('checkpoints/checkpoint_{}.pth'.format(bd_name)) 

        
    

    from PIL import Image

    def process_image(image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        
        # Process a PIL image for use in a PyTorch model
        
        pil_image = Image.open(image_path)
        
        # Resize
        if pil_image.size[0] > pil_image.size[1]:
            pil_image.thumbnail((5000, 256))
        else:
            pil_image.thumbnail((256, 5000))
            
        # Crop 
        left_margin = (pil_image.width-224)/2
        bottom_margin = (pil_image.height-224)/2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        
        pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
        
        # Normalize
        np_image = np.array(pil_image)/255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        
        # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
        # Color channel needs to be first; retain the order of the other two dimensions.
        np_image = np_image.transpose((2, 0, 1))
        
        return np_image

    def convertisseur(image_path):
        os.makedirs("img_convert", exist_ok=True)
        fichier = image_path.split('/')
        fichier_nom = fichier[-1].split('.')
        fichier_nom_sans_ext = fichier_nom[0]
        img = Image.open(image_path)
        rgb_im = img.convert('RGB')

        rgb_im.save('img_convert/{}.jpg'.format(fichier_nom_sans_ext))

        new_file_path = 'img_convert/{}.jpg'.format(fichier_nom_sans_ext)

        return new_file_path

    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        if title is not None:
            ax.set_title(title)
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax


    # Implement the code to predict the class from an image file

    def predict(image_path, model, topk=3):   ##############################################################
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        while(nb_classes<topk):
            topk=topk-1

        image = process_image(image_path)
        
        
        # Convert image to PyTorch tensor first
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        #print(image.shape)
        #print(type(image))
        
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        image = image.unsqueeze(0)
        
        output = model.forward(image)
        
        probabilities = torch.exp(output)
        
        # Probabilities and the indices of those probabilities corresponding to the classes
        top_probabilities, top_indices = probabilities.topk(topk)
        
        # Convert to lists
        top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
        top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
        
        # Convert topk_indices to the actual class labels using class_to_idx
        # Invert the dictionary so you get a mapping from index to class.
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        print(idx_to_class)

        top_classes = [idx_to_class[index] for index in top_indices]

        return top_probabilities, top_classes


    if analyze == True : 

        chemin = convertisseur(route) ###################################################
        print(chemin)
        
        
        probs, classes = predict(chemin, model.cuda())  
        print(probs)
        print(classes)

        
        #COMMENT IL RECUPERE LES JSON
        # Display an image along with the top 5 classes

        # Plot flower input image
        plt.figure(figsize = (6,10))
        plot_1 = plt.subplot(2,1,1)

        image = process_image(chemin)
        route_split = route.split('/')

        try : 
            flower_title = json_to_name[route_split[-2]] 
        except KeyError : 
            name_image = chemin.split('/')
            flower_title = name_image[-1]

        imshow(image, plot_1, title=flower_title)
        # Convert from the class integer encoding to actual flower names
        flower_names = [json_to_name[i] for i in classes]

        # Plot the probabilities for the top 5 classes as a bar graph
        plt.subplot(2,1,2)

        sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0])

        plt.show()
#bd_name = "flowers"
#json_name = "flower_to_name"
    
#filename="./data/flowers/train/5/image_05089.jpg"
#code(bd_name, json_name, filename, analyze = True)
