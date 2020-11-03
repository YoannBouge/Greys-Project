from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


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
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.FloatTensor)
    #print(image.shape)
    #print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    
    output = model.forward(image)
    print(output)
    
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath,map_location=torch.device('cpu'))
    
    if checkpoint['arch'] == 'vgg16':
        
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 4)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

#################################################Rempli les chemin la bien 
data_dir = 'data/oct'
test_dir = data_dir + '/test'

import json
with open('json/OCT_to_name.json', 'r') as f :
    json_to_name = json.load(f)
    


# Cette fonction test toutes les images du repertoire de test de la classe donnée en paramêtre i,
# et renvoie la probabilité que le model chargé dans le checkpoint trouve la bonne pathologie pour tel classe. 
def evaluation (i,json_to_name):
    vert=[]
    rouge=[]
    bon_reponse=0
    compute =0
    model = load_checkpoint('checkpoints/checkpoint_oct.pth')
    for dirname, _, filenames in os.walk(test_dir+'/{}'.format(i)):
        for filename in filenames:
            
            print(os.path.join(dirname, filename).replace('\\','/'))
            
            compute+=1
            chemin = convertisseur(os.path.join(dirname, filename).replace('\\','/'))
            probs, classes = predict(chemin, model,topk=4) 
            j=str(i)
            print("Résultats attendu = "+json_to_name[j]+"Rsultats Trouver "+ classes[0])
            if(classes[0]==j):
                print('LA MEME CHOSE BONREPO++' )
                vert.append(probs[0])
            else :
                for k in range (4): 
                    if(classes[k]==j):
                        rouge.append(probs[k])
                print ("Mauvaise réponse cette foi ci  ")           
    tot=(len(vert)+len(rouge))
    print("Nombre total d'essaies pour cette classe ")
    print(tot)
    ver=len(vert)
    pourc=ver/tot
    print("Pourcentage de bonne réponse pour cette classe ")
    print(pourc)
    
    
    return  pourc

#PS FAUT GARDER PROBAS DANS UN TABLEAU AILLEURS AVEC LE I QUI LE CORRESPEND          
probas=evaluation(4,json_to_name)