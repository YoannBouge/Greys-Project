from PIL import Image, ImageEnhance
import pylab as py
from os import listdir

#Mettre le chemin vers le dossier :
image_data = "C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/Base_De_Donnee/Tumeur_cerveau/no"



#inverse de l'image
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    print("ouverture ok !")
    largeur,hauteur=imageSource.size
    imageBut=Image.new("RGB",(largeur,hauteur))
    for y in range (hauteur):
        for x in range (largeur):
            p=imageSource.getpixel((x,y))
            imageBut.putpixel((x,-y+hauteur-1),p)
    imageBut.save("inv"+fichier[i])
    print("Inversion ok ! " + str(i) +"/"+ str(len(fichier)))



#Effet miroir
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    print("ouverture ok !")
    colonne,ligne = imageSource.size
    imF = Image.new(imageSource.mode,imageSource.size)
    for y in range(ligne):
           for x in range(colonne):
                pixel = imageSource.getpixel((x,y))
                imF.putpixel((colonne-x-1,y), pixel)
    imF.save("mir"+fichier[i])
    print("Mirroir ok ! " + str(i) +"/"+ str(len(fichier)))


#Augmenter la netteté
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    enhancer = ImageEnhance.Sharpness(imageSource)
    enhanced_im = enhancer.enhance(10.0)
    enhanced_im.save("net"+fichier[i])
    print("Netteté ok ! " + str(i) +"/"+ str(len(fichier)))

 
#Augmenter la luminosité
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    enhancer = ImageEnhance.Brightness(imageSource)
    enhanced_im = enhancer.enhance(1.8)
    enhanced_im.save("lum"+fichier[i])
    print("Luminosité ok ! " + str(i) +"/"+ str(len(fichier)))


#Augmenter le contraste
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    enhancer = ImageEnhance.Contrast(imageSource)
    enhanced_im = enhancer.enhance(4.0)
    enhanced_im.save("cont"+fichier[i])
    print("contraste ok ! " + str(i) +"/"+ str(len(fichier)))

    
#Rotation
fichier= [f for f in listdir (image_data)]
for i in range (len(fichier)):
    print(fichier[i])
    imageSource=Image.open(image_data + "/" + fichier[i])
    rotated_im_45 = imageSource.rotate(30)
    rotated_im_45.save("rot3"+fichier[i])
    print("rot3 ok ! " + str(i) +"/"+ str(len(fichier)))
