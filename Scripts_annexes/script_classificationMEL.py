import csv
import shutil,os

with open('C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/Melanomes/HAM10000_metadata.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = ('C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/Melanomes/HAM10000_images_part_1/'+ row['image_id']+'.jpg')
        if row['dx'] == 'df':
            print("df")
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/df' )
        elif row['dx'] == 'nv': 
            print ("nv")  
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/nv' ) 
        elif row['dx'] == 'mel':
            print ("mel")
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/mel' )
        elif row['dx'] == 'vasc':
            print ("vasc")
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/vasc' )
        elif row['dx'] == 'bcc':
            print ("bcc")
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/bcc' )
        elif row['dx'] == 'akiec':
            print ("akiec")
            shutil.move(image_path,'C:/Users/ajmid/Desktop/L3_Bioinfo/Sem_2/Projet/BD_melanome/akiec' )
        else:
            print("else")
            
                