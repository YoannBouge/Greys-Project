# -*-coding:utf-8 -*

from tkinter import * 
from tkinter import filedialog
from tkinter import ttk
from tkinter import tix
import tkinter.font as tkfont
from PIL import Image, ImageTk
import os
import json
import Image_classifier as ic
import shutil

print("start")
taille_fenetre_X = 1800
taille_fenetre_Y = 850 

# On crée une fenêtre, racine de notre interface
fenetre = tix.Tk()

interface_IC = Frame(fenetre, width = taille_fenetre_X, height = taille_fenetre_Y, borderwidth = 1,  highlightbackground="black", highlightcolor="black", highlightthickness=2)
interface_IC.pack(fill=BOTH)
interface_IC.grid_propagate(0)

bouton_quitter = Button(fenetre, text="Quitter", command=fenetre.quit, cursor= "hand2")
bouton_quitter.pack(side = RIGHT)

message_to_userF = Frame(fenetre, width=100, height=100, borderwidth=1,  highlightbackground="blue", highlightcolor="blue", highlightthickness=1)
message_to_userF.pack()
message_to_user = Label(message_to_userF, text=" ", fg = 'red')
message_to_user.pack()


# Déclaration des polices d'écriture
ar36b = tkfont.Font(family = "Arial", size = 36, weight = "bold")
ar18 = tkfont.Font(family ="Arial", size = 18)
ar12 = tkfont.Font(family ="Arial", size = 12)




# On créer des cadres dans la fentre (Entrainement / Analyse)

entrainement = Frame(interface_IC, width=taille_fenetre_X/3, height=taille_fenetre_Y, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=2)
entrainement.grid(row=0, column=0, sticky='nwse')
#entrainement.grid_propagate(0)
analyse = Frame(interface_IC, width=taille_fenetre_X/3, height=taille_fenetre_Y, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=2)
analyse.grid(row=0, column=1, sticky='nwse')
#analyse.grid_propagate(0)
statistiques = Frame(interface_IC, width=taille_fenetre_X/3, height=taille_fenetre_Y, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=2)
statistiques.grid(row=0, column=2, sticky='nwse')
#statistiques.grid_propagate(0)


# Fonction de setup selection 
bd_name = "BD_melanome"            #par défaut case Melanome selectionnée
json_name = "MEL_to_name"          #par défaut case Melanome selectionnée
densenet = False                   #par défaut on utilise vgg16 (True => densenet)
def set_bd_oct(): 
    global bd_name
    bd_name = "oct"
    global json_name
    json_name = "OCT_to_name"
    global densenet
    densenet = False

def set_bd_flowers(): 
    global bd_name
    bd_name = "flowers"
    global json_name
    json_name = "flower_to_name"
    global densenet
    densenet = False

def set_bd_mel(): 
    global bd_name
    bd_name = "BD_melanome"
    global json_name
    json_name = "MEL_to_name"
    global densenet
    densenet = True

def set_bd_tumeurC(): 
    global bd_name
    bd_name = "brain_tumor"
    global json_name
    json_name = "TUMOR_to_name"
    global densenet
    densenet = False

def set_bd_CelluleS(): 
    global bd_name
    bd_name = "blood_cells"
    global json_name
    json_name = "CEL_to_name"
    global densenet
    densenet = False



##############################################################################################
                                # Entrainement contenu
##############################################################################################

train_label = Label(entrainement, text="Entrainer :", font = ar36b)
train_label.pack(side = TOP)

cadre_imp = Frame(entrainement, width=100, height=100, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=0)
cadre_imp.pack(pady=(70,40))

desc_entrainement = Label(cadre_imp, text="Importer une image d'entrainement : ", font = ar18)
desc_entrainement.pack()

selecteur_e = Frame(cadre_imp, width=100, height=100, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=1) 
selecteur_e.pack(pady = 15) # Ce cadre affiche la bordure autour des selecteurs

# Fonction qui definit la 2nd listbox en fonction de la base de donnée selectionnée
def select_bd_train(event):
    select = liste_deroulante_e.get()
    global selection_lb1 
    print("Vous avez selectionné :", select)
    if select == 'OCT':
        set_bd_oct()
        listeBox_classes.delete(0,999)
        listeBox_classes.insert(1, "DME")
        listeBox_classes.insert(2, "CNV")
        listeBox_classes.insert(3, "DRUSEN")
        listeBox_classes.insert(4, "NORMAL")
    elif select == 'Melanome':
        set_bd_mel()
        listeBox_classes.delete(0,999)
        listeBox_classes.insert(1, "BKL")
        listeBox_classes.insert(2, "DF")
        listeBox_classes.insert(3, "NV")
        listeBox_classes.insert(4, "MEL")
        listeBox_classes.insert(5, "VASC")
        listeBox_classes.insert(6, "BCC")
        listeBox_classes.insert(7, "AKIEC")
    elif select == "Tumeur Cérébrale":
        set_bd_tumeurC()
        listeBox_classes.delete(0,999)
        listeBox_classes.insert(1, "Yes")
        listeBox_classes.insert(2, "No")
    elif select == 'Cellule sanguine':
        set_bd_CelluleS()
        listeBox_classes.delete(0,999)
        listeBox_classes.insert(1, "EOSINOPHIL")
        listeBox_classes.insert(2, "LYMPHOCYTE")
        listeBox_classes.insert(3, "MONOCYTE")
        listeBox_classes.insert(4, "NEUTROPHIL")
    elif select == 'Fleure':
        set_bd_flowers()
        listeBox_classes.delete(0,999)
        with open('json/flower_to_name.json', 'r') as f:
            flower_to_name = json.load(f)
        i = 1
        while i <103 : 
            listeBox_classes.insert(i, flower_to_name[str(i)])
            i = i + 1

#  Fonction qui definit le chemin du repertoire dans lequel l'image importer va etre enregistrée
f_path = ""
def clic2(evt):
    j = listeBox_classes.curselection()
    global f_path
    f_path = "data/"+bd_name+"/train/"+str(j[0]+1)
    print (f_path)

def importation(): 
    if f_path == "" :
        message_to_user['text'] = "Vous n'avez pas sélectionné de classe."
    else : 
        file_imp = '{}'.format(filedialog.askopenfilename(title="Ouvrir une image",filetypes=[('jpg files','.jpg'),('bmp files','.bmp'),('all files','.*')])) 
        print(file_imp)
        shutil.copy(file_imp,f_path)
        message_to_user['text'] = "Image bien importé. Merci !"
    
 

# création de la liste déroulante de gauche
listeBD = ["Melanome", "OCT", "Tumeur Cérébrale", "Cellule sanguine"] #, "Fleure"]  
liste_deroulante_e = ttk.Combobox(selecteur_e, values = listeBD)
liste_deroulante_e.current(0)
liste_deroulante_e.bind("<<ComboboxSelected>>", select_bd_train)
liste_deroulante_e.grid(row=0, column=0, sticky='ns')

#code pour la scrollbar
yDefilB = Scrollbar(selecteur_e, orient='vertical')
yDefilB.grid(row=0, column=2, sticky='ns')
listeBox_classes = Listbox(selecteur_e, yscrollcommand=yDefilB.set)
listeBox_classes.bind('<ButtonRelease-1>', clic2)
listeBox_classes.grid(row=0, column=1, sticky='ns')
yDefilB['command'] = listeBox_classes.yview
#code pour la scrollbar


    #validation
cadre_train = Frame(entrainement, width=100, height=100, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=0)
cadre_train.pack(pady=(90,0))

importer = Button(cadre_imp, text ="Importer", cursor="hand2", font =ar18, bg="red", command=importation)
importer.pack(pady=0)

bal1 = tix.Balloon()
bal1.bind_widget(importer, msg = "Importer une image permet de fournir une image supplémentaire\n à une base de donnée lorsque vous connaissez sa classe.\n Avoir une plus grande base de donnée permet d'obtenir \n des résultats plus précis lors de l'analyse (après entrainement).")

def call_code_train ():
    if densenet == False : 
        ic.code(bd_name, json_name, nb_epoch = nb_epoch, train = True)
    else : 
        ic.code(bd_name, json_name, nb_epoch = nb_epoch, train = True, use_model="densenet")


############# EPOCH SELECTOR
epoch_label = Label(cadre_train, text = "Séléctionnez le nombres d'epoch :", font = ar18)
epoch_label.pack()

epoch_select = Frame(cadre_train, width=100, height=100)
epoch_select.pack(pady= 10)
nb_epoch = 8;
def incr():
    global nb_epoch
    nb_epoch = nb_epoch + 1
    aff_nb_epoch['text'] = str(nb_epoch)

def decr():
    global nb_epoch
    if nb_epoch > 1 : 
        nb_epoch = nb_epoch - 1
    aff_nb_epoch['text'] = str(nb_epoch)

aff_nb_epoch = Label(epoch_select, text = "8", font = ar18)
aff_nb_epoch.place(x = 10, y = 10, height=80, width=40)

plus = Button(epoch_select, text="+", command=incr)
plus.place(x = 51, y = 10, height=40, width=40)

moins = Button(epoch_select, text="-", command=decr)
moins.place(x = 51, y = 51, height=40, width=40)

#############

entrainer = Button(cadre_train, text ="Entrainer", cursor="hand2", font =ar18, bg="red", command = call_code_train)
entrainer.pack()

bal2 = tix.Balloon()
bal2.bind_widget(entrainer, msg = "1°/ Choississez la base de donnée à entrainer ci-dessus.\n 2°/ Choississez le nombre d'époque. \n 3°/ Lancez l'entrainement en cliquant sur \"Entrainer\".")
############ PROGRESS BAR : TODO



############

##############################################################################################
                                # Analyser contenu
##############################################################################################

analyse_label = Label(analyse, text="Analyser :", font = ar36b)
analyse_label.pack(side = TOP)

desc_analyse = Label(analyse, text="Mon image est une : ", pady = 30, font = ar18)
desc_analyse.pack()

    #Selecteur

selecteur_a = Frame(analyse, width=100, height=100)
selecteur_a.pack()

def select_bd(event):
    select = liste_deroulante_a.get()
    print("Vous avez selectionné :", select)
    if select == 'OCT':
        set_bd_oct()
    elif select == 'Melanome':
        set_bd_mel()
    elif select == "Tumeur Cérébrale":
        set_bd_tumeurC()
    elif select == 'Cellule sanguine':
        set_bd_CelluleS()
    elif select == 'Fleure':
        set_bd_flowers()


labelChoix = Label(selecteur_a, text = "Choississez votre type d'image : ")
labelChoix.pack()
# création de la liste déroulante 
liste_deroulante_a = ttk.Combobox(selecteur_a, values = listeBD)
liste_deroulante_a.current(0)
liste_deroulante_a.bind("<<ComboboxSelected>>", select_bd)
liste_deroulante_a.pack()



    #Parcour
parcour = Frame(analyse, width=100, height=100, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=1)
parcour.pack(pady = 30)
filename = ''   ## chemin du fichier a analyser

def fileDialog():
    #filename = filedialog.askopenfilename(initialdir="/", title ="Choississez un fichier", filetype = (("jpeg", "*.jpg"), ("All Files", "*.*")))
    global filename
    filename = '{}'.format(filedialog.askopenfilename(title="Ouvrir une image",filetypes=[('jpg files','.jpg'),('bmp files','.bmp'),('all files','.*')])) 
    print(filename)
    size = len(filename)
    chemin_fichier.delete('1.0', END)
    chemin_fichier.insert(INSERT, filename)

    load = Image.open(filename)                                 
    width, height = load.size                                   
    x, y = calcul_resize(width, height)                        
    load = load.resize((int(x), int(y)), Image.ANTIALIAS)  
    print(load.size)
    render = ImageTk.PhotoImage(load)                           
    img['image'] = render
    img.image = render

def calcul_resize(x, y):
    #maxi = max(x,y)
    if x/y > 16/9: #maxi == x:
        ratio = x/480
    else:
        ratio = y/270
    return x/ratio, y/ratio

chemin_fichier = Text(parcour, height = 2, width = 50)
chemin_fichier.pack(side = LEFT)
parcourir = Button(parcour, text = "Parcourir", command = fileDialog, cursor="hand2", font = ar18)
parcourir.pack(side = LEFT)


    #Apercu 

apercu = Frame(analyse, width=256, height=144)
apercu.pack()
img = Label(apercu)
img.pack()


    #Validation
valid_a = Frame(analyse, width=100, height=100, borderwidth=1,  highlightbackground="black", highlightcolor="black", highlightthickness=1)
valid_a.pack(pady = 30)

def call_code_analyse ():
    if filename == '' :
        message_to_user['text'] = "Il faut d'abord séléctionné une image. Cliquez sur parcourir."
    else :
        if densenet == False : 
            ic.code(bd_name, json_name, route = filename, analyze = True)
        else : 
            ic.code(bd_name, json_name, route = filename, analyze = True, use_model = "densenet")


analyser = Button(valid_a, text ="Analyser", cursor="hand2", font =ar18, bg="red", command=call_code_analyse)
analyser.pack()

bal3 = tix.Balloon()
bal3.bind_widget(analyser, msg = "1°/ Choississez la base de donnée.\n 2°/ Importer votre image à analyser en cliquant sur \"Parcourir\". \n 3°/ Lancez l'analyse en cliquant sur \"Analyser\".")


##############################################################################################
                                # Statistiques contenu
##############################################################################################

statistiques_label = Label(statistiques, text="Statistiques :", font = ar36b)
statistiques_label.pack(side = TOP)

desc_stats = Label(statistiques, text="Statistiques sur les bases de données : ", pady = 30, font = ar18)
desc_stats.pack(padx= 30)

def load_image_stats(filename):
    load = Image.open(filename)                                 
    width, height = load.size                                   
    x, y = calcul_resize(width, height)                         
    load = load.resize((int(x), int(y)), Image.ANTIALIAS)       
    render = ImageTk.PhotoImage(load)
    return render

def afficher_graphiques(file1, file2):
    r = load_image_stats(file1)
    stats_image['image'] = r
    stats_image.image = r
    if file2 == None: 
        stats_image2.pack_forget()
    else : 
        r2 = load_image_stats(file2)
        stats_image2['image'] = r2
        stats_image2.image = r2
        stats_image2.pack(pady=20)

def aff_stats(event):
    select = liste_deroulante_s.get()
    if select == 'Générale':
        stats_label['text'] = "Statistiques Générales : "
        afficher_graphiques("img_stats/general1.png", "img_stats/general2.png")
    elif select == 'OCT':
        stats_label['text'] = "Statistiques sur les OCT : "
        afficher_graphiques("img_stats/oct_stats.png", None)
    elif select == 'Mélanome':
        stats_label['text'] = "Statistiques sur les Melanomes : "
        afficher_graphiques("img_stats/mel_stats.png", None)
    elif select == "Tumeur Cérébrale":
        stats_label['text'] = "Statistiques sur les Tumeurs cérébrale : "
        afficher_graphiques("img_stats/brain_stats.png", None)
    elif select == 'Cellule sanguine':
        stats_label['text'] = "Statistiques sur les Cellules sanguines : "
        afficher_graphiques("img_stats/blood_stats.png", None)
    elif select == 'Fleure':
        stats_label['text'] = "Statistiques sur les Fleures : "
        afficher_graphiques("img_stats/flower_stats.png", None)


listeBD_s = ["Générale", "Mélanome", "OCT", "Tumeur Cérébrale", "Cellule sanguine"]
liste_deroulante_s = ttk.Combobox(statistiques, values = listeBD_s)
liste_deroulante_s.current(0)
liste_deroulante_s.bind("<<ComboboxSelected>>", aff_stats)
liste_deroulante_s.pack()


stats_label = Label(statistiques, text="Statistiques Générales : ", font = ar12)
stats_label.pack(pady=30 )

stats_image = Label(statistiques)
r = load_image_stats("img_stats/general1.png")
stats_image['image'] = r
stats_image.image = r
stats_image.pack()
stats_image2 = Label(statistiques)
r2 = load_image_stats("img_stats/general2.png")
stats_image2['image'] = r2
stats_image2.image = r2
stats_image2.pack(pady = 20)





################################################################################
# Responsive 
################################################################################

interface_IC.grid_columnconfigure(index=(0,1), weight=1)
interface_IC.grid_rowconfigure(index=0, weight=1)



# On démarre la boucle Tkinter qui s'interompt quand on ferme la fenêtre
fenetre.mainloop()



