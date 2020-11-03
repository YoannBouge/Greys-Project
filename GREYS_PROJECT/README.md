# Grey's Project

Grey's Project est une application de reconnaissance d'imagerie médicale. Basée sur un réseau de neurones, qui fonctionne via un apprentissage supervisé. L'application permet de différencier un cas pathologique d'un cas sain. Dans le cas pathologique, l'application va prédire un type de pathologie en particulier si elle l'a déjà rencontré.

L'application fonctionne sur les bases de données suivantes : 
* Tomographies en cohérence optique (OCT)
* Images par résonance magnétique du cerveau (IRM)
* Photographies dermatoscopiques 
* Echantillons sanguins


## Pour commencer

Créez-vous un dépot git, pour ce faire : 

Créez un nouveau répertoire, ouvrez un terminal dans ce repertoire et utilisez la commande ``git clone`` suivit de l'URL HTTPS du projet. 
L'URL se trouve en cliquant sur le bouton bleu _Clone_ en haut à droite de cette page. 

### Pré-requis

Téléchargez le navigateur Anaconda (Python 3.7 version) : [Anaconda](https://www.anaconda.com/distribution/) 

Vous pourrez utiliser l'éditeur que vous souhaitez, mais de notre coté nous utilisons [VisualStudio Code](https://code.visualstudio.com/).

Si vous souhaitez lancer un entrainement, vous aurez besoins des bases de données. Les liens sont dans les dossiers respectifs dans le repertoire data. 
Si vous souhaitez uniquement faire des analyses d'image vous devez télécharger le checkpoint associé. Les liens des checkpoints de chaques bases de données se trouvent dans le dossier checkpoints.  

### Installation

Installation de l'environnement de travail... 
Nous allons installer toutes les librairies et modules dont nous aurons besoins. 

Ouvrez le prompt Anaconda et tapez les commandes suivantes : 

- ``conda create -N myenv python = 3.7 anaconda`` Creer un environnement conda. 
- ``conda activate myenv`` Active l'environnement précédemment créé. 
- ``conda install -c anaconda numpy`` Installe la librairie _numpy_
- ``pip install setuptools`` Installe la librairie _setupstools_
- ``conda install -c conda-forge matplotlib`` Installe la librairie _matplotlib_
- ``conda install -c conda-forge opencv`` Installe la librairie _opencv_

Vous pouvez vérifier que vous avez tout les modules en tapant ``conda list``.

Il ne reste plus qu'à installer Pytorch. Pour ce faire, suivez ce lien : [Pytorch](https://pytorch.org/get-started/locally/).
Choississez les paramètres adaptés à votre machine, puis tapez la commande que le site vous propose dans votre prompt Anaconda. 
Nous vous conseillons **Stable (1.4)** pour Pytorch Build, **Pip** pour Package et **Python** comme Langage.  


## Démarrage

Une fois votre environnement de travail prêt, vous pouvez commencer. 
Ouvrez le navigateur **Anaconda** et placez vous dans l'environnement de travail précédemment créé. Lancez votre éditeur depuis Anaconda. 
Placez-vous dans le dossier **greys/GREYS_PROJECT**, ouvrez un terminal et activez votre environnement : ``conda activate myenv``.
Pour démarrer l'application, vous pouvez lancer le fichier **interface_ImgClassifier.py** avec la commande suivante : 
- ``python ./interface_Imgclassifier.py``

Voila ce qui devrait s'afficher à l'écran : 

![Screenshot application Greys](https://drive.google.com/open?id=1qFojj_rPtSn1j36NG-kH8pGNeZP8EA5q)

## Auteurs

- **Doriane AJMI** - [GitHub](https://forge.univ-lyon1.fr/p1616047) 
- **El Mehdi MAKHLOUF MOHAMED** - [GitHub](https://forge.univ-lyon1.fr/p1808420)
- **Yoann BOUGE** - [GitHub](https://forge.univ-lyon1.fr/p1711842)




