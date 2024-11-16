# Alternance à la Caisse d’Épargne Midi-Pyrénées : 

## Introduction

**Superviseurs** :
- Jerôme Martin
- Bertrand Michelot

Je tiens à remercier en particulir Mathieu LePajolec (https://www.linkedin.com/in/mathew-l/), data scientist prestataire, qui a travaillé quelques mois au sein de l'équipe pendant mon alternance. Ses conseils et ses encouragements ont été d'une grande aide.

Lors de mon contrat de professionnalisation d'un an à la CEMP (Caisse d’Épargne Midi-Pyrénées), j'ai mené en totale autonomie les trois projets suivants : 

- **Projet 1** : Développement d'une méthode innovante de détection de signatures sur PDF. Pour des raisons de sécurité, je n’ai pas pu utiliser de solutions existantes à l’exception de OpenCV et d’un OCR. Ma solution basée sur techniques de vision par ordinateur a atteint une précision proche de 100 % sur les données disponibles.
- **Projet 2** : Elaboration d'une méthode pour prédire le volume de crédits immobiliers, en utilisant des séries temporelles et des techniques de deep learning. Cette méthode a été particulièrement performante pour les données pré-COVID. 
- **Projet 3** : Automatisation de la détection de mots interdits (RGPD) en utilisant des méthodes de NLP, réduisant ainsi les besoins en révisions manuelles.

Une des principales difficultés de ces projets a été la politique de sécurité de la Caisse d'Épargne, qui limitait l'accès aux ressources de calcul (pas de GPU disponible et un simple ordinateur portable comme outil de travail) ainsi qu'à de nombreuses bibliothèques Python. Il a donc été nécessaire de faire preuve d'ingéniosité pour surmonter ces contraintes dans certains projets.


## Arborescence générale 

- `projet_back_office_credits`: projet 2
- `projet_detection_signature`: projet 1
- `projet_mots_interdits`: projet 3
- `rapport_et_oral_stage`: compte rendu de mon stage
- `Installation_poste_travail_banque.docx`: processus d'installation particulier de python sur le poste de travail en raison des pare-feux de l'entreprise.


### Projet 1 : projet_detection_signature

#### Objectif du projet

Le but de ce projet est de détecter la présence d’une signature sur plusieurs types de documents PDF.  
Pour davantage d’informations, se référer au rapport de stage *2022-23-stage5A-Roig-Lila_PFE.pdf*, aux présentations PowerPoint ou aux commentaires et descriptions des fonctions implémentées. Il est conseillé de lire le rapport de stage avant de reprendre le projet pour avoir une idée de la démarche globale.

---

#### Mise en route du projet

Pour faire fonctionner le projet, il faut au préalable installer Python sur sa machine (voir le *OneNote Data & Décisionnel Notebook* et la partie « A lire pour le prochain alternant »).  
Ensuite, créer un nouvel environnement conda spécifique à ce projet (par exemple intitulé `pdf`). Dans cet environnement, installer :
- **Jupyter Notebook** (voir le OneNote)
- Les librairies listées dans le fichier `requirements.txt` avec les versions indiquées.

Une fois cela fait, l’environnement de travail est prêt.

---

#### Description des répertoires et fichiers

##### `data/`
- Contient un petit nombre de fichiers PDF labellisés servant de données d’entrée pour les fonctions du projet. Ceci permet de tester les différents cas possibles avec un petit nombre de PDF.

##### `brouillons/`
- Contient les anciens codes ayant permis d’implémenter la solution finale. Ces codes sont conservés au cas où on souhaiterait y revenir, mais ils ne sont plus utiles et ce répertoire peut être supprimé.

##### `run_detectSignature_tesseract.ipynb`
- Un notebook permettant de savoir comment utiliser les fonctions implémentées dans le répertoire `utilities`.

##### `utilities/`
- Contient les fonctions Python `.py` développées dans le cadre du projet. Toutes les fonctions sont commentées et documentées. Pour savoir comment utiliser ces fichiers `.py`, exécuter le notebook `run_detectSignature_tesseract.ipynb`.

###### Fichiers présents dans `utilities/` :

- **`poppler-23.01.0/`** : Librairie installée dans le cadre de la configuration de la librairie Python `pdf2image` (voir `requirements.txt` pour l’installation). Ce répertoire a été ajouté pour contourner le pare-feu de l’entreprise.

- **`set_global.py`** : Définit les variables globales utilisées par les autres fichiers `.py`.

- **`preprocessing.py`** : Contient les fonctions pour charger, afficher et pré-traiter les images afin qu’elles soient mieux lisibles par un OCR (Tesseract). Ces fonctions sont utilisées par les modules `template.py` et `detectSignature.py`.

- **`template.py`** : Module permettant de définir un PDF modèle pour la détection de signature.  
  - L’utilisateur sélectionne un PDF modèle, saisit une liste de mots-clés `keyWords_template` présents sur la page contenant la signature, et dessine des rectangles autour des zones supposées contenir la signature.  
  - Les positions relatives des rectangles par rapport aux mots-clés sont calculées et stockées.  
  **Fichiers générés en sortie** :
  - `template_shape.txt`: Dimensions du PDF modèle.
  - `template_text.txt`: Texte de la page contenant la signature.
  - `saved_template.pickle`: Positions relatives des rectangles par rapport aux mots-clés.

- **`signatureDetect.py`** : Module permettant de vérifier la présence d’une signature sur un PDF à partir d’un modèle défini avec `template.py`.  
  - Ce module redimensionne le PDF à analyser aux dimensions du modèle, identifie la page contenant la signature, et place les rectangles en fonction des positions calculées.  
  - Il calcule le pourcentage de pixels continus dans les rectangles pour déterminer s’ils contiennent une signature.  
  **Fichier généré en sortie** :  
  - Un fichier `.csv` contenant les colonnes `file_name` (nom des PDF) et `result` (résultat de la classification : "signature", "no signature", ou "Page contenant la signature non détectée").

- **`signature_detect/`** : Répertoire contenant les fichiers `__init__.py`, `cropper.py` et `judger.py`. Ces fichiers détectent les signatures dans les rectangles en calculant le pourcentage de pixels continus.

- **`run_template.py`** : Permet d’exécuter le module `template.py` depuis le terminal.

- **`run_detectSignature.py`** : Permet d’exécuter le module `signatureDetect.py` depuis le terminal.

- **`run_retrieveErrorFiles.py`** : Permet de récupérer les PDF pour lesquels le résultat est "Page contenant la signature non détectée" afin de les soumettre à un autre modèle.

---

#### Autres fichiers

- **`EAI_pdfs_round2.txt`** : Fichier généré par `run_retrieveErrorFiles.py`, contenant les noms des PDF nécessitant un deuxième modèle.

- **Mots-clés** :  
  - `keyWords_template_bs.txt`, `keyWords_template_eai.txt`, `keyWords_template_lea.txt`, `keyWords_template_qcfqr.txt`: Contiennent les mots-clés pour chaque type de document PDF (BS, EAI, LEA, QCFQR).

- **Modèles PDF** :  
  - `template_bs_type1.pdf`: Modèle pour les documents BS.  
  - `template_eai_type1.pdf`, `template_eai_type2.pdf`, `template_eai_type3.pdf`: Modèles pour les documents EAI.  
  - `template_lea.pdf`: Modèle pour les documents LEA.  
  - `template_qcfqr_type1.pdf`, `template_qcfqr_type2.pdf`: Modèles pour les documents QCFQR.

---
