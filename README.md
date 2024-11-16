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

# Arborescence du projet : Détection de Signature

```plaintext
Racine du projet :
├── README.md                # Documentation principale du projet
├── requirements.txt         # Liste des bibliothèques Python nécessaires
├── run_detectSignature_tesseract.ipynb  # Notebook principal pour tester les fonctions

Dossiers :
├── data/                    
│   ├── template_bs_type1.pdf         # Modèle pour les documents BS
│   ├── template_eai_type1.pdf        # Premier modèle pour les documents EAI
│   ├── template_eai_type2.pdf        # Deuxième modèle pour les documents EAI
│   ├── template_eai_type3.pdf        # Troisième modèle pour les documents EAI
│   ├── template_lea.pdf              # Modèle pour les documents LEA
│   ├── template_qcfqr_type1.pdf      # Premier modèle pour les documents QCFQR
│   ├── template_qcfqr_type2.pdf      # Deuxième modèle pour les documents QCFQR
│   ├── EAI_pdfs_round2.txt           # Liste des PDF nécessitant un deuxième modèle

├── brouillons/                # Codes anciens (non utilisés dans la version finale)
│   ├── ancienne_version1.py
│   ├── ancienne_version2.py

├── utilities/                 # Modules Python développés
│   ├── poppler-23.01.0/       # Librairie nécessaire pour pdf2image
│   ├── set_global.py          # Définition des variables globales
│   ├── preprocessing.py       # Chargement, affichage et pré-traitement des images
│   ├── template.py            # Création d'un modèle PDF pour la détection de signature
│   ├── signatureDetect.py     # Détection des signatures sur des PDF analysés
│   ├── run_template.py        # Exécution de la création du modèle depuis le terminal
│   ├── run_detectSignature.py # Exécution de la détection de signature depuis le terminal
│   ├── run_retrieveErrorFiles.py  # Récupération des fichiers PDF non détectés
│   ├── signature_detect/      # Modules pour la détection des signatures
│       ├── __init__.py
│       ├── cropper.py
│       ├── judger.py

Fichiers générés par le projet :
├── template_shape.txt         # Dimensions du modèle PDF
├── template_text.txt          # Texte de la page contenant la signature
├── saved_template.pickle      # Positions des rectangles relatifs aux mots-clés
├── result.csv                 # Résultats finaux avec noms des fichiers et classification
