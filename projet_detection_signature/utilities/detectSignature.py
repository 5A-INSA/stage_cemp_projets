"""

Ce module permet de verifier si un pdf est signé ou non. 
Au préalable, l'utilisateur a du définir un template avec le module template.
Ainsi, la liste de mots clés keyWords_template ainsi que la variable 
list_relative_points ont été calculés avec le module template.  
Pour rappel, list_relative_points représente les coordonnées relatives des rectangles par 
rapport aux mots de keyWords_template sur le pdf template. 

Ce module utilise ensuite le texte template_text renvoyé par le module template
et l'OCR Tesseract pour lire les pages du pdf à analyser et trouver la page 
dont le texte est le plus similaire à la page du pdf template. La page trouvée 
est censée être la page contenant la signature.  

Une fois la page du pdf contenant la signature identifiée, ce module extrait les zones du pdf à analyser 
supposées contenir la signature. 

Pour extraire ces zones, l'algorithme utilise la variable list_relative_points
pour obtenir les coordonnées des rectangles à extraire, en fonction de la position
des mots de keyWords_template du pdf à analyser.
Le fait de calculer la position relative des mots de keyWords_template par rapport 
aux rectangles permet d'ajuster la position des rectangles à chaque pdf à analyser.

Enfin, une fois les zones supposées contenir la signature extraites, un pourcentage de 
pixels continus est calculé pour déterminer si le pourcentage de remplissage de la zone
est suffisant. Si le remplissage est suffisant sur l'une des zones extraites, 
le pdf est considéré comme signé. 

Contient les fonctions : 
- compute_strings_similarity()
- find_page_with_Tesseract()
- find_expression_in_list()
- find_1_keyWord_coord()
- extract_signature_boxes()
- verify_signature()
- check_signature_num()
- drawProgressBar()
- main_detectSignature()
"""


#***********************************************************
# Chargement des librairies
#***********************************************************
from set_global import PATH_TESSERACT, PATH_POPPLER
from preprocessing import *

from signature_detect.cropper import Cropper
from signature_detect.judger import Judger

#---- Système
import os  #se déplacer dans les dossiers
import sys #affichage barre progression
import time #mesure du temps d'exécution
from datetime import datetime #date et heure du jour
#---- Calculs 
import numpy as np
import pandas as pd
#---- Traitement des PDFs
import cv2 #traitement d'image (jpg)
from PIL import Image
import fitz
#----- Détection de texte avec Tesseract                                                                                 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = PATH_TESSERACT # Définit le chemin de tesseract
import re
import unicodedata #retirer les accents d'un texte
#----- Sauvegarde 
import pickle 
import csv
from pathlib import Path

#***********************************************************
# Extraction de la page contenant la signature à l'aide de la liste de mots clés keyWords_page
#***********************************************************
def compute_strings_similarity(s1,s2):
    """
    Cette fonction permet de dire si deux chaînes de caractères 
    sont similaires. 
    
    Input:
    -----
    - s1, s2 (strings) : chaînes de caractères dont on veut 
      comparer la similarité
   
    Output:
    -------
    - similarity (float) : float entre 0 et 1 représentant la similarité de s1 et s2.
      Une valeur de 0 indique des documents très différents tandis qu'une valeur 
      de 1 indique des documents identiques.
    
    ================================== Notes ==================================
    La similarité ou indice de Jaccard, mesure la similarité entre deux ensembles. 
    Il est défini comme le rapport entre la taille de l'intersection des ensembles et 
    la taille de l'union des ensembles. En d'autres termes, il s'agit de la proportion 
    d'éléments communs entre deux ensembles.

    l'indice de Jaccard est particulièrement utile lorsque la présence ou l'absence d'éléments
    dans les ensembles est plus importante que leur fréquence ou leur ordre. 
    Par exemple, il peut être utilisé pour comparer la similarité de deux documents en considérant
    les ensembles de mots qui apparaissent dans chaque document.

    L'indice de Jaccard est calculé comme suit :
    J(A,B) = |A ∩ B| / |A ∪ B|

    où A et B sont des ensembles, et |A| et |B| représentent la cardinalité ou la taille des ensembles.

    L'indice de Jaccard est compris entre 0 et 1, où 0 indique qu'il n'y a pas d'éléments communs entre
    les ensembles, et 1 indique que les ensembles sont identiques.

    Dans notre cas, l'indice de Jacquart semble pertinent car l'OCR ne lit pas toujours les mots dans 
    le même sens selon les documents et nous souhaitons juste regarder si les deux textes présentent 
    les mêmes mots sans se soucier de leur position dans le texte.
    Au contraire des distances comme levenshtein (qui calcule la différence entre deux chaînes de caractères
    en comptant le nombre minimum d'insertions, de suppressions ou de substitutions d'un seul caractère nécessaires
    pour transformer une chaîne en une autre) ou la métrique SequenceMatcher de la librairie difflib (qui se base
    sur l'algorithme LCS consistant à trouver la plus longue sous-séquence "logique" présente dans les deux chaînes) 
    pour lesquelles la position des mots importe.
    ===========================================================================
    
    References:
    ----------
    - différentes méthodes pour calculer la similarité entre 2 textes:
     https://spotintelligence.com/2022/12/19/text-similarity-python/
    - python code pour jaccard-similarity
      https://studymachinelearning.com/jaccard-similarity-text-similarity-metric-in-nlp/
    """  
    # Retire les accents
    s1 = ''.join(c for c in unicodedata.normalize('NFD', s1) if unicodedata.category(c) != 'Mn')
    s2 = ''.join(c for c in unicodedata.normalize('NFD', s2) if unicodedata.category(c) != 'Mn')
    # Liste des mots uniques dans un document
    words_s1 = set(s1.lower().split())
    words_s2 = set(s2.lower().split())
    
    # Trouve l'intersection des listes de mots de s1 & s2
    intersection = words_s1.intersection(words_s2)

    # Trouve l'union des mots de la liste s1 & s2
    union = words_s1.union(words_s2)
        
    # Calcule le score de similarité de Jaccard 
    # en utilisant la longueur de l'ensemble d'intersection divisée
    # par la longueur de l'ensemble d'union. Float compris entre 0 et 1.
    similarity = float(len(intersection)) / len(union)
    
    return similarity


def find_page_with_Tesseract(pdf,template_text,similarity_thresh=0.5,template_shape=(2338,1654),
                             start_with="end",verbose:bool=False):
    """
    Cette fonction permet de renvoyer la page du pdf dont le texte est le plus similaire
    à celui de template_text (= texte de la page template.)
    
    Si start_with="end", la recherche se fait en partant de la dernière page du pdf 
    et en remontant j'usqu'à la première page car la signature se trouve généralement
    à la fin du pdf. Si start_with="begin", la recherche se fait en partant de la 
    première page du pdf. Dès que la page est trouvée, la recherche s'arrête. 
    Avant de lire la page avec tesseract, une rotation de la page et un
    débruitage + seuillage de l'image sont appliqués dans le but d'améliorer
    la lecture.     
    
    Rq: Avant de calculer la position des mots, l'image à analyser est mise à la même
    échelle que l'image template. Ceci s'avère utile par la suite pour le calcul des 
    coordonées dans le cas où l'image à analyser n'aurait pas la même taille que l'image 
    template.
    
    >>> Utilise les fonctions get_angle_openCV(), get_angle_tesseract(), 
        rotate_im(), denoise_image() et compute_strings_similarity().
        Utilise la fonction show_image() si verbose=True.
        Pour davantage d'information, consulter la description de ces fonctions. 
    
     Input:
    ------
    - pdf (liste de PIL.PpmImagePlugin.PpmImageFile): ensemble des pages du pdf
    - template_text (string): texte contenu dans la page du pdf template;
      préalablement calculé avec la fonction read_page().
    - similarity_thresh (float) : seuil de similarité entre 2 textes. 
      Si la similarité calculée est en dessous de ce seuil, alors les deux textes ne sont
      pas similaires.
    - template_shape (tuple) : (height,width) ou (height,width,depth) taille de l'image
      ayant servi de template. Avant de calculer la position des mots, l'image à analyser
      est mise à la même échelle que l'image template. 
    - start_with (string) : peut prendre les valeurs "begin" et "end" et permet d'indiquer
      si on commence la recherche de la page contenant la signature en partant du début du 
      pdf (avec start_with="begin") ou de la fin du pdf (avec start_with="end"). 
    - verbose (bool): si True, affiche l'image aux différentes étapes de l'algorithme.
    
    Output:
    ------
    - final_im (np.array): image contenant au moins un des mots   
      clés de keyWords.    
    - image_info (dict): dictionnaire renvoyé par pytesseract.image_to_data()
      contenant toute l'information de la page. Ceci permet de réutiliser 
      image_info dans d'autres fonctions, sans avoir besoin de relire la page
      tesseract.
    - warning (bool): si True, aucun mot clé n'a été trouvé dans le pdf. 
      La dernière page du pdf sera alors retournée par défaut.
      
    Référence:
    ---------
    - transformation d'une image du format PIL.Image au format np.array :
      https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
    """

    # ==========================================
    # Recherche de la page du pdf contenant au moins un des mots clés
    # ==========================================
    #si aucun mot clé n'est trouvé dans le texte, warning passera à True
    warning = False
    #boolean indiquant si on a trouvé ou non la page contenant la signature 
    found = False 
    
    # Pour être + rapide, on lit généralement les pages en partant de la fin (avec 
    # start_with="end") car la signature est généralement présente à la fin du pdf.
    # -------------------------------------
    if start_with=="end": #Si on commence la recherche de page par la fin du pdf
        i = len(pdf)-1 #compteur pour les pages du pdf.
        def condition(i): return i>-1 #fonction pour la boucle while
        increment = -1 #incrément
    elif start_with=="begin": #Si on commence la recherche de page par le début du pdf
        i = 0 #compteur pour les pages du pdf. 
        def condition(i): return i<len(pdf) #fonction pour la boucle while
        increment = 1 #incrément
    else:
        print("\nstart_with doit être égal à 'end' ou 'begin'")
    
    # I) Recherche des mots clés sur chaque page du pdf
    # -------------------------------------
    while not(found) and condition(i) :
        #page i du pdf
        im = pdf[i] 
        #Transformation de l'image du format PIL.Image au format np.array
        im = np.array(im)
        #Redimensionnement de l'image à la même taille que le template
        im = cv2.resize(im, (template_shape[1],template_shape[0]), interpolation = cv2.INTER_AREA)
        
        if verbose: #affichage de l'image si verbose=True
            print(">>> Fonction find_page_with_Tesseract() <<<")
            print("\nImage d'origine :")
            show_image(im)
        
        # 1) Rotation de l'image pour une meilleure lecture 
        # -------------------------------------
        # Etape 1 : rotation fine avec get_angle_openCV()
        angle1 = get_angle_openCV(im)
        if angle1 != 0:
            im = rotate_im(im,angle1)
            if verbose: #affichage de l'image si verbose=True
                print("Image après rotation openCV :")
                print("angle openCV :", angle1)
                show_image(im)
                
        # Etape 2 : rotation avec get_angle_tesseract() pour différencier entre 0° et 180°
        # si la page est blanche, tesseract lève une erreur. Dans ce cas, on ne lit pas la page.
        try:
            angle2 = get_angle_tesseract(im)
            if angle2 != 0:
                im = rotate_im(im,angle2)
                if verbose: #affichage de l'image si verbose=True
                    print("Image après rotation Tesseract :")
                    print("angle Tesseract :", angle2)
                    show_image(im)
        except pytesseract.TesseractError:
            pass
            
        # 2) Pre-processing de l'image pour améliorer la lecture
        # -------------------------------------
        im = denoise_image(im)
        if verbose: #affichage de l'image si verbose=True
            print("Image après pre-processing :")
            show_image(im)
        
        # 3) Extraction du texte de la page i avec tesseract
        # -------------------------------------
        results = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        txt = ' '.join(results['text'])
        
        #calcule la similarité entre le texte de la page i du pdf 
        #et le texte du pdf template text_template.
        similarity = compute_strings_similarity(s1=template_text,s2=txt)
        
        if verbose : print("Similarity :", similarity)

        #si les textes sont similaires
        if similarity > similarity_thresh:
            found = True #on met found à true
            final_im = im #on récupère la page contenant la signature 
            image_info = results #on récupère les informations de la page
        #sinon, on passe à la page suivante
        else:
            i+=increment

    #II) Si après recherche de toutes les pages, aucun mot clé n'a été trouvé,
    #    le document est considéré comme une anomalie
    # -------------------------------------
    if found == False: 
        warning = True #anomalie
        final_im = pdf[-1] #la dernière page du pdf est retournée par défault.
        image_info = pytesseract.image_to_data(final_im, output_type=pytesseract.Output.DICT)
        
    return final_im, image_info, warning 


#***********************************************************
# Recherche la première occurence d'un des mots de la liste 
# to_find = keyWords_template sur l'image et renvoie sa position.
#***********************************************************
def find_expression_in_list(list_of_strings,expression,special_char='#!#'):
    """
    Cette fonction permet de trouver la position d'une expression dans 
    une liste de strings où chaque élément de cette liste est un mot.
    
    Input:
    ------
    - list_of_strings (list de string) : liste de strings où chaque élément 
      de la liste correspond à un mot.
    - expression (string) : expression à trouver dans list_of_strings.
      Peut contenir plusieurs mots séparés par des espaces.
    - special_char (string) : caractère spécial dont on se sert pour 
      trouver l'indice. Attention, special_char ne doit PAS être un 
      caractère regex comme |, ?, ., ^, $
    
    Output:
    ------
    - idx (integer) : indice du début de la position de l'expression 
      dans la liste list_of_strings. Si aucune correspondance n'est
      trouvée, idx=None. 
    
    ========================== Notes ==========================
    Par exemple: Si list_of_strings = ["Voici", "une", "liste", "où",  
    "chaque", "élément", "est", "un", "mot"] et expression = ["chaque élément"]
    alors la fonction renvoie la valeur 5 qui correspond à la position
    du début de l'expression dans la liste list_of_strings.
    
    On retire d'abord tous les accents de list_of_strings et expression
    pour rendre la recherche + robuste.
    
    Pour ce faire, on fusionne list_of_strings et expression en un seul string 
    en séparant les espaces par special_char='#': 
    list_of_strings = "Voici#une#liste#où#chaque#élément#est#un#mot"
    expression = "chaque#élément"
    
    On recherche la première occurence de "chaque#élément" dans 
    list_of_strings avec regex. On obtient ainsi l'indice du premier
    élément ici 19 car le 'c' de "chaque#élément" est le caractère
    n°19 dans list_of_strings (les indices commencent à 0).
    
    Enfin, on regarde list_of_strings[0:19] et on compte le nombre
    de special_char ce qui nous donne l'indice du premier mot de expression
    dans la liste list_of_strings donc idx = 5.
    ===========================================================
    """

    # On fusionne la liste list_of_strings en un seul string en
    # séparant chaque élément de la liste par le caractère spécial    
    list_of_strings = special_char.join(list_of_strings)
    # On retire tous les accents pour rendre la recherche + robuste
    list_of_strings = ''.join(c for c in unicodedata.normalize('NFD', list_of_strings) if unicodedata.category(c) != 'Mn')
    
    # On remplace dans expression les espaces par le caractère spécial    
    expression = expression.replace(' ', special_char)  
    # On retire tous les accents pour rendre la recherche + robuste
    expression = ''.join(c for c in unicodedata.normalize('NFD', expression) if unicodedata.category(c) != 'Mn')

    # On recherche la première occurence de l'expression dans 
    # la liste de strings
    res = re.search(expression,list_of_strings)
    
    # si on a trouvé une correspondance:
    if res:
        # On compte le nombre de special_char du début de list_of_strings 
        # jusqu'à au premier élément de expression et on trouve donc 
        # l'indice du début de expression dans list_of_strings
        idx = list_of_strings[:res.start()].count(special_char)
    else:
        idx = None

    return idx


def find_1_keyWord_coord(im, to_find, image_info=None, debug=False, min_conf=80):
    """
    Cette fonction recherche la première occurence 
    d'un des mots de la liste to_find sur l'image et renvoie sa position sur l'image.
    Si un mot de la liste n'est pas trouvé, on passe à la recherche du mot suivant, 
    sinon la recherche  s'arrête. La recherche se fait dans l'ordre des mots de to_find.
    Cette fonction utilise l'OCR Tesseract. 
    >>> utilise la fonction find_expression_in_list() 
    
    Pour davantage de détails, voir les Notes plus bas.

    Input:
    ------
    - im (np.array): image sur laquelle recherche les coordonées des mots clés.
    - to_find (list of string): liste des mots clés à rechercher. La recherche se fait
      dans l'ordre des mots de to_find. Dès qu'un mot est trouvé, on ne recherche pas les 
      mots suivants. Attention à ne pas mettre d'espace à la fin et au début des mots ou des 
      expressions. Ex : ['TITULAIRE','REPRESENTANT LEGAL','BANQUE'] 
    - image_info (dict) : information contenue dans l'image obtenue avec la fonction 
      pytesseract.image_to_data(). Si image_info=None, alors on lit la page pour en extraire
      l'information. Si image_info!=None, on ne lit pas la page et on utilise directement l'information
      contenue dans image_info.
    - debug (bool): si True, l'ensemble du texte détecté est affiché avec son niveau de confiance.
      Si un mot clé a été trouvé, l'image avec le mot clé encadré est également affichée.
    - min_conf (int): niveau de confiance minimale pour l'affichage du texte trouvé dans le cas 
      où debug=True.

    Output:
    -------
    - key_point (tuple de 2 int): coordonnées (x,y) du mot trouvé. Vaut (0,0)
      si aucun mot clé n'est trouvé.
    - keyWord (string): mot trouvé. 
    - wargning (bool): si True, aucun des mots de la liste to_find n'a pu être 
      détecté sur l'image. 

    References:
    ----------
    - détection de l'emplacement du texte avec Tesseract:
    https://pyimagesearch.com/2020/05/25/tesseract-ocr-text-localization-and-detection/

    - interprétation de la fonction image_to_data:
    https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o

    ============================ Notes ============================
    Nous nous situons déjà sur la page contenant la signature.

    Nous cherchons à construire 2 cadres pour lesquels nous analyserons leur remplissage afin de déterminer
    si le pdf est signé ou non. Le 1er cadre concernera la signature du titulaire, si le 1er cadre n'est pas 
    rempli, nous irons voir le 2e cadre concernant la signature du représentant légal.

    Avec tesseract, on obtient une liste des mots trouvés dans le texte avec leur position correspondante.
    Les cadres seront donc déterminés en fonction de la position des mots clés sur la page. 

    Ainsi, nous contruisons une liste to_find = ['TITULAIRE','REPRESENTANT','BANQUE'] qui contient les mots
    clés que nous recherchons sur la page. A la première étape, l'algo cherche le premier mot clé 'TITULAIRE'. 
    Si ce mot est présent sur la page, on construit les 2 cadres en fonction des coordonnées de ce mot. 
    Si ce mot n'est pas trouvé, on passe au mot suivant 'REPRESENTANT' et on repète la procédure jusqu'à
    ce qu'un mot soit trouvé. 

    Cette méthode en plusieurs étapes permet davantage de robustesse dans le cas où l'OCR ne parviendrait
    pas à lire un des mots. 

    Rq: Normalement, l'OCR lit le texte ligne par ligne du haut vers le bas mais pas toujours !
    ==============================================================
    """

    #boolean pour dire si on a trouvé ou non un mot clé
    found = False
    #si on ne trouve aucun mot clé on a un warning
    warning = True
    #initialisation
    keyWord = ''
    #si ces valeurs ne sont pas mises à jour, aucun mot clé n'a été trouvé.
    key_point = (0,0)

    image = im.copy() 

    # ==============================================
    # Extraction des informations de l'image
    # ==============================================
    #extrcation de toutes les informations de l'image à l'aide de Tesseract
    if image_info: #si on a déjà récupéré au préalable les infos de la page 
        #liste de tous les mots trouvés par Tesseract
        Text = image_info["text"]
    else: #sinon, on lit la page pour en récupérer les infos
        image_info = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        #liste de tous les mots trouvés par Tesseract
        Text = image_info["text"] 

    # ==============================================
    # L'algo cherche si le mot to_find[0] est présent dans le texte.
    # Si oui, on renvoie la position du mot to_find[0], 
    # sinon on passe au mot suivant fo_find[1] etc.
    # ==============================================
    i = 0 #compteur

    while not found and i<len(to_find) : #si i>=len(to_find), on aura un pb pour accéder à to_find[i]
        keyWord = to_find[i] 

        #indice du mot to_find[i] dans la liste Text
        # Méthode 1 avec np.where. Ne trouve que le mot exactement identique à 'TITULAIRE' par ex.
        #idx = np.where(np.array(Text) == keyWord)[0] 
        # Méthode 2 avec Regex. Peut trouver 'TITULAIRE' dans 'DUTITULAIREE'
        #idx = [j for j, item in enumerate(Text) if re.search(keyWord, item)]
        #-- Méthode 3 avec find_expression_in_list
        idx = find_expression_in_list(Text,keyWord)

        if idx!=None: #si on a trouvé le mot clé (ie idx n'est pas vide)
            found = True #on a trouvé un mot clé
            warning = False #on a pas de warning
            #idx = idx[0] 
            # (x,y) = positions du mot to_find[i]
            # le point d'origine (0,0) se trouve en haut à gauche du document 
            x = image_info["left"][idx] #abscisse du mot (point en haut à gauche)
            y = image_info["top"][idx] #ordonnée du mot (point en haut à gauche)
            key_point = (x,y)       
            
        else: #sinon, on passe à la recherche du mot suivant
            i+=1

    # ==============================================
    # Dans le cas où on a trouvé un mot,
    # on peut afficher le mot trouvé sur l'image pour debugging
    # ==============================================
    if debug : 
        print(">>> Fonction find_1_keyWord_coord() <<<")
        debug_text = [] #liste contenant tous les mots trouvés sur la page

        # loop over each of the individual text localizations
        for j in range(0, len(image_info["text"])):
            # extract the OCR text itself along with the confidence of the
            # text localization
            text = image_info["text"][j] #recognized text string
            conf = int(image_info["conf"][j]) #confidence of the detected text localization.

            # filter out weak confidence text localizations
            if conf > min_conf :
                # display the confidence and text to our terminal
                debug_text.append(text + " (conf:" + str(conf) + ")")         

        print("\n----- Texte trouvé avec un niveau de confiance >= " + str(min_conf) + " : -----")
        print(debug_text)
        
        # Affichage des mots trouvés
        if found:
            w = image_info["width"][idx] #hauteur du rectangle contenant le mot
            h = image_info["height"][idx] #largueur du rectangle contenant le mot
            # draw a bounding box around the text
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # draw the text 
            stripped_keyWord = ''.join(c for c in unicodedata.normalize('NFD', keyWord) if unicodedata.category(c) != 'Mn')
            cv2.putText(image, stripped_keyWord, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # display image
        print("\n----- Mot détecté sur l'image : -----")
        show_image(image)      
    # ==============================================

    return key_point, keyWord, warning



#***********************************************************
# Renvoie les zones supposées contenir la signature 
#***********************************************************
def extract_signature_boxes(im,key_point,keyWord,list_relative_points,debug:bool=False):
    """
    Cette fonction renvoie les images contenues dans les rectangles 
    dessinés au préalable par la fonction create_template().
    L'un des rectangles est supposé contenir une signature.
    
    Input:
    ------
    - im (np.array) : image contenant la signature
    - key_point (tuple) : coordonées (x,y) du mot clé renvoyé par la fonction 
      find_1_keyWord_coord()
    - keyWord (string) : mot clé renvoyé par la fonction 
      find_1_keyWord_coord()
    - list_relative_points (list de dict) : liste de dictionnaires où chaque dictionnaire
      représente les coordonnées d'un rectangle par rapport aux différents mots clés. 
      list_relative_points est renvoyé par la fonction compute_relative_coord().
    - debug (bool): si debug = True, les zones extraites sont affichées.
    
    Output:
    ------
    - list_im_boxes (list of np.array : liste d'images correspondant au contenu des différents
      rectangles. 
    """ 
    # liste contenant les images extraites. 
    list_im_boxes = [] 

    #parcourt des rectangles
    for i in range(len(list_relative_points)):
        # coordonées relatives du rectangle
        xr_relatif,yr_relatif,wr,hr = list_relative_points[i][keyWord]
        # coordonées du mot clé
        x,y = key_point

        # création de la boite : extrcation de l'image 
        im_box = im[y+yr_relatif:y+yr_relatif+hr, x+xr_relatif:x+xr_relatif+wr]
        # ajout de l'image à la liste d'images
        list_im_boxes.append(im_box)
        
    if debug: 
        print(">>> extract_signature_boxes() <<<")
        print("\n Zones de l'image extraites :")
        for im_box in list_im_boxes:
              show_image(im_box)        
        
    return list_im_boxes


#***********************************************************
#  Vérifie si les zones extraites contiennent bien une signature
#***********************************************************
def verify_signature(im_bin,
                    cropper_params = {"min_region_size":1e3,"border_ratio":0.01},
                    judger_params = {"pixel_ratio": [0.01,1]},
                    verbose:bool=False):
    
    """
    Cette fonction permet de regrouper toutes les étapes nécessaires 
    pour détecter la signature sur un pdf scanné à l'aide de la librairie 
    signature-detect. L'image doit au préalable avoir été binarisée. 
    >>> Utilise la fonction show_image()
    
    Input:
    -----
    - im_bin (np.array): image binaire sur laquelle détecter la signature. 
      L'image devrait avoir été pre-processed auparavant.
    - cropper_params (dict): cropper_params = {"min_region_size":1e3,"border_ratio":0.01}
      Contient les paramètres du Cropper.
    - judger_params (dict): judger_params = {"pixel_ratio": [0.01,1]}
      Contient les paramètres du Judger.
    - verbose (bool): si True, la description de l'Extractor, Cropper et Judger sont 
      affichés ainsi que des images. Affiche de plus les zones de pixels connectés trouvés 
      pour visualiser ce qu'il se passe.
    
    Ouput:
    ------
    - is_signed (bool): si True, le document est signé.
    - signature (np.array): image de la signature extraite obtenue 
      après le Cropper. Si aucune signature n'a pu être détectée, signature=None. 
    
    ================================ Notes ================================
    La package signature-detect contient plusieurs modules résumés brièvement ici : 
    
    LOADER:
    Charge le pdf, le transforme en image et applique un pre-processing
    sur l'image pour obtenir une image binaire. 
    On n'utilise donc pas le Loader de signature-detect. On met en entrée
    une image déjà pre-processed par un équivalent du Loader que nous avons
    implémenté au préalable.

    EXTRACTOR:
    L'extractor enlève de l'image binaire les éléments qui sont trop grand ou trop petits pour 
    correspondre à une signature. Comme nous avons déjà extrait la portion de l'image supposée
    contenir la signature, nous n'utiliserons pas l'extractor. 

    CROPPER:
    Le cropper découpe l'image binaire selon des régions de pixels connectés. 
    Cette fonction est utile dans le cas où par exemple une image n'est pas 
    signée mais contient un bout de cadre. Si on utilise pas le cropper,
    le judger détectera que le bout de cadre est une signature. 

    JUDGER:
    Le judger lit les régions données par le cropper et applique plusieurs 
    critères de taille et de remplissage de la région pour déterminer si celle-ci 
    contient une signature.    
    =======================================================================
    
    Référence:
    ---------
    - tuto package signature-detect:
      https://github.com/EnzoSeason/signature_detection/blob/main/demo.ipynb
    """
    im = im_bin.copy()

    # =====================================
    # 1. CROPPER
    # =====================================
    # définition du Cropper
    cropper = Cropper(min_region_size=cropper_params["min_region_size"],
                      border_ratio= cropper_params["border_ratio"])
    # découpage de l'image selon des régions de pixels connectés
    results = cropper.run(im)

    # affichage si verbose=True
    if verbose: 
        print(">>> verify_signature() <<<")
        print(cropper)

    # =====================================
    # 2. JUDGER
    # =====================================
    # définition du Judger
    judger = Judger(pixel_ratio = judger_params["pixel_ratio"],
                    debug = verbose)

    #----- si results contient une valeur, il peut y avoir une signature
    if len(results)>0:
        # la première valeur de results est la région de pixels
        # connectés la plus grande et est supposée correspondre
        # à la signature
        signature = results[0]["cropped_mask"]
        # le judger détermine si il y a signature 
        is_signed = judger.judge(signature)

        # affichage si verbose=True
        if verbose:
            print(">>> Image après Cropper:")
            show_image(signature)

    #----- si results ne contient aucune valeur, il n'y a pas de signature    
    else:
        is_signed = False
        signature = None

    # affichage si verbose=True
    if verbose: 
        print(judger)
        print(">>> Document signé:", is_signed)

    return is_signed, signature


'''
# Version de verify_signature sans le cropper
def verify_signature(im_bin,pixel_ratio,debug:bool=False):
    """
    Cette fonction détecte la présence d'une signature sur une image.
    L'image doit au préalable avoir été binarisée i.e. contenir 
    uniquement 2 valeurs 0 et 255. 
    >>> Utilise la fonction show_image()

    Input:
    -----
    - im_bin (np.array): image binaire sur laquelle détecter la signature. 
      L'image devrait avoir été pré-traitée auparavant (obligatoirement binarisée, et 
      potentiellement débruitée, seuillée...).
    - pixel_ratio (tuple = (low,high)): tuple de 2 floats. 
      Si low < le nombre de 0 / le nombre de 255 < high.
    - debug (bool): si True, le % de remplissage de l'image (= compute_pixel_ratio)
      est affiché.

    Ouput:
    ------
    - is_signed (bool): si True, le document est signé.
    """
    
    im = im_bin.copy()
    
    
    # Véficiation que le l'image binaire est valide 
    # i.e. elle est ni toute noire (que des 0) ni toute blanche (que des 1)
    # -----------------------------
    is_valid_mask = True

    values = np.unique(im)
    if len(values) != 2:
        is_valid_mask = False
    if values[0] != 0 or values[1] != 255:
        is_valid_mask = False
        
    # Si l'image est valide, on calcule le % de remplissage de l'image 
    # pour déterminer si le document est signé ou non
    # -----------------------------    
    if is_valid_mask: #si l'image est valide 
        bincounts = np.bincount(im.ravel())
        compute_pixel_ratio = bincounts[0] / bincounts[255] 
        #bincounts = count number of occurrences of each value in array of non-negative ints.            
        # si le % de remplissage de pixels est en dehors des seuils, le document n'est pas signé
        if compute_pixel_ratio < pixel_ratio[0] or compute_pixel_ratio > pixel_ratio[1]:
            is_signed = False
        # sinon, le document est signé
        else: 
            is_signed = True
    else: #si l'image n'est pas valide, le document n'est pas signé
        is_signed = False  
        compute_pixel_ratio = "-"
        
    # Affichage si debugging
    # ----------------------------- 
    if debug:
        print("image sur laquelle on calcule la ratio:")
        show_image(im)
        print("computed pixel_ratio: ",compute_pixel_ratio)
        print("image signée :", is_signed)
    return is_signed
'''

#***********************************************************
# Vérifie si on est en présence d'une signature numérique
#***********************************************************
def check_signature_num(path_pdf):
    """
    Cette fonction permet de détecter si un document pdf numérique
    a été signé. 
    
    Input:
    -----
    - path_pdf (strig): chemin absolu vers le document pdf à vérifier.

    Ouput:
    ------
    - is_signed (bool): si True, le document a été signé.
    
    ================================ Notes ================================
    function get_sigflags(): PDF only: Return whether the document contains 
    signature fields. This is an optional PDF property: if not present (return value -1),
    no conclusions can be drawn – the PDF creator may just not have bothered using it.
    
    Returns
    -1: not a Form PDF / no signature fields recorded / no SigFlags found.
    1: at least one signature field exists.
    3: contains signatures that may be invalidated if the file is saved (written)
    in a way that alters its previous contents, as opposed to an incremental update.
    =======================================================================
    
    Reference:
    ---------
    - forum détection signature pdf:
      https://github.com/pymupdf/PyMuPDF/issues/326
    - fonction get_sigflags:
      https://pymupdf.readthedocs.io/en/latest/document.html#Document.get_sigflags
    """
    # ouvre le document
    doc = fitz.open(path_pdf) 
    # regarde si le document est signé
    sig = doc.get_sigflags()
    # ferme le document
    doc.close()
    # met à jour la variable is_signed
    if sig == -1:
        is_signed = False
    else:
        is_signed = True
    return is_signed


#***********************************************************
# Effectue toutes les opérations du module detectSignature à la suite
#***********************************************************
def drawProgressBar(percent, add_text='', barLen=20):
    """
    Affiche à l'écran une barre de progrès montrant l'avancement
    d'une fonction.
    
    Input:
    ------
    - percent (float) : pourcentage d'avancement entre 0 et 1.
    - add_text (string) : texte optionnel à ajouter à l'affichage
    - barLen (int) : longueur de la barre de progrès.
    
    Output:
    ------
    - Affichage de la barre de progrès à l'écran.
    
    Refrence:
    --------
    - https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
    """
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%  {}".format("=" * int(barLen * percent), barLen, percent * 100, add_text))
    sys.stdout.flush()

def sec2min(time):
    """
    Cette fonction prend en entrée un argument time en secondes
    et renvoie un tuple contenant la valeur de time en (minutes,secondes).
    """
    minutes = int(time//60) 
    secondes = round(time - minutes*60,3)
    return (minutes,secondes)

def main_detectSignature(path_data:str, keyWords_template:list, name_pdf:str=None,
                         saved_template=None,template_shape=None,template_text=None,path_saved_template:str=None,
                         path_save_result:str=None, path_save_imsig:str=None,                     
                         similarity_thresh:float=0.5, start_with="end", debug:bool = False,
                         cropper_params:dict={"min_region_size":1e2,"border_ratio":0.01},
                         judger_params:dict={"pixel_ratio": [0.001,1]}): #/!\ judger_params a changé    
    """
    Cette fonction analyse des documents pdfs et détecte si ils sont signés.

    Cette fonction :
    - Charge les pdfs à vérifier
    - Charge les coordonnées du template sauvegardées au nom saved_template.pickle 
      ou utilise la valeur entrée en argument. 
      Pour chaque pdf : 
    - Détecte si le PDF est signé numériquement, sinon :
    - Détecte la page contenant la signature à l'aide de Tesseract.
    - Calcule la position du premier mot de la liste keyWords_template trouvé sur la page.
    - Extrait les zones de l'image contenant la signature à l'aide de la position relative
      des cadres de saved_template par rapport aux coordonnées des mots de keyWords_template.
    - Parcourt de toutes les zones de l'image extraites et teste la présente de signature 
      pour chaque zone. Si au moins une zone contient une signature, le pdf est signé.    
    - Sauvegarde le résultat toutes les 10 itérations au chemin indiqué par path_save_result.
    - Si le fichier indiqué par path_save_result existe déjà, on reprend l'analyse des pdfs
      en analysant les pdfs qui se situent dans le répertoire path_data (ou indiqués par name_pdfs)
      mais qui ne sont pas dans le fichier indiqué par path_save_result. 

    >>> Utilise les fonctions load_pdf(), find_page_with_Tesseract()
    find_1_keyWord_coord(), extract_signature_boxes(), verify_signature(), check_signature_num()
    et show_image().
    Pour d'avantage d'informations sur l'utilisation et le rôle de ces 
    fonctions, se référer à leur documentation.  

    Input:
    ------
    - path_data (string) : chemin du répertoire des pdfs à analyser. Ex: path_data="C:/Users/A1234/data"
    
    - name_pdf (string, liste de string ou None) : nom du ou des pdfs à analyser. Ex: name_pdf="monpdf_1.pdf" 
      si on veut analyser 1 seul pdf OU name_pdf=["monpdf_1.pdf","monpdf_2.pdf"] pour plusieurs pdfs.
      Si name_pdf=None, tout le répertoire indiqué par path_data est analysé. 
      
    - keyWords_template (liste de strings) : liste contenant les mots clés à détecter dans la 
      page du pdf. Cette liste doit être identique à la liste keyWords_template sélectionnée dans 
      le template.
      
    - saved_template (liste de dictionnaire ou None) : saved_template=list_relative_points. Liste de 
      dictionnaires où chaque dictionnaire représente les coordonnées d'un rectangle supposé contenir 
      la signature par rapport aux différents mots clés de keyWords_template. Si saved_template=None,
      alors le template sera récupéré dans le répertoire indiqué par path_saved_template en
      chargeant le fichier "saved_template.pickle" prélablement créé par le module template.
      
    - template_shape (tuple ou None) : (height,width) ou (height,width,depth) taille de l'image
      ayant servi de template. Si template_shape=None, alors template_shape sera récupéré dans le répertoire
      indiqué par path_saved_template en chargeant le fichier "template_shape.txt" prélablement créé par le 
      module template.
      
    - template_text (string ou None): texte contenu dans la page du pdf template. 
      Si template_text=None, alors le texte sera récupéré dans le répertoire indiqué par 
      path_saved_template en chargeant le fichier "template_text.txt" prélablement créé 
      par le module template. 
      
    - path_saved_template (string ou None) : chemin du répertoire contenant la sauvegarde du template
      "saved_template.pickle", la shape du template "template_shape.txt" et le texte du 
      template "template_text.txt" préalablement créés par le module template. 
      Si path_saved_template=None, alors cela veut dire que l'on a renseigné les arguments
      saved_template, template_shape et template_text. 
      
    - path_save_result (string ou None) : chemin du fichier .csv vers lequel sauvegarder la sortie de cette
      fonction, Ex: path_save_result = "C:/Users/A1234/result/mon_resultat.csv". Si on a plus de 10 pdfs
      à analyser, la sauvegarde se fait tous les 10 pdfs. 
      Si path_save_result=None, rien n'est sauvegardé. 
      Si le fichier indiqué par path_save_result existe déjà, alors le code lit ce fichier et reprend 
      l'analyse des documents là où elle s'était précédemment arrêtée. Pour savoir à quel pdfs on s'était
      arrêtés lors de l'analyse précédente, on lit le fichier indiqué par path_save_result et on regarde 
      quels documents du répertoire path_data (ou quels document indiqués par name_pdf) ne sont 
      pas dans le fichier .csv. On analyse ces documents. 

    - path_save_imsig (string ou None) : chemin du répertoire où sauvegarder : 
      1) L'image du cadre supposé contenir la signature (avant Cropper).
         L'image est sauvegardée dans le répertoire path_save_imsig/before_cropper.
      2) L'image de la signature extraite par le Cropper dans le cadre supposé contenir 
         la signature (après le Cropper).  
         L'image est sauvegardée dans le répertoire path_save_imsig/after_cropper.
      Ex : path_save_imsig="C:\A0000\resultats_signatures" et path_save_imsig contient les 
      sous-répertoire before_cropper et after_cropper. 
      Si le répertoire path_save_imsig, before_cropper et after_cropper n'existent pas déjà, 
      il sont créés. Si path_save_sig=None, aucune image n'est sauvegardée.  
      Le nom des images est : "nomDuPDF_bci.jpg" où dans "bci", "i" est le numéro du cadre 
      supposé contenir la signature (il peut y avoir plusieurs cadres) et "bc" indique 
      que c'est une image "before cropper". "nomDuPDF_aci.jpg" indique une image "after cropper". 
      
    - similarity_thresh (float) : seuil de similarité entre 2 textes pour la fonction 
      find_page_with_Tesseract(). Si la similarité calculée est en dessous de ce seuil, 
      alors les deux textes ne sont pas similaires.

    - start_with (string) : peut prendre les valeurs "begin" et "end" et permet d'indiquer
      si on commence la recherche de la page contenant la signature en partant du début du 
      pdf (avec start_with="begin") ou de la fin du pdf (avec start_with="end"). 
      
    - debug (boolean) : si True, des images et du texte sont affichés aux différentes étapes
      pour pouvoir mieux visualiser ce que fait la fonction. Lorsque qu'il y a trop d'images à analyser,
      il est déconseillé de mettre debug=True pour ne pas trop afficher. 
      
    - cropper_params = {"min_region_size":1e3,"border_ratio":0.01}. 
      Contient les paramètres du Cropper pour la fonction verify_signature(). 
      
    - judger_params (dict): judger_params = {"pixel_ratio": [0.01,1]}
      Contient les paramètres du Judger pour la fonction verify_signature().

    Output:
    ------
    - result (dataframe) : dataframe contenant le résultat de l'analyse. Si path_save_result est 
      différent de None, la dataframe result est sauvegardée au chemin indiqué par path_save_result. 
    """

    # Temps de départ
    st = time.time()

    # ----------------------------
    # 1) Lecture des fichiers pdfs
    # ----------------------------
    if not(name_pdf): #si name_pdf=None, on analyse tout le répertoire donné par path_data
        directory = os.listdir(path_data)
    elif type(name_pdf)==str: #si name_pdf est un string, on analyse seulement le pdf nommé par name_pdf
        directory = [name_pdf]
    else: #sinon, cela veut dire que name_pdf est déjà une liste
        directory = name_pdf
    #utile pour drawProgressBar dans le cas où directory vide
    len_dir = len(directory)
    if len_dir==0: 
        file_name=''; len_dir=1; total_min=0; total_sec=0 
    
    # ----------------------------
    # 2) Récupération de saved_template, template_shape, template_text
    # ----------------------------
    # Récupération des coordonnées relatives du template... 
    if not(saved_template):
        with open(path_saved_template + '/saved_template.pickle', 'rb') as handle:
            saved_template = pickle.load(handle)

    # ...et de la shape du pdf template...
    if not(template_shape):
        with open(path_saved_template + '/template_shape.txt') as f:
            template_shape = f.read()  
        template_shape = eval(template_shape) #conversion str -> tuple
  
    # ...et du texte contenu dans le pdf template
    if not(template_text):
        with open(path_saved_template + '/template_text.txt', encoding="utf-8") as f:
            template_text = f.read()  

    # ----------------------------
    # 3) Création de la dataframe de résultat
    # ----------------------------        
    result = pd.DataFrame() 
    
    if path_save_result :
        # exist_file=True si le fichier indiqué par path_save_result existe déjà 
        exist_file = Path(path_save_result).is_file() 
        
        # ----------------------------  
        # 4.a) Création du fichier csv pour écrire le résultat 
        # si le fichier indiqué par path_save_result n'existe pas, on ne reprend pas les calculs de 
        # l'analyse précédente et donc on crée un nouveau fichier csv.
        # ----------------------------  
        if not(exist_file): # création du fichier csv
            print("\nCréation du fichier",path_save_result)
            # création du fichier en mode lecture
            f = open(path_save_result, 'w', encoding='utf-8')
            # création du csv writer
            writer = csv.writer(f,delimiter=',') #csv.writer(f,delimiter=";") 
            # écriture de l'entête du fichier csv
            writer.writerow(['file_name','result','execution_time'])
            # fermeture du fichier 
            f.close()

        # ----------------------------
        # 4.b) Lecture du fichier csv pour écrire le résultat 
        # si fichier indiqué par path_save_result existe, alors on reprend les calcules précédents
        # là où on s'était arrêtés. 
        # ----------------------------
        else:
            print("\nLe fichier {} existe déjà. On reprend l'analyse à partir du dernier élément du fichier.".format(path_save_result))
            # récupération de l'information contenue dans le fichier indiqué par
            # path_save_result sous forme d'une dataframe
            result = pd.read_csv(path_save_result,delimiter=',')
            # on supprime de directory le nom des fichiers qui sont déjà dans result
            # pour ne pas analyser 2 fois les mêmes fichiers
            read_files = list(result["file_name"].values) #fichier lus dans le csv
            directory = list(set(directory).difference(set(read_files))) #nouveau directory
            #utile pour drawProgressBar dans le cas où directory vide
            len_dir = len(directory)
            if len_dir==0: 
                file_name=''; len_dir=1; total_min=0; total_sec=0

    print("\nDebut analyse...")
    
    # Parcourt de tous les pdf du répertoire
    #====== BEGIN for loop [
    j = 0
    for file_name in directory:
        
        # temps de début de l'itération
        st_it = time.time()

        # ----------------------------
        # 5) Affichage d'une barre de progression si debug=False
        # ----------------------------
        if not(debug):
            drawProgressBar(percent=j/len_dir,add_text="file "+file_name)
        if debug :
            print("\n**************************************")
            print("Fichier", file_name)
            print("**************************************")

        # ----------------------------
        # 6) Chargement du pdf à vérifier
        # ----------------------------
        pdf = load_pdf(file_name,path_data,poppler_path) ####### 

        # ----------------------------
        # 7) Détection de la présence d'une signature numérique
        # ----------------------------
        #le document est-il signé numériquement?
        is_signed_num = check_signature_num(path_data+"/"+file_name) 
        if is_signed_num: #La signature est-elle numérique ?
            output_text = "signature numerique"
        
        else: #Sinon, la signature est peut-être papier

            # ----------------------------
            # 8) Détection de la page contenant la signature 
            # ----------------------------
            im, image_info, warning1 = find_page_with_Tesseract(pdf=pdf,
                                                                template_text=template_text,
                                                                similarity_thresh=similarity_thresh,
                                                                template_shape=template_shape,
                                                                start_with=start_with,
                                                                verbose=debug)

            if warning1: #si la page n'a pas pu être détectée, on arrête l'analyse et on met un warning
                output_text = "Page contenant la signature non detectee"
            else: #sinon, on poursuit l'analyse

                # ----------------------------
                # 9) Calcul de la position du premier mot de la liste keyWords_template trouvé sur la page
                # ----------------------------
                key_point, keyWord, warning2 = find_1_keyWord_coord(im,keyWords_template,image_info,debug)

                if warning2: #si les mots clés n'ont pas été trouvés, on arrête l'analyse et on met un warning
                    output_text = "Mots cles pour identifier la position du cadre non detectes"
                else: #sinon, on poursuit l'analyse

                    # ----------------------------
                    # 10) Extraction des zones de l'image contenant la signature 
                    # ----------------------------
                    list_im_boxes = extract_signature_boxes(im,key_point,keyWord,saved_template,debug=debug)

                    # ----------------------------
                    # 11) Parcourt de toutes les zones de l'image extraites et teste la présente de signature pour chaque zone.
                    # ----------------------------
                    is_signed = [] 

                    for i in range(len(list_im_boxes)):
                        im_box = list_im_boxes[i]
                        res, im_sig = verify_signature(im_box,
                                              cropper_params = cropper_params,
                                              judger_params = judger_params,
                                              verbose = debug)
                        is_signed.append(res) 
                    
                        # ----------------------------      
                        # 12) Sauvegarde de l'image "im_box" dans le cadre avant cropper 
                        # ET de l'image après crpper "res" et si path_save_sig!=None
                        # ----------------------------
                        if path_save_imsig and not (im_sig is None):
                            # Si le répertoire indiqué n'existe pas, on le crée
                            if not os.path.isdir(path_save_imsig): os.makedirs(path_save_imsig) 
                            # Création des répertoire before_cropper et after_cropper s'ils n'existent pas déjà 
                            if not os.path.isdir(path_save_imsig+'/before_cropper'):os.makedirs(path_save_imsig+'/before_cropper')
                            if not os.path.isdir(path_save_imsig+'/after_cropper'):os.makedirs(path_save_imsig+'/after_cropper')
                            # Sauvegarde des images dans les dossiers before_cropper et after_cropper
                            pil_im_sig = Image.fromarray(im_sig) #transformation de l'image au format pil
                            pil_im_box = Image.fromarray(im_box)
                            pil_im_sig.save(path_save_imsig+"/after_cropper/"+file_name[:-4]+"_ac"+str(i+1)+".jpg") #sauvegarde de l'image
                            pil_im_box.save(path_save_imsig+"/before_cropper/"+file_name[:-4]+"_bc"+str(i+1)+".jpg") 

                    # Si au moins 1 rectangle contient une signature, le pdf est signé    
                    if np.any(is_signed): 
                        output_text = "signature ok"
                    else:
                        output_text = "no signature"
            
        # ----------------------------
        # 13) Calcul du temps d'exécution
        # ----------------------------
        et = time.time() #temps de l'itération
        
        it_execution_time = et - st_it #temps d'éxécution de l'itération 
        it_min,it_sec = sec2min(it_execution_time) #conversion en minutes et secondes
        
        total_execution_time = et - st #temps d'éxécution total depuis le début 
        total_min,total_sec = sec2min(total_execution_time) #conversion en minutes et secondes
        
        # ----------------------------
        # 14) Mise à jour de la dataframe résultat
        # ----------------------------
        result = result.append({'file_name' : file_name, 'result' : output_text, 
                                'pdf_execution_time': "{} min {} sec".format(it_min,it_sec),
                                'total_execution_time': "{} min {} sec".format(total_min,total_sec)},
                               ignore_index = True)  
        
        # ----------------------------
        # 15) Ecriture du résultat dans le fichier .csv
        # ----------------------------
        if path_save_result :
            # sauvegarde toutes les 10 itérations
            if j!=0 and j%10==0: # "if" à retirer si on veut sauvegarder à toutes les itérations
                result.to_csv(path_save_result,index=False) # Sauvegarde du résultat
                
                
        j+=1
    #====== END for loop ]
    
    # ----------------------------
    # 16) Zauvegarde finale dans tous les cas
    # (même si le nombre total d'itérartions est < 10)
    # ----------------------------
    if path_save_result :
        result.to_csv(path_save_result,index=False)

    # ----------------------------
    # 17) Affichage de la fin de la barre de progression si debug=False
    # ----------------------------
    if not(debug):
        drawProgressBar(percent=j/len_dir, add_text='file '+ file_name)

    # ----------------------------
    # 18) Affichage écran 
    # ----------------------------
    print("\n...fin analyse !")
    print("Temps d'exécution total: {} minutes {} secondes".format(total_min,total_sec))
    
    return result