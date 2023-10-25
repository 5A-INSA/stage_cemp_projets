"""

Ce module permet de définir le template pour la détection de signature.
Ce module demande à l'utilisateur de choisir un pdf servant de template et de définir à la 
main le numéro de page contenant la signature.
L'utilisateur doit par la suite saisir une liste de mots clés keyWords_template. Ces mots 
doivent être présents sur la page contenant la signature. 
Enfin, l'utilisateur doit dessiner sur la page du pdf template, les rectangles supposés 
contenir la signature. 
La fonction calcule alors les coordonnées de ces rectangles par rapport à la position des mots 
clés de keyWords_template renvoyée par l'OCR (Tesseract). Ces coordonnées relatives sont renvoyées 
sous le nom de la variable list_relative_points. 

ATTENTION, l'image template doit être "parfaite" c'est-à-dire que tous les mots clés 
keyWords_template doivent être présents et lisibles par Tesseract dans la page du 
pdf template. Sinon, une erreur peut se produire. 
Dans ce cas, une solution est de changer l'image d'entrée ou de modifier la liste
des mots clés. 

Règles d'or pour le choix de keyWords_template : 
  - Chaque élément de la liste keyWords_template peut être un seul mot ou une expression
  - Tous les mots de la liste dovent être présent dans la page
  - Chaque mot de la liste ne doit être présent qu'une seule fois dans la page
  - Ne pas mettre d'espace après les mots de la liste keyWords_template.
  - Les éléments de keyWords_template peuvent contenir des accents ou non,
    l'algorihtme est insensible aux accents.

Contient les fonctions : 
- read_page()
- get_screenSize()
- ResizeWithAspectRatio()
- extract_start_rectangle()
- display_instructions()
- create_template()
- find_expression_in_list()
- find_all_keyWord_coord()
- compute_relative_coord()
- main_template()
"""

#***********************************************************
# Chargement des librairies
#***********************************************************
from set_global import PATH_TESSERACT, PATH_POPPLER
from preprocessing import *

#---- Calculs 
import numpy as np
#---- Traitement des PDFs
import cv2 #traitement d'image (jpg)
#----- Détection de texte avec Tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = PATH_TESSERACT # Définit le chemin de tesseract
import re
import unicodedata
#----- Sauvegarde 
import pickle 

#***********************************************************
# Lecture du texte de la page template
#***********************************************************
def read_page(im,output_list=False):
    """
    Cette fonction prend en argument une image le texte
    de l'image à l'aide de l'OCR Tesseract.
    Cette fonction débruite au préalable l'image pour améliorer
    la lecture par tesseract mais la fonction ne rotatione PAS 
    l'image pour la mettre dans le bon sens. 
    
    Intput:
    -------
    - im (np.array) : image à lire
    - output_list (bool) : si True, le texte est renvoyé sous
      forme de liste où chaque élément de la liste est un mot du texte.
      Si False, text est un string.

    Output:
    -----
    - text (liste de string OU string) : texte contenu dans l'image où 
      chaque élément de la liste est un mot du texte si output_list=True
      et texte contenu dans l'image sous forme de string si output_list=False.
    """
    
    # Débruitage de l'image pour améliorer la lecture.    
    im_prep = denoise_image(im)
    text = pytesseract.image_to_data(im_prep,output_type=pytesseract.Output.DICT)["text"]
    
    if not(output_list):
        text = ' '.join(text)
        
    return text

#***********************************************************
# Demande à l'utilisateur de dessiner les rectangles contenant la signature
#***********************************************************
def get_screenSize():
    """
    Cette fonction permet d'obtenir la taille de l'écran d'ordinateur.
    
    Output:
    -------
    - screenHeight, screenWidth (int) : hauteur et largeur de l'écran.
    
    Reference:
    ----------
    https://stackoverflow.com/questions/11041607/getting-screen-size-on-opencv
    """
    # create a window in full-screen mode and ask for its width and height
    cv2.namedWindow("dst", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("dst",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    (a,b,screenWidth,screenHeight) = cv2.getWindowImageRect('dst')
    cv2.destroyAllWindows()
    return screenHeight, screenWidth #(height, width)
    
    
def ResizeWithAspectRatio(image, width=None, height=None):
    """
    Cette fonction redimensionne l'image d'entrée à une taille spécifique 
    tout en conservant le rapport hauteur/largeur.
    
    Input:
    ------
    - image (np.array): image que l'on souhaiute redimmensionner.
    - width, height (int): largeur et hauteur de l'image.
    
    Output:
    ------
    - resized_image (np.array): image redimensionnée.
    - r (float) : ratio de redimensionnement.
    
    Reference:
    ----------
    - Redimensionne l'image en conservant le bon ratio:
      https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
    """
    
    dim = None
    (h, w) = image.shape[:2] #(height, width)

    if width is None and height is None:
        return image
    
    if height : 
        r = height / float(h)
        dim = (int(w * r), height) #(width, height)
    else:
        r = width / float(w)
        dim = (width, int(h * r)) #(width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image, r


def extract_start_rectangle(point_matrix):
    """
    Cette fonction prend en argument les coordonées (x1,y1) et (x2,y2) définissant 
    un rectangle et renvoie le point (xr,yr) en haut à gauche du rectangle ainsi que 
    hr et wr représentant respectivement la hauteur et la largeur du rectangle.
    
    Input:
    -----
    - point_matrix (np.array) : tableau des points définissant le rectangle
      point_matrix = np.array ([[x1,y1],
                                [x2,y2]])
    
    Output:
    ------
    - box_prop (list) : box_prop = [xr,yr,wr,hr] où (xr,yr) est le point en haut à gauche du 
      rectangle et wr,hr sont respectivement la largeur et la hauteur du rectangle.  
    
    =================== Exemple:  =================== 
    (x1,y1) = coordonées de debut du rectangle, (x2,y2) = coordonées de fin du rectangle     
    
    repère : (0,0) *-- x 
                   |
                   y
    
    (x1,y1) *-------
            |      |            alors (xr,yr) = (x1,y1)
            |      |
            -------* (x2,y2) 
    
            -------* (x1,y1)
            |      |            alors (xr,yr) = (x2,y1)
            |      |
    (x2,y2) *-------    
    =================================================
    """
    
    x1,y1 = point_matrix[0]
    x2,y2 = point_matrix[1]

    # Définition largeur et hauteur du rectangle
    wr = abs(x2 - x1) 
    hr = abs(y2 - y1)

    # On s'assure que les coordonées (xr,yr) représentent
    # le point en haut à gauche du rectangle
    if (x1 <= x2) and (y1 <= y2):
        xr,yr = x1,y1
    elif (x1 >= x2) and (y1 <= y2):
        xr,yr = x2,y1
    elif (x1 >= x2) and (y1 >= y2):
        xr,yr = x2,y2
    else: #(x1<= x2) and (y1>=y2)
        xr,yr = x1,y2

    box_prop = [xr,yr,wr,hr] 
    return box_prop


def display_instructions(displayed_text):
    """
    Cette fonction crée une image blanche et affiche le texte 
    contenu dans displayed_text.
    
    Input:
    ------
    - displayed_text (list of string) : liste contenant le texte à afficher.
      A chaque nouvel élément de la liste, une nouvelle ligne de texte est affichée.
    
    Output:
    ------
    - im_text (np.array) : image contenat le texte à afficher.
    """

    # Création d'une image blanche pour y afficher du texte d'aide
    im_text = np.zeros([200,500,3],dtype=np.uint8)
    im_text.fill(255)

    # Affichage du texte d'aide
    x,y = (10,30)
    for text in displayed_text:
        im_text = cv2.putText(im_text, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y = y + 20
        
    return im_text

def create_template(im,show_instructions=True,windowHeight=None,windowWidth=None):
    """
    Cette fonction permet de définir un template en traçant les rectangles 
    devant contenir les signatures. Ce template n'est à définir qu'une seule fois. 
    >>> Utilise les fonctions get_screenSize(), ResizeWithAspectRatio() 
        et extract_start_rectangle()

    Input:
    ------
    - im (np.array) : image sur laquelle tracer le template.
    - show_instructions (boolean) : si True, une fenêtre contenant 
      les instructions est affichée.
    - windowHeight, windowWidth (int) : taille de l'image à afficher. 
      Si les deux valeurs sont à None, la taille de l'image est calculée 
      pour s'accomoder à la taille de l'écran d'ordinateur. 

    Output:
    ------
    - box_prop_list (liste de np.array) : liste des proportions du rectangle. 
      box_prop_list[0] = [xr,yr,wr,hr] où (xr,yr) est le point en haut à gauche du rectangle
      tracé et wr,hr sont respectivement la largeur et la hauteur du rectangle. 
      Ainsi, si on dessine 2 rectangles sur l'image len(box_prop_list) = 2 et :
      box_prop_list = [[xr1,yr1,wr1,hr1], [xr2,yr2,wr2,hr2]].
      Pour davantage d'infos, voir la doc de la fonction extract_start_rectangle().
    - img (np.array) : image sur laquelle on a tracé les rectangles.

    Reference:
    ----------
    - Dessiner un rectangle sur une image avec OpenCv: 
      https://www.tutorialspoint.com/opencv-python-how-to-draw-a-rectangle-using-mouse-events
      https://mlhive.com/2022/04/draw-on-images-using-mouse-in-opencv-python

    ============================ Notes ============================
    Une fois ce template avec les rectangles définis, on calculera les coordonées des rectangles 
    en fonction des coordonées de mots clés sur la page (ex: mot clé SIGNATURE). 
    Ainsi, lorsqu'un nouveau pdf se présentera, les coordonées des rectangles s'adapteront directement 
    aux coordonnées des mots clés et les rectangels seront donc toujours bien situés.
    ==============================================================
    """
    global rectangle_list, point_matrix, counter, drawing, ix, iy, img
    
    #windowHeight,windowWidth = None,None
    img = im.copy()

    # ======================================
    # Définition des variables 
    # ======================================
    # Récupère la taille de l'écran de l'ordinateur
    if windowHeight is None and windowWidth is None:
        windowHeight,windowWidth = get_screenSize()    

    # Variables pour dessiner sur l'image
    drawing = False
    ix,iy = -1,-1
    # Compteur pour savoir le n° du rectangle que l'on dessine
    # (on peut dessiner plusieurs rectangles sur une même image)
    counter = 0

    # Crée une matrice de points pour stocker les coordonnées du clic de la souris 
    # sur l'image ie stocker les coordonées des rectangles dessinés.
    point_matrix = np.zeros((2,2),int) # point_matrix = array([[x1,y1],
                                       #                      [x2,y2]])

    # Liste des coordonées des rectangles dessinés. 
    # Contient les point_matrix de chaque rectangle  
    # i.e. c'est une liste de tableaux array([[x1,y1], 
    #                                         [x2,y2]])
    # où [x1,y1] sont les coordonées de début du rectangle 
    # et [x2,y2] sont les coordonnées de fin de rectangle. 
    rectangle_list = [] 
    
    print("\nVeuillez dessiner le template.")
    # ======================================
    # define mouse callback function to draw a rectangle
    # ======================================
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, img, counter, point_matrix 

        if event == cv2.EVENT_LBUTTONDOWN: #check if mouse event is left button down
            # checking if any left click (cv2.EVENT_LBUTTONDOWN) is happening or not
            drawing = True #on commence à enregistrer la souris
            ix = x
            iy = y
            # storing pixel coordinates of each point where we have done mouse click on image
            point_matrix[counter] = x,y #accès à la 1ère ligne de point_matrix: coordonées (x1,y1) du rectangle 
            counter += 1 # on a dessiné le 1er point (x1,y1), incrémentation du compteur.

        elif event == cv2.EVENT_LBUTTONUP: #check if mouse event is Left button up
            drawing = False #on arrête d'enregistrer la souris
            cv2.rectangle(img, (ix, iy),(x, y),(0, 0, 255),1) # on dessine le rectangle
            point_matrix[counter] = x,y #accès à la 2e ligne de point_matrix: coordonées (x2,y2) du rectangle 
            # ajout des coordonées (x1,y1),(x2,y2) (= le 1er rectangle) à la liste de tous les rectangles dessinés
            rectangle_list.append(point_matrix) 
            counter = 0  # remise à zéro du compteur pour le prochain rectangle
            point_matrix = np.zeros((2,2),int) # remise à zéro de la matrice de points pour le prochain rectangle

    # ======================================
    # Create window to draw and connect the mouse
    # ======================================   
    # Create a window to draw the rectangles
    cv2.namedWindow("Template")

    # And bind the function to window :
    # connect the mouse button to our callback function
    # callback function is a function passed into another function as an argument
    cv2.setMouseCallback("Template", draw_rectangle)
    
    # ======================================
    # Create window to display instuctions
    # ======================================  
    if show_instructions:
        displayed_text = ['Dessiner les rectangles contenant la signature.',
                    '   "Exit" pour annuler.', '   "Enter" pour sauvegarder.',
                    '   "Ctrl+z" pour retour en arriere.']
        im_text = display_instructions(displayed_text)

    # ======================================
    # Display the window
    # ======================================
    #on recupère la valeur du ratio que la première fois qu'on passe dans la boucle while
    #sinon, les autres fois, ratio vaut toujours 1 
    first_ratio = True 
    while True:
        
        # Redimensionne l'image en conservant le ratio pour que la hauteur 
        # de l'image soit la même que la hauteur de l'écran.
        # Sinon, OpenCV n'adapte pas la taille de la fenêtre à l'écran
        if first_ratio : # stocke valeur du ratio uniquement la 1er fois qu'on passe dans la boucle
            img,ratio = ResizeWithAspectRatio(img, height=windowHeight, width=windowWidth)
            first_ratio = False
        else : # pour les autres fois, ratio vaut toujours 1
            img,_ = ResizeWithAspectRatio(img, height=windowHeight, width=windowWidth)
            
        
        # Affiche l'image
        cv2.imshow("Template", img)
        
        if show_instructions:
            cv2.imshow("Instructions", im_text)

        k = cv2.waitKey(10)

        if k == 27: #key "escape" to stop and delete changes
            print("Exit - aucun template créé.")
            img = im.copy() #reset image
            rectangle_list = [] #reset list of coordinates 
            break # escape window

        elif k == 13: #key "enter" to stop and save changes
            print("Template créé avec succès.")
            break # escape window and all parameters are saved

        elif k == 26: #key "ctrl + z" to come back 
            img = im.copy() #reset image
            counter = 0 #reset counter
            rectangle_list = [] #reset list of coordinates 

    cv2.destroyAllWindows()
    
    # Comme on a appliqué un redimensionnement, il faut remettre 
    # les coordonées à l'échelle d'origine.
    # On calcule de plus avec la fonction extract_start_rectangle les 
    # proportions du rectangle (voir la doc de extract_start_rectangle)
    # pour d'avantage d'infos
    box_prop_list = []
    for i in range(len(rectangle_list)):
        temp = np.multiply(rectangle_list[i],1/ratio).astype(int)
        box_prop_list.append(extract_start_rectangle(temp))
                 
    return box_prop_list,img


#***********************************************************
# Calcul des coordonnées des mots clés keyWords_template sur la page contenant la signature
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
    
    On retire d'abord tous les accents et les apostrophes de list_of_strings 
    et expression pour rendre la recherche + robuste.
    
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
    # On retire les apostrophes qui creent des pb pour re.search
    list_of_strings = re.sub("'|’", "",list_of_strings)

    # On remplace dans expression les espaces par le caractère spécial    
    expression = expression.replace(' ', special_char)  
    # On retire tous les accents pour rendre la recherche + robuste
    expression = ''.join(c for c in unicodedata.normalize('NFD', expression) if unicodedata.category(c) != 'Mn')
    # On retire les apostrophes qui creent des pb pour re.search
    expression = re.sub("'|’", "",expression)

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

def find_all_keyWord_coord(im, keyWords_template, debug=False):
    """
    Cette fonction recherche la position de tous les 
    mots clés de la liste keyWords_template. ATTENTION, dans ce cas, l'image doit être "parfaite" 
    c'est à dire que tous les mots clés doivent être lisibles par Tesseract. 
    Sinon, une erreur peut se produire. Dans ce cas, une solution est de changer 
    l'image d'entrée ou de modifier la liste des mots clés. 
    >>> utilise la fonction find_expression_in_list() 


    Input:
    ------
    - im (np.array): image sur laquelle recherche les coordonées des mots clés.
    - keyWords_template (list of string): liste des mots clés à rechercher. La recherche se fait
      dans l'ordre des mots de keyWords_template. Dès qu'un mot est trouvé, on ne recherche pas les 
      mots suivants. Attention à ne pas mettre d'espace à la fin et au début des mots ou des expressions. 
      Ex : ['TITULAIRE','REPRESENTANT LEGAL','BANQUE'] 
    - debug (bool): si True, l'ensemble du texte détecté est affiché.
      Si un mot clé a été trouvé, l'image avec le mot clé encadré est également affichée.
   
    Output:
    -------
    - key_points_dict (dict): dictionnaire de coordonnées 
      ['TITULAIRE':(xa,ya),'REPRESENTANT':(xb,yb),'BANQUE':(xc,yc)] des
      mots trouvés. En key : le mot et en value ses coordonées.
      
      =============================== Notes ==================================
      Règles d'or pour le choix de keyWords_template : 
      - Chaque élément de la liste keyWords_template peut être un seul mot ou une expression
      - Tous les mots de la liste dovent être présent dans la page
      - Chaque mot de la liste ne doit être présent qu'une seule fois dans la page
      - Ne pas mettre d'espace après les mots de la liste keyWords_template.
      ========================================================================
      
    """

    image = im.copy() 

    # ==============================================
    # Extraction des informations de l'image
    # ==============================================
    #debruitage de l'image pour meilleure lecture
    im_prep = denoise_image(image)
    #extrcation de toutes les informations de l'image à l'aide de Tesseract
    results = pytesseract.image_to_data(im_prep, output_type=pytesseract.Output.DICT)
    #liste de tous les mots trouvés par Tesseract
    Text = results["text"] 
    
    # Dictionnaire de coordonnées des mots. En key : le mot et en value 
    # ses coordonées
    key_points_dict = dict()

    # ==============================================
    # Recherche de la position de tous les mots clés
    # ==============================================
    # parcourt de tous les mots de la liste
    for i in range(len(keyWords_template)):
        #mot clé 
        keyWord = keyWords_template[i] 
        #indice du mot keyWords_template[i] dans la liste Text
        #-- Méthode 1 avec np.where. Ne trouve que le mot exactement identique à 'TITULAIRE' par ex.
        #idx = np.where(np.array(Text) == keyWord)[0][0]
        #-- Méthode 2 avec Regex. Peut trouver 'TITULAIRE' dans 'DUTITULAIREE'
        #idx = [j for j, item in enumerate(Text) if re.search(keyWord, item)][0]
        #-- Méthode 3 avec find_expression_in_list
        idx = find_expression_in_list(Text,keyWord)
        try :
            # (x,y) = positions du mot keyWords_template[i]
            # le point d'origine (0,0) se trouve en haut à gauche du document 
            x = results["left"][idx] #abscisse du mot (point en haut à gauche)
            y = results["top"][idx] #ordonnée du mot (point en haut à gauche)

            #ajout des coordonées du mot keyWords_template[i] à la liste des coordonées
            key_points_dict[keyWord] = (x,y) 
 
            if debug:
                w = results["width"][idx] #hauteur du rectangle contenant le mot
                h = results["height"][idx] #largueur du rectangle contenant le mot
                # draw a bounding box around the text
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # draw the text 
                stripped_keyWord = ''.join(c for c in unicodedata.normalize('NFD', keyWord) if unicodedata.category(c) != 'Mn')
                cv2.putText(image, stripped_keyWord, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                # display image

        except TypeError as e:
            
            if debug : 
              print(">>> Fonction find_all_keyWord_coord() <<<")
              print("\n----- Texte trouvé : -----")
              print(Text)
              print("\n----- Mots détectés sur l'image : -----")
              show_image(image)

            raise TypeError("\n\nLe mot '{}' n'a pas pu être trouvé sur la page du pdf. " 
            "Tous les mots clés doivent être trouvés sur la page. "
            "Modifier la liste des mots clés à trouver ou sélectionner une autre page plus lisible.".format(keyWord)) from e
    
    # ==============================================
    # Dans le cas où on a trouvé les mots,
    # on peut afficher les mots trouvés sur l'image pour debugging
    # ==============================================
    if debug : 
        print(">>> Fonction find_all_keyWord_coord() <<<")
        print("\n----- Texte trouvé : -----")
        print(Text)
        print("\n----- Mots détectés sur l'image : -----")
        show_image(image)
    # ==============================================
    
    return key_points_dict


#***********************************************************
# Calcule les coordonnées de ces rectangles par rapport à la position des mots clés keyWords_template
#***********************************************************
def compute_relative_coord(box_prop_list,key_points_dict):
    """
    Cette fonction calcule les coordonées relatives d'un point (xr,yr) 
    par rapport à un nouveau point d'origine (x0,y0). Le point 
    (x0,y0) devient alors la "nouvelle origine". 
    
    Input:
    ------
    - box_prop_list (liste de np.array) : liste de proportions des rectangles. 
      box_prop_list[0] = [xr,yr,wr,hr] où (xr,yr) est le point en haut à gauche du rectangle
      et wr,hr sont respectivement la largeur et la hauteur du rectangle. 
      Le point (xr,yr) est le point pour lequel on veut calculer la position relative par rapport
      à une nouvelle origine (x0,y0).
      Rq : si on a 2 rectangles alors box_prop_list = [[xr,yr,wr,hr], [xr2,yr2,wr2,hr2]].
    
    - key_points_dict (dict): dictionnaire des points d'origine (x0,y0),(x1,y1)... par rapport 
      auxquels on calcule la position relative de (xr,yr)
      En key : le mot auquel correspond la coordonée (x0,y0), en value la coordonée (x0,y0).
      Par exemple : key_points_dict = ['TITULAIRE':(x0,y0),'REPRESENTANT':(x1,y1),'BANQUE':(x2,y2)]
    
    Output:
    ------
    - list_relative_points (list de dict) : liste de dictionnaires où chaque élément de la liste
      représente les coordonnées relatives d'un rectangle.
      Par exemple : list_relative_points[0] = {'TITULAIRE': [xr-x0 , yr-y0, wr, hr],
      'REPRESENTANT': [xr-x1 , yr-y1, wr, hr],'BANQUE': [xr-x2 , yr-y2, wr, hr]},
      est un dictionnaire représentant les coordonnées relatives du premier rectangle
      box_prop_list[0] = [xr,yr,wr,hr] par rapport aux différents mots clés servant de 
      coordonnées d'origine key_points_dict = ['TITULAIRE':(x0,y0),
      'REPRESENTANT':(x1,y1),'BANQUE':(x2,y2)].
      list_relative_points[1] représente les coordonées relatives du 2e rectangle par 
      rapport aux différentes origines.
      On remarque que la largeur et la hauteur wr1 et hr1 restent inchangées pour 
      le rectangle 1.
      
      ===================== Notes =======================  
       Si nous avons redimensioné d'un rapport Rx = nouveauX/ancienX sur l'axe des x
       et d'un rapport Ry = nouveauY/ancienY sur l'axe des y.
       Alors nos nouvelles coordonnées pour le point (x,y) sont (Rx * x, Ry * y).
      ===================================================    
    """
    
    list_relative_points = []

    for i in range(len(box_prop_list)):    
        # dictionnaire 
        temp_dict = dict()

        for keyWord in key_points_dict.keys():
            # nouvelle origine (x,y)
            x,y = key_points_dict[keyWord] 
            # coordonées du point (xr,yr)
            xr,yr,wr,hr = box_prop_list[i]
            # nouvelles coordonées du point (xr,yr)
            # par rapport au point (x,y)
            xr_relatif = xr - x
            yr_relatif = yr - y
            # mise à jour 
            temp_dict[keyWord] = [xr_relatif,yr_relatif,wr,hr]

        list_relative_points.append(temp_dict)
    
    return list_relative_points


#***********************************************************
# Effectue toutes les opérations du module template à la suite
#***********************************************************
def main_template(path_template:str, name_template:str, page_template:int,
                  keyWords_template, path_save_result:int=None, debug:bool=False,
                  show_instructions=True, windowHeight=None, windowWidth=None):
    """
    Cette fonction demande à l'utilisateur de définir le pdf template et de
    tracer sur ce pdf de référence les cadres supposés contenir la signature.
    
    Cette fonction : 
    - Charge le pdf servant de template.
    - Extrait la page du pdf template contenant la signature en renseignant un numéro de page.
    - Demande à l'utilisateur de tracer les rectangles supposés contenir la signature.
      Glisser déposer la souris sur l'image pour tracer les cadres. Appuyer sur "Echap" pour 
      annuler l'opération et ne pas enregistrer, "Enter" pour enregistrer les modifications
      et "Ctrl + z" pour revenir en arrière.
    - Trouve la position de tous les mots clés de la liste keyWords dans le template à l'aide de l'OCR Tesseract.
    - Calcule et renvoie les coordonnées relatives des rectangles par rapport à l'ensemble des mots clés. 
    
    >>> Utilise les fonctions load_pdf(), find_page_with_number()
        create_template(), find_all_keyWord_coord(), compute_relative_coord()
        et show_image().
        Pour d'avantage d'informations sur l'utilisation et le rôle de ces 
        fonctions, se référer à leur documentation. 
        
    Input:
    ------
    - path_template (string) : chemin du répertoire le pdf template. Ex: "C:/Users/A1234/data"
    - name_template (string) : nom du pdf template. Ex: "monpdf.pdf"
    - page_template (integer) : page du pdf template contenant la signature. 
      Attention, les pages commencent à l'indice 0. 
      Avec page_template=-1, la dernière page du pdf est renvoyée. 
    - keyWords_template (liste de strings) : liste contenant les mots clés à détecter dans la 
      page du pdf template.
    - path_save_result (string) : chemin du répertoire vers lequel sauvegarder la sortie de cette fonction.
      Si path_save=None, rien n'est sauvegardé. 
    - debug (boolean) : si True, des images et du texte sont affichés aux différentes étapes
      pour pouvoir mieux visualiser ce que fait la fonction.
    - show_instructions (boolean) : si True, une fenêtre contenant les instructions est affichée 
      lorsque l'utilisateur doit saisir le template.
    - windowHeight, windowWidth (int) : taille de l'image template à afficher à l'écran. 
      Si les deux valeurs sont à None, la taille de l'image est calculée 
      pour s'accomoder à la taille de l'écran d'ordinateur. 
      
    
    Output:
    ------
    - list_relative_points (liste de dictionnaire) : liste de dictionnaires où chaque dictionnaire
      représente les coordonnées d'un rectangle par rapport aux différents mots clés de keyWords_template. 
    - template_shape (tuple) : (height, width, depth) taille de l'image template.
    - template_text (string) : texte contenu dans la page du pdf template.
    """
    
    # 1) Chargement du pdf template
    pdf = load_pdf(name_template,path_template)

    # 2) Page du template contenant la signature
    im = find_page_with_number(pdf,page_template)
    template_shape = im.shape #taille de l'image template
    template_text = read_page(im) #texte de l'image template 

    #==============================
    if debug:
        print("Page sélectionnée : ")
        show_image(im)
    #==============================

    # 3) Demande à l'utilisateur de tracer les rectangles supposés contenir la signature
    box_prop_list,im_template = create_template(im,show_instructions=show_instructions,
                                                windowHeight=windowHeight,windowWidth=windowWidth)

    #==============================
    if debug:
        print("Template sélectionné :")
        show_image(im_template)
    #==============================

    # 4) Trouve la position de tous les mots clés dans le template
    key_points_dict = find_all_keyWord_coord(im, keyWords_template, debug=debug)

    # 5) Calcule les coordonnées relatives des rectangles par rapport à l'ensemble des mots clés
    list_relative_points = compute_relative_coord(box_prop_list,key_points_dict)
    
    # 6) Sauvegarde des coordonnées relatives et sauvegarde de la taille du template
    if path_save_result:
        with open(path_save_result + '/saved_template.pickle', 'wb') as handle:
            pickle.dump(list_relative_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(path_save_result + '/template_shape.txt', 'w') as f:
            f.writelines(str(template_shape))

        with open(path_save_result + '/template_text.txt', 'w', encoding="utf-8") as f:
            f.writelines(template_text)

        print("Template sauvegardé.")
    else:
        print("Template non sauvegardé.")
        
    return list_relative_points, template_shape,template_text