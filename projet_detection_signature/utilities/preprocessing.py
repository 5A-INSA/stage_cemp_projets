"""
   
Ce module contient des fonctions pour le chargement, affichage et pré-traitement des images
afin que celles-ci soient mieux lisibles par un OCR (Tesseract).
Ces fonctions sont utiles pour les modules template et detectSignature.

Contient les fonctions : 
- load_pdf()
- find_page_with_number()
- show_image()
- get_angle_openCV()
- get_angle_tesseract()
- rotate_im()
- denoise_image()
"""

#***********************************************************
# Chargement des librairies
#***********************************************************
from set_global import PATH_TESSERACT, PATH_POPPLER

#---- Calculs 
import numpy as np
#---- Graphes 
import matplotlib.pyplot as plt
#---- Traitement des PDFs
import cv2 #traitement d'image (jpg)
import PIL #traitement d'image (jpg)
from pdf2image import convert_from_path #transformation pdf en jpg
#----- Détection de texte avec Tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = PATH_TESSERACT # Défini le chemin de tesseract
#----- Chemin du poppler pour la fonction convert_from_path de la librairie OpenCV
poppler_path  = PATH_POPPLER


#***********************************************************
# Chargement des pdfs
#***********************************************************
def load_pdf(pdf_name,pdf_path,poppler_path=PATH_POPPLER):
    """
    Cette fonction permet de charger le pdf au format
    PIL.PpmImagePlugin.PpmImageFile.

    Input:
    -----
    - pdf_name (string): nom du pdf (avec l'extension). 
      Ex: si on a 123.pdf alors pdf_name='123.pdf'.
    - pdf_path (string): chemin du pdf 
    - poppler_path (string): chemin du poppler     

    Ouput:
    -----
    - pages (liste de PIL.PpmImagePlugin.PpmImageFile): liste 
      contenant les pages du pdf.   

    ========================= Notes ==========================
    Nous aurons besoin par la suite d'utiliser la librairie OpenCv. 
    Or, pour lire un pdf avec OpenCv il faut d'abord le transformer en jpg ou jpeg avec la 
    fonction convert_from_path de la librairie pdf2image 
    (doc: https://stackoverflow.com/questions/61832964/how-to-convert-pdf-into-image-readable-by-opencv-python). 
    Cependant, pour pouvoir utiliser convert_from_path, il faut installer poppler pour Windows en suivant 
    le lien https://github.com/oschwartz10612/poppler-windows/releases/ 
    (lien indiqué dans le tuto : https://github.com/Belval/pdf2image).

    Une fois sur le site indiqué par le lien de téléchargement, cliquer sur la version 
    Release-23.01.0-0.zip et placer le téléchargement dans le répertoire de votre choix. 
    Dé-zipper le dossier intitulé poppler-23.01.0 et créer une variable poppler_path qui vaut: 
    chemin du répertoire poppler-23.01.0\\Library\\bin. 
    Par exemple, j'ai placé mon dossier poppler-23.01.0 dans le répertoire utilities défini par le chemin :
    C:\\Users\\AXXXXX\\Destop\\utilities.
    Ainsi ma variable poppler_path vaudra : 
    poppler_path  = "C:\\Users\\AXXXXX\\Destop\\utilities\\poppler-23.01.0\\Library\\bin" 
    (penser à doubler les \ pour python).
    ==========================================================
    """
    # On se place dans le dossier des pdfs 
    #os.chdir(pdf_path) 
    # Récupération des pages du pdf
    pages = convert_from_path(pdf_path + "/" + pdf_name, poppler_path=poppler_path)    
    return pages


#***********************************************************
# Extraction d'une page du pdf 
#***********************************************************
def find_page_with_number(pdf,numero_page):
    """
    Cette fonction simple prend en entrée un pdf et 
    un numéro de page et renvoie l'image du pdf associée au
    numéro de page spécifié en argument.
    
    Input:
    ------
    - pdf (liste de PIL.PpmImagePlugin.PpmImageFile): ensemble des pages du pdf
    - numero_page (int) : numéro de page. ATTENTION, la première page a le numéro 0.
      Le numéro -1 permet d'accéder à la dernière page du pdf.
    
    Output:
    ------
    - im (np.array): image correspondant au n° de page spécifié en entrée
    """
    
    im = pdf[numero_page]
    return np.array(im)

#***********************************************************
# Affichage des images
#***********************************************************
def show_image(img):
    """
    Cette fonction permet d'afficher une image sous forme 
    d'un graphique matplotlib.
    
    Input:
    -----
    - img (PIL.PpmImagePlugin.PpmImageFile OU np.array): image à afficher
    
    Output:
    ------
    - Graphique de l'image.    
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


#***********************************************************
# Rotation des images
#***********************************************************
def get_angle_openCV(img,thresh=300):
    """
    Cette fonction permet de calculer l'angle de rotation d'une image 
    uniquement à l'aide de la librairie OpenCV. 
    L'avantage de cette méthode est qu'elle permet de détecter tous les 
    angles de rotation de l'image, et pas seulement les angles 90° et 180°.
    Le désaventage est que cette méthode n'est pas capable de faire la 
    différence entre un angle de 0° et un angle de 180°.

    Input:
    -----
    - im (np.array): image à rotationer.
    - thresh (int): lors de l'extraction des lignes avec HoughLinesP, seules sont renvoyées 
      les lignes qui ont obtenu suffisamment de votes (> thresh).

    Output:
    -------
    - median_angle (float): médiane de tous les angles de l'image.

    ============================= Notes =============================
    La méthode consiste à extraire les contours de l'image avec cv2.Canny et 
    extraire les lignes principales de l'image avec cv2.HoughLinesP. 
    On itère sur chaque ligne et on calcule l'angle entre chaque ligne
    et l'horizontale avec math.atan2. On obtient une liste d'angles.
    On calcule l'angle médian. 

    L'équation normale d'une droite est r = x*cos(w) + y*sin(w) où r est la distance
    perpendiculaire entre l'origine et la ligne et w est l'angle en radians
    entre la ligne perpendiculaire et l'axe horizontal. 

    Si on pose a = cos(w) et b = sin(w) alors a^2 + b^2 = 1.
    De plus, r = x*a+y*b => y = (r-x*a)/b. Avec cette dernière equation, on peut
    trouver les coordonées de n'importe quel point de la droite.

    Ainsi si on par exemple prend x1 = a*r + c*b (où c est une constante)
    alors y1 = (r-x1*a)/b = (r-(a*r + c*b)*a)/b = (r - a^2*r)/b + a*c
    = r(1-a^2)/b + a*c or 1 = a^2 + b^2  donc y1 =  rb + a*c.
    ================================================================
    
     Reference:
    ----------
    - site dont est tiré ce code :
      https://stackoverflow.com/questions/55119504/is-it-possible-to-check-orientation-of-an-image-before-passing-it-through-pytess
    - Normal form line equation :
      https://math.stackexchange.com/questions/2882903/how-can-i-plot-a-straight-line-which-is-in-normal-form
    - Différence entre HoughLines et HoughLinesP :
      https://rsdharra.com/blog/lesson/14.html
    - Documentation HoughLines et HoughLinesP :
      https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html
    - calcul des angles : 
      https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors/42258870#42258870
    - rotation d'une image selon ses lignes horizontales
      https://stackoverflow.com/questions/39752235/python-how-to-detect-vertical-and-horizontal-lines-in-an-image-with-houghlines-w
      https://stackoverflow.com/questions/67684060/rotate-image-to-align-features-with-x-axis-in-opencv-python      
    """
    
    # Trouve les bords de l'image
    # Permet aussi de s'assurer d'avoir une image binaire pour cv2.HoughLines
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)

    # HoughLines: Transform used to detect straight lines.
    lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi/180, threshold=thresh)
    #Seules sont renvoyées les lignes qui ont obtenu suffisamment de votes (>thresh).
    #lines = liste ou chaque élément de la liste vaut: [[rho,theta]] où rho est la distance
    #perpendiculaire en pixels entre l'origine et la ligne et theta est l'angle en radians
    #entre la ligne perpendiculaire et l'axe horizontal. Il s'agit d'une équation de droite
    #sous la forme normale.

    angles = []

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        radians = np.arctan2(y2-y1, x2-x1)
        degrees = np.degrees(radians)

        angles.append(degrees)
        
    median_angle = np.median(angles)

    return median_angle
    
def get_angle_tesseract(im,orientation_conf=5):
    """
    Cette fonction permet de calculer l'angle de rotation d'une image 
    uniquement à l'aide de la librairie Tesseract. 
    Néamoins, cette méthode n'est capable de détecter que les images 
    rotationées à 90° ou à 180°.
    En revanche, l'algorithme fait bien la différence entre une image 
    rotationée à 0° et une image rotationée à 180°. 
    Si le niveau de confiance renvoyépar tesseract
    est < orientation_conf, alors l'image n'est pas rotationée.
    
    Input:
    -----
    - im (np.array): image à rotationer.
    - orientation_conf (float): niveau de confiance renvoyée par tesseract
      quant à la précision de la prédiction de l'orientation du texte. 
      Je considère que orientation_conf >=5 ou 6 donne des résultats 
      plutôts corrects.
    
    Output:
    -------
    - angle (float): angle principal de l'image.
    
    ===================== Notes =====================
    Une méthode consiste donc à appliquer d'abord la fonction 
    get_angle_openCV() pour tourner l'image de l'angle trouvé.
    Cette fonction ne fait pas la différence entre une image 
    tournée à 0° et une image tournée à 180°.
    Donc on applique ensuite la fonction get_angle_tesseract() 
    pour tourner l'image à 0° dans le cas où celle-ci serait à 180°.
    =================================================
    
    Reference:
    ---------
    - Détecter si l'image est rotationée avec Tesseract :
      https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/
      https://stackoverflow.com/questions/55119504/is-it-possible-to-check-orientation-of-an-image-before-passing-it-through-pytess
    """
    
    osd = pytesseract.image_to_osd(im,output_type=pytesseract.Output.DICT)
    
    if osd['orientation_conf'] >= orientation_conf:
        angle = osd['orientation']
    else : 
        angle = 0
    return angle

    
def rotate_im(image, angle):
    """
    Cette fonction permet de rotationner une image 
    d'un certain angle, sans ronger les contours lors
    de la rotation.
    ATTENTION : si l'image est binaire, la sortie im_rot
    ne sera plus bianire car la rotation "dé-binarise" 
    l'image.
    
    Input:
    -----
    - image (np.array): image à rotartionner
    - angle (float): angle en dregrés de la rotation.
    
    Output:
    -------
    - im_rot (np.array): image rotationée correctement.
    
    Reference:
    ----------
    - rotation "correcte" de l'image:
      https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    im_rot = cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255),
                            flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT)
    return im_rot


#***********************************************************
# Débruitage et seuillage des images
#***********************************************************
def denoise_image(im):
    """
    Cette fonction permet de débruiter une image pour améliorer sa lecture par l'OCR.

    Input:
    -----
    - im (np.array): image que l'on veut débruiter

    Ouput:
    ------
    - im_thresh (np.array): image débruitée

    =========================== Notes ============================
    On commence par convertir l'image en niveau de gris pour pouvoir appliquer 
    n'importe quel seuillage avec OpenCV. 
    
    On applique ensuite un filtre bilateral pour débruiter l'image tout en conservant les contours.
    Il eciste principalement 3 filtres utilisés pour le débruitage : le filtre gaussien, le filtre 
    median et le filtre bilatéral. 
    Le filtre gaussien et le filtre médian sont tous deux utiles pour débruiter l'image tout en
    conservant les contours. A priori, le filtre médian conserve mieux les contours que le filtre 
    gaussien. Cependant, ce sont des filtres basés sur le domaine de la fonction ne tiennent pas compte
    du fait qu'un pixel est un pixel de bord ou non. 
    Ces filtres se contentent d'attribuer des poids en fonction de la proximité spatiale, 
    ce qui a pour effet d'estomper les contours.
    
    
    Au final, le bilateral filtering (voir explications dans les liens en Reference) 
    est une meilleure méthode pour débruiter tout en préservant les countours. 
    Il existe 2 types de filtres : 
    Les filtres de plage (de valeurs), attribuent des poids en fonction de la différence d'intensité des pixels. 
    Les filtres de domaine attribuent des poids en fonction de la proximité spatiale des pixels (ex: 
    filtre gaussien ou filtre médian).
    Le filtre bilatéral combine filtre de domaine et filtre de plage.
    Le filtre de domaine s'assurera que seuls les pixels proches (disons une fenêtre de 3×3) sont pris en compte 
    pour le flou, puis le filtre de plage s'assurera que les poids dans cette fenêtre de 3×3 sont donnés en 
    fonction de la différence d'intensité par rapport au pixel central. 
    De cette manière, les bords sont préservés. 


    On applique enfin un suillage Adaptatif (Adaptive Thresholding). 
    Le seuillage Adaptatif est plus précis qu'un seuillage binaire et est utile si 
    les conditions d'éclairage d'une image varient d'une zone à l'autre.
    Il détermine le seuil d'un pixel à partir d'une petite région autour de celui-ci
    et permet d'obtenir différents seuils pour différentes régions d'une même image.
    Pour chaque pixel, le seuillage adaptatif examine son voisinage en utilisant une fenêtre de 
    taille (block_size x block_size). La valeur du seuil pour chaque pixel est une somme pondérée 
    par la gaussienne des valeurs de voisinage moins une constante C.
    Si la valeur du pixel est supérieure à la valeur du seuil elle est fixée à la valeur max.
    ==============================================================

    Référence:
    ----------
    - Améliore la qualité d'un pdf : 
      https://medium.com/analytics-vidhya/enhance-a-document-scan-using-python-and-opencv-9934a0c2da3d
      
    - blog parlant de filtre médian et du filtre gaussien mais aussi plein d'autres notions de traitements d'image:
      https://theailearner.com/tag/cv2-medianblur/
      https://theailearner.com/2019/05/06/gaussian-blurring/
      https://theailearner.com/2019/05/07/bilateral-filtering/

      https://ieeexplore.ieee.org/document/9083712

    - pre-processing pour l'OCR :
      https://nanonets.com/blog/ocr-with-tesseract
      
    """
    # Mettre l'image en gris pour appliquer n'importe quel openCV thresholding par la suite 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # retire le bruit (cv2.bilateralFilter est meilleur que cv2.medianBlur mais un peu + lent)
    # im_desoised = cv2.medianBlur(im, 3) #image, kernel size
    im_desoised = cv2.bilateralFilter(src=im, #input image
                                      d=5, #filter size. 
                                      sigmaColor = 15, #Filter sigma in the color space (Range Filter)
                                      sigmaSpace = 15) #Filter sigma in the coordinate space (Domain Filter)
    # Rq: Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications.
    # If the sigma values are small (< 10), the filter will not have much effect, whereas if they are large 
    # (> 150), they will have a very strong effect, making the image look “cartoonish”.
    
    # Appliquation du seuillage sur l'image
    im_thresh = cv2.adaptiveThreshold(im_desoised, #image on which to apply the threshold
                                   255,  # maximum value assigned to pixel values exceeding the threshold
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
                                   cv2.THRESH_BINARY,  # thresholding type
                                   9,  # block size (9x9 window) #essayer aussi 13
                                   10)  # constant (
    
    return im_thresh