# ******************************************************
# Importation des librairies nécessaires
# ******************************************************
import argparse
import os
import detectSignature

# ******************************************************
# Définition du parser
# ******************************************************
parser = argparse.ArgumentParser(
    prog='detectSignature',

    description="Ce module permet de verifier si un pdf est signé ou non. \
        Au préalable, l'utilisateur a du définir un template avec le module template.\
        Ainsi, la liste de mots clés keyWords_template ainsi que la variable \
        list_relative_points ont été calculés avec le module template.  \
        Pour rappel, list_relative_points représente les coordonnées relatives des rectangles par \
        rapport aux mots de keyWords_template sur le pdf template. \
        \
        Ce module utilise ensuite le texte template_text renvoyé par le module template \
        et l'OCR Tesseract pour lire les pages du pdf à analyser et trouver la page \
        dont le texte est le plus similaire à la page du pdf template. La page trouvée \
        est censée être la page contenant la signature. \
        \
        Une fois la page du pdf contenant la signature identifiée, ce module extrait les zones du pdf à analyser \
        supposées contenir la signature. \
        \
        Pour extraire ces zones, l'algorithme utilise la variable list_relative_points \
        pour obtenir les coordonnées des rectangles à extraire, en fonction de la position \
        des mots de keyWords_template du pdf à analyser. \
        Le fait de calculer la position relative des mots de keyWords_template par rapport  \
        aux rectangles permet d'ajuster la position des rectangles à chaque pdf à analyser. \
        \
        Enfin, une fois les zones supposées contenir la signature extraites, un pourcentage de  \
        pixels continus est calculé pour déterminer si le pourcentage de remplissage de la zone \
        est suffisant. Si le remplissage est suffisant sur l'une des zones extraites, \
        le pdf est considéré comme signé. ",

    epilog="Utilise les fonctions de detectSignature.py :  \
        - compute_strings_similarity() \
        - find_page_with_Tesseract() \
        - find_expression_in_list()\
        - find_1_keyWord_coord() \
        - extract_signature_boxes() \
        - verify_signature() \
        - drawProgressBar() \
        - main_detectSignature() \
        - Pour d'avantage d'information sur ces fonctions, ce référer à leur documentation."
)


parser.add_argument("--path_data", type=str, 
                    help='Chemin du répertoire contenant les pdf à analyser. Ex: "C:/Users/A1234/data" ')

parser.add_argument("--path_name_pdf", type=str, 
                    help='Chemin du fichier .txt contenant le nom des pdfs à analyser.\
                    Si path_name_pdf=None, tout le répertoire indiqué par path_data est analysé.\
                    Si on renseigne la variable path_name_pdf, la variable name_pdf sera ignorée.',
                    default=None)

parser.add_argument("--name_pdf", type=str, 
                    help='Nom du pdf à analyser. Ex: "monpdf.pdf". \
                    Si name_pdf=None, tout le répertoire indiqué par path_data est analysé.\
                    Cette variable sera utilisée si path_name_pdf=None et ignorée si path_name_pdf\
                    est renseignée.',
                    default=None)

parser.add_argument("--path_keyWords_template", type=str, 
                    help='Chemin du fichier .txt contenant keyWords_template. keyWords_template est une liste \
                    contenant les mots clés à détecter dans la page du pdf template. \
                    Ex: "C:/Users/A1234/directory/keyWords_template.txt. \
                    Cette liste doit être identique à la liste keyWords_template sélectionnée dans le template.', 
                    default=os.getcwd())

parser.add_argument("--path_saved_template", type=str, 
                    help='Chemin du répertoire contenant la sauvegarde du template saved_template.pickle, la  \
                    shape du template template_shape.txt et le texte du pdf template template_text \
                    préalablement créés par le module template.', 
                    default=os.getcwd())

parser.add_argument("--similarity_thresh", type=float,
                    help= "seuil de similarité entre 2 textes pour la fonction \
                    find_page_with_Tesseract(). Si la similarité calculée est en dessous de ce seuil, \
                    alors les deux textes ne sont pas similaires",
                    default=0.5)

parser.add_argument("--start_with", type=str,
                    help= "Peut prendre les valeurs 'begin' et 'end' et permet d'indiquer \
                    si on commence la recherche de la page contenant la signature en partant du début du \
                    pdf (avec start_with='begin') ou de la fin du pdf (avec start_with='end')",
                    default="end")     

parser.add_argument("--path_save_result", type=str, 
                    help="Chemin du fichier .csv vers lequel sauvegarder la sortie de cette fonction. \
                    Ex: path_save_result = 'C:/Users/A1234/result/mon_resultat.csv'. \
                    Si on a plus de 10 pdfs à analyser, la sauvegarde se fait tous les 10 pdfs. \
                    Si path_save_result=None, rien n'est sauvegardé.  \
                    Si le fichier indiqué par path_save_result existe déjà, alors le code lit ce \
                    fichier et reprend l'analyse des documents là où elle s'était précédemment arrêtée. \
                    Pour savoir à quel pdfs on s'était arrêté lors de l'analyse précédente, on lit le \
                    fichier indiqué par path_save_result et on regarde quels documents du répertoire path_data \
                    (ou quels document indiqués par name_pdf) ne sont  pas dans le fichier .csv. \
                    On analyse ces documents. ",
                    default=None)


parser.add_argument("--path_save_imsig", type=str, 
                    help=" chemin du répertoire où sauvegarder : \
                    1) L'image du cadre supposé contenir la signature (avant Cropper). \
                        L'image est sauvegardée dans le répertoire path_save_imsig/before_cropper. \
                    2) L'image de la signature extraite par le Cropper dans le cadre supposé contenir  \
                        la signature (après le Cropper).  \
                        L'image est sauvegardée dans le répertoire path_save_imsig/after_cropper.\
                    Ex : path_save_imsig='C:\A0000\resultats_signatures' et path_save_imsig contient les \
                    sous-répertoire before_cropper et after_cropper. \
                    Si le répertoire path_save_imsig, before_cropper et after_cropper n'existent pas déjà, \
                    il sont créés. Si path_save_sig=None, aucune image n'est sauvegardée.  \
                    Le nom des images est : 'nomDuPDF_bci.jpg' où dans 'bci', 'i' est le numéro du cadre \
                    supposé contenir la signature (il peut y avoir plusieurs cadres) et 'bc' indique \
                    que c'est une image 'before cropper'. 'nomDuPDF_aci.jpg' indique une image 'after cropper'. ",
                    default=None)


parser.add_argument("--debug", type=int, choices = [0,1], 
                    help="Si debug=1, des images et du texte sont affichés \
                    aux différentes étapes pour mieux visualiser ce que fait la fonction. \
                    Lorsque qu'il y a trop d'images à analyser, il est déconseillé de mettre \
                    debug=1 pour ne pas surcharger l'affichage.",
                    default = 0)

args = parser.parse_args()


# ******************************************************
# Valeurs par défault
# ******************************************************
"""
cropper_params (dict) :
Contient les paramètres du Cropper pour la fonction verify_signature(). 
"""
cropper_params={"min_region_size":1e2,"border_ratio":0.01}

"""
judger_params (dict) :
Contient les paramètres du Judger pour la fonction verify_signature(). 
"""
#judger_params={"size_ratio":[1,4], "pixel_ratio": [0.001,1]}
judger_params={"pixel_ratio": [0.001,1]}

# ******************************************************
# Récupération du nom des pdfs à analyser si path_name_pdf!=None
# ******************************************************
if args.path_name_pdf: #si path_name_pdf!=None, on va chercher name_pdf dans un fichier .txt
    with open (args.path_name_pdf, encoding="utf-8") as f:
        name_pdf_file = f.readlines()
    name_pdf = [word.strip('\n') for word in name_pdf_file]
else: #sinon, name_pdf est celui donné en argument.
    name_pdf = args.name_pdf

# ******************************************************
# Récupération de keyWords_template
# ******************************************************
with open (args.path_keyWords_template, encoding="utf-8") as f:
    keyWords_template_file = f.readlines()  
    
keyWords_template = [word.strip('\n') for word in keyWords_template_file]


# ******************************************************
# "Récupération" de saved_template, template_shape et template_text
# ******************************************************
"""
saved_template (liste de dictionnaire) :
saved_template=list_relative_points. Liste de dictionnaires
où chaque dictionnaire représente les coordonnées d'un rectangle par rapport aux différents mots 
clés de keyWords_template. Si saved_template=None, alors le template sera récupéré dans le 
répertoire indiqué par path_saved_template à l'aide du fichier saved_template.pickle.
"""
saved_template=None

"""
template_shape (tuple) :
(height,width) ou (height,width,depth) taille de l'image
ayant servi de template. Si template_shape=None, alors le template sera récupéré dans le répertoire
indiqué par path_saved_template.
"""
template_shape=None

"""
template_text (string): $
texte contenu dans la page du pdf template. Si template_text=None,
alors le texte sera récupéré dans le répertoire indiqué par path_saved_template.
"""
template_text=None

# ******************************************************
# Appel de la fonction main_template() avec les arguments définis dans le parser ci-dessus
# ******************************************************
result = detectSignature.main_detectSignature(
    path_data = args.path_data,
    keyWords_template = keyWords_template,
    path_save_result= args.path_save_result,
    path_save_imsig = args.path_save_imsig,
    name_pdf = name_pdf, 
    saved_template = saved_template, #list_relative_points
    template_shape = template_shape, 
    similarity_thresh = args.similarity_thresh,
    start_with = args.start_with,
    template_text = template_text,
    path_saved_template = args.path_saved_template,
    debug = args.debug,
    cropper_params = cropper_params,
    judger_params = judger_params)


# ******************************************************
# Pour lancer ce code depuis le terminal :
# ****************************************************** 
# conda activate pdf
# cd "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\Python\utilities"
# python run_detectSignature.py --path_data "C:\Users\A3193307\projet_signature\All_LEA_100" --path_keyWords_template "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\Python\keyWords_template_lea.txt" --path_saved_template "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\Python\utilities" --similarity_thresh 0.5 --start_with "end" --path_save_result "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\result\result3.csv" --debug 0