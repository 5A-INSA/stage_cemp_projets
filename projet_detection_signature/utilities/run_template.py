# ******************************************************
# Importation des librairies nécessaires
# ******************************************************
import argparse
import os
import template

# ******************************************************
# Définition du parser
# ******************************************************
parser = argparse.ArgumentParser(
    prog='template',

    description="Ce module permet de définir le template pour la détection de signature. \
    Ce module demande à l'utilisateur de choisir un pdf servant de template et de définir  \
    à la main le numéro de page contenant la signature.  \
    \
    L'utilisateur doit par la suite saisir une liste de mots clés keyWords_template.  \
    Ces mots doivent être présents sur la page contenant la signature. \
    \
    Enfin, l'utilisateur doit dessiner sur la page du pdf template, les rectangles supposés \
    contenir la signature.  \
     \
    La fonction calcule alors les coordonnées de ces rectangles par rapport à la position des mots \
    clés de keyWords_template renvoyée par l'OCR (Tesseract). Ces coordonnées relatives sont renvoyées \
    sous le nom de la variable list_relative_points. \
    \
    ATTENTION, l'image template doit être 'parfaite' c'est-à-dire que tous les mots clés \
    keyWords_template doivent être présents et lisibles par Tesseract dans la page du \
    pdf template. Sinon, une erreur peut se produire. \
    Dans ce cas, une solution est de changer l'image d'entrée ou de modifier la liste \
    des mots clés. \
    \
      /.\ Règles d'or pour le choix de keyWords_template : \
      - Chaque élément de la liste keyWords_template peut être un seul mot ou une expression\
      - Tous les mots de la liste dovent être présent dans la page\
      - Chaque mot de la liste ne doit être présent qu'une seule fois dans la page.",


    epilog="Utilise les fonctions de template.py :  \
    - read_page() \
    - get_screenSize() \
    - ResizeWithAspectRatio()  \
    - extract_start_rectangle() \
    - display_instructions() \
    - create_template() \
    - find_expression_in_list() \
    - find_all_keyWord_coord() \
    - compute_relative_coord() \
    - main_template() \
    - Pour d'avantage d'information sur ces fonctions, ce référer à leur documentation."
)


parser.add_argument("--path_keyWords_template", type=str, 
                    help='Chemin du fichier .txt contenant keyWords_template. keyWords_template est une liste \
                    contenant les mots clés à détecter dans la page du pdf template. \
                    Ex: "C:/Users/A1234/directory/keyWords_template.txt.', 
                    default=os.getcwd())


parser.add_argument("--path_template", type=str, 
                    help='Chemin du répertoire contenant le pdf template. Ex: "C:/Users/A1234/directory"')

parser.add_argument("--name_template", type=str, 
                    help='Nom du pdf template. Ex: "monpdf.pdf"')

parser.add_argument("--page_template", type=int, 
                    help="Page du pdf template contenant la signature. \
                    Attention, les pages commencent à l'indice 0. \
                    Avec page_template=-1, la dernière page du pdf est renvoyée. ")

parser.add_argument("--path_save_result", type=str, 
                    help="Chemin du répertoire vers lequel sauvegarder la sortie de cette fonction. \
                    Si path_save=None, rien n'est sauvegardé.",
                    default=os.getcwd())

parser.add_argument("--debug", type=int, choices = [0,1], 
                    help="Si debug=1, des images et du texte sont affichés \
                    aux différentes étapes pour mieux visualiser ce que fait la fonction",
                    default = 0)

parser.add_argument("--show_instructions", type=int, choices = [0,1],  
                    help="si show_instructions=1, une fenêtre contenant les instructions est affichée \
                    lorsque l'utilisateur doit saisir le template.",
                    default = 1)

parser.add_argument("--windowHeight", type=int, 
                    help="Hauteur de l'image à afficher à l'écran. \
                    Si les deux arguments windowHeight et windowWidth sont à None, la taille de l'image \
                    est calculée pour s'accomoder à la taille de l'écran d'ordinateur.",
                    default = None)

parser.add_argument("--windowWidth", type=int, 
                    help="Largeur de l'image à afficher à l'écran. \
                    Si les deux arguments windowHeight et windowWidth sont à None, la taille de l'image \
                    est calculée pour s'accomoder à la taille de l'écran d'ordinateur.",
                    default = None)


args = parser.parse_args()


# ******************************************************
# Récupération de keyWords_template
# ******************************************************
with open (args.path_keyWords_template, encoding="utf-8") as f:
    keyWords_template_file = f.readlines()  
    
keyWords_template = [word.strip('\n') for word in keyWords_template_file]

# ******************************************************
# Appel de la fonction main_template() avec les arguments définis dans le parser ci-dessus
# ******************************************************
list_relative_points, template_shape, template_text = template.main_template(
      path_template = args.path_template,
      name_template = args.name_template,
      page_template = args.page_template, 
      path_save_result = args.path_save_result,
      keyWords_template = keyWords_template,
      debug = args.debug,
      show_instructions = args.show_instructions,
      windowHeight = args.windowHeight, 
      windowWidth = args.windowWidth)

# ******************************************************
# Pour lancer ce code depuis le terminal :
# ****************************************************** 
# conda activate pdf
# cd "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\Python\utilities"
# python run_template.py --path_keyWords_template "C:\Users\A3193307\Groupe BPCE\CEMP - Data & Décisionnel - Data Science\Analyse Documents\Python\keyWords_template_lea.txt" --path_template "C:/Users/A3193307/Groupe BPCE/CEMP - Data & Décisionnel - Data Science/Analyse Documents/Python" --name_template "template_lea.pdf" --page_template -1 --path_save_result "C:/Users/A3193307/Groupe BPCE/CEMP - Data & Décisionnel - Data Science/Analyse Documents/Python/utilities"