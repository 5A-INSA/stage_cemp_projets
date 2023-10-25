"""
A quoi sert ce module ?
    Admettons que nous ayons fait tourner le module detectSignature.py (avec run_detectSignature.py) 
    sur nos données avec un certain template défini au préalable par le module template.py (avec run_template.py).
    Nous avons sauvegardé le résultat dans un fichier .csv contenant les colonnes "file_name" (pour le nom des fichiers
    pdfs) et "result" (pour le résultat de la classification).
    Dans les résultat obtenus, nous lisons que certains pdfs ont été classés avec la mention
    "Page contenant la signature non detectee" dans la colonne "result". 
    Ceci peut être du au fait que ces pdfs ne correspondent pas au 1er template défini mais à un 2e template.
    Nous souhaitons donc faire utiliser run_detectSignature.py sur ces pdfs avec un 2e template.
    
    Pour répondre à la problématique ci-dessus, nous utilisons tout d'abord la fonction retrieve_error_files()
    pour trouver le nom des pdfs de la colonne "file_name" contenant la mention "Page contenant la signature 
    non detectee" dans la colonne "result". Pour ce faire, nous utilisons en argument col1="file_name", 
    col2="result" et to_search="Page contenant la signature non detectee". 
    Nous obtenons donc un fichier .txt avec les résultats de la recherche, par exemple intitulé "round2_files.txt".
    
    Ensuite, nous sélectionnons un 2e template avec run_template.py et nous faisons tourner run_detectSignature.py
    sur les pdfs indiqués dans le fichier "round2_files.txt" en mettant comme valeur pour la variable "path_name_pdf" 
    le chemin vers "round2_files.txt". 
"""

# ******************************************************
# Importation des librairies nécessaires
# ******************************************************
import argparse
import pandas as pd
import numpy as np


# ******************************************************
# Fonction retrieve_error_files()
# ******************************************************
def retrieve_error_files(csv_file,col1,col2,to_search,text_file):
    """
    Cette fonction prend en argument la table de résultats au format .csv indiquée par csv_file. 
    La table csv_file doit contenir au moins les colonnes col1 et col2 mais peut contenir d'autrces colonnes.
    La fonction crée en sortie un fichier .txt à l'endroit indiqué par text_file. 
    Le fichier .txt de sortie contient les éléments de col1 qui correspondant à la valeur to_search
    dans col2.
    Ex: csv_file = pd.DataFrame({'col1': ['file1', 'file2', 'file3', 'file4'],
                                 'col2': ['ok'   , 'ko'   ,    'ok', 'ko'   ]})
    si to_search='ko' alors text_file='file2 \n file3'.
    
    ========================================= Notes =========================================
    >>> A quoi sert cette fonction ?
    Admettons que nous ayons fait tourner le module detectSignature.py (avec run_detectSignature.py) 
    sur nos données avec un certain template défini au préalable par le module template.py (avec run_template.py).
    Nous avons sauvegardé le résultat dans un fichier .csv contenant les colonnes "file_name" (pour le nom des fichiers
    pdfs) et "result" (pour le résultat de la classification).
    Dans les résultat obtenus, nous lisons que certains pdfs ont été classés avec la mention
    "Page contenant la signature non detectee" dans la colonne "result". 
    Ceci peut être du au fait que ces pdfs ne correspondent pas au 1er template défini mais à un 2e template.
    Nous souhaitons donc faire utiliser run_detectSignature.py sur ces pdfs avec un 2e template.
    
    Pour répondre à la problématique ci-dessus, nous utilisons tout d'abord cette fonction pour trouver
    le nom des pdfs de la colonne "file_name" contenant la mention "Page contenant la signature non detectee" 
    dans la colonne "result". Pour ce faire, nous utilisons en argument col1="file_name", col2="result" et 
    to_search="Page contenant la signature non detectee". 
    Nous obtenons donc un fichier .txt avec les résultats de la recherche, par exemple intitulé "round2_files.txt".
    
    Ensuite, nous sélectionnons un 2e template avec run_template.py et nous faisons tourner run_detectSignature.py
    sur les pdfs indiqués dans le fichier "round2_files.txt" en mettant comme valeur pour la variable "path_name_pdf" 
    le chemin vers "round2_files.txt". 
    =========================================================================================
    
    Input:
    ------
    - csv_file (string) : chemin vers le fichier .csv pour lequel on veut lire le contenu.
      Ex : "C:/Users/A00000/result/my_file.csv"
      Le fichier .csv doit contenir au moins les colonnes indiquées par col1 et col2.
    - col1,col2 (string) : noms des colonnes du fichier .csv.
      Cette fonction recherche les éléments de col1 qui correspondant à la valeur to_search dans col2.
    - to_search (string) : string indiquant quelle valeur rechercher dans col_2
    - text_file (string) : chemin du fichier .txt contenant les résultats de la recherche. 
    
    Output:
    ------
    - match_list (list of string) : liste des éléments de col1 qui correspondant à la valeur 
      to_search dans col2.
    - enregistrement du fichier .txt à l'endroit indiqué par text_file.
    """
    # Lecture du fichier csv
    df_csv = pd.read_csv(csv_file) 
    # Recherche des éléments de col1 qui correspondant à la valeur to_search dans col2.
    match_list  = list(df_csv[col1].iloc[np.where(df_csv[col2]==to_search)].values)
    # Sauvegarde du résultat dans un fichier .txt
    with open(text_file, 'w', encoding="utf-8") as f:
        for item in match_list: # write each item on a new line
            f.write("%s\n" % item)
    return match_list


# ******************************************************
# Définition du parser
# ******************************************************
parser = argparse.ArgumentParser(
    prog='retrieveErrorFiles',

    description="A quoi sert ce module ? \
    Admettons que nous ayons fait tourner le module detectSignature.py (avec run_detectSignature.py) \
    sur nos données avec un certain template défini au préalable par le module template.py (avec run_template.py). \
    Nous avons sauvegardé le résultat dans un fichier .csv contenant les colonnes 'file_name' \
    (pour le nom des fichiers pdfs) et 'result' (pour le résultat de la classification). \
    Dans les résultat obtenus, nous lisons que certains pdfs ont été classés avec la mention \
    'Page contenant la signature non detectee' dans la colonne 'result'. Ceci peut être du au fait \
    que ces pdfs ne correspondent pas au premier template défini mais à un \
    deuxième template.  Nous souhaitons donc faire utiliser run_detectSignature.py sur ces pdfs \
    avec un deuxième template. \
    \
    Pour répondre à la problématique ci-dessus, nous utilisons tout d'abord la fonction retrieve_error_files() \
    pour trouver le nom des pdfs de la colonne 'file_name' contenant la mention 'Page contenant la signature  \
    non detectee' dans la colonne 'result'. Pour ce faire, nous utilisons en argument col1='file_name', \
    col2='result' et to_search='Page contenant la signature non detectee'. \
    Nous obtenons donc un fichier .txt avec les résultats de la recherche, \
    par exemple intitulé 'round2_files.txt'.\
    \
    Ensuite, nous sélectionnons un 2e template avec run_template.py et nous faisons tourner run_detectSignature.py \
    sur les pdfs indiqués dans le fichier 'round2_files'.txt en mettant comme valeur pour la variable \
    'path_name_pdf' le chemin vers 'round2_files.txt'."  ,

    epilog="Utilise les fonctions :  \
        - retrieve_error_files() "
)


parser.add_argument("--path_csv_file", type=str, 
                    help='Chemin vers le fichier .csv pour lequel on veut lire le contenu. \
                    Ex : "C:/Users/A00000/result/my_file.csv"')

parser.add_argument("--col1", type=str, 
                    help='Nom de la colonne du fichier .csv dont on enregistrera certaines valeurs.\
                    On recherche les éléments de col1 qui correspondant à la valeur \
                    to_search dans col2.')

parser.add_argument("--col2", type=str, 
                    help='Nom de la colonne du fichier .csv dans laquelle on recherche. \
                    On recherche les éléments de col1 qui correspondant à la valeur \
                    to_search dans col2.' )

parser.add_argument("--to_search", type=str, 
                    help='String indiquant quelle valeur rechercher dans col_2.')

parser.add_argument("--path_text_file", type=str, 
                    help='Chemin du fichier .txt contenant les résultats de la recherche.\
                    Ex : "C:/Users/A00000/result/round2_files.txt"')

args = parser.parse_args()

# ******************************************************
# Appel de la fonction retrieve_error_files() avec les arguments définis dans le parser ci-dessus
# ******************************************************
match_list = retrieve_error_files(
    csv_file = args.path_csv_file,
    col1 = args.col1,
    col2 = args.col2,
    to_search = args.to_search,
    text_file = args.path_text_file
)