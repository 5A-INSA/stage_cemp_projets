"""
Ce fichier permet de définir les chemins absolus qui seront 
utilisées dans tous les autres fichiers.py.

Pour adapter les chemins à un autre ordinateur, il suffira de modifer
uniquement ce fichier, notamment la variable PATH.     

Contient les variables 
- PATH
- PATH_DATA
- PATH_UTILS
- PATH_PLOTS
"""

# Définition des chemins abolus
# ------------------------------------
# Chemin du répertoire de travail (ajouter le double \\)
PATH = "C:\\Users\\A3193307\\Groupe BPCE\\CEMP - Data & Décisionnel - Data Science\\Back office Crédit (Projet DMO)\\Back Office Credit version 4"
# Chemin des données
PATH_DATA = PATH + "/data"
# Chemin des scripts python .py
PATH_UTILS =  PATH + "/utilities"
# Chemin du dossier où sauvegarder les plots
PATH_PLOTS = PATH + "/plots"