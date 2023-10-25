"""
Ce fichier permet de définir les variables globales qui seront 
utilisées dans tous les autres fichiers.py.  

On définit également les variables globales nécessaires pour la résolution
du problème dénommé "task1". Ce problème vise à prédire la variable 
'NB_DOSS_DAY' c'est-à-dire le nombre de dossiers qui arrivent au 
backOffice par jour ou par semaine.

Contient les variables:
>> variables globales (task1 et task2)
- VAR_QUANTI_CREDIT
- VAR_DATE_CREDIT
- VAR_QUALI_CREDIT
- VAR_QUALI_CREDIT_S
- VAR_QUALI_CREDIT_B

>> variables pour task1
- VAR_REP_1
- VAR_DATE_CREDIT_1
- VAR_QUANTI_CREDIT_1
- VAR_QUALI_CREDIT_1
- VAR_QUANTI_CREDIT_1_f
- VAR_QUANTI_CREDIT_1_i
"""

# ***********************************************************************
# Définition des variables globales des données CREDIT
# ***********************************************************************
# variables quantitatives
VAR_QUANTI_CREDIT = ['MEDOS','TAINT','TITEGI','QLDDOS','NBASSGPE','NBASSEXT','NBGAR','DELINS','DELDEC','DELEDI','CONSCE']
# variables dates
VAR_DATE_CREDIT = ['DATEDI','DTEDI','DDDOSP','DTINS','DATDEC','DTDEC']
# variables qualitatives 
VAR_QUALI_CREDIT = ['LIETB','CODOSB','COCO','COPROG','COOBJ','TOPPSC','COPOST','COEMINS','COEMDEC','COEMEDI','LIBLGG']


# Variables qualitatives qui ont peu de modalités (small)
VAR_QUALI_CREDIT_S = ['LIETB','COPROG','COOBJ','TOPPSC','LIBLGG']
# Variables qualitatives qui ont beaucoup de modalités (big)
VAR_QUALI_CREDIT_B = list((set(VAR_QUALI_CREDIT).difference(VAR_QUALI_CREDIT_S)))


# ***********************************************************************
# Définition des variables pour task1 
# Ces variables ont été sélectionnées à partir de l'analyse exploratoire. 
# ***********************************************************************

# Variables qualitatives, quantitatives et dates 
# -------------------------------------
# variable réponse
VAR_REP_1 = ['NB_DOSS_DAY']
# variable date
VAR_DATE_CREDIT_1 = ['DATEDI']
# variables quantitatives 
VAR_QUANTI_CREDIT_1 = ['MEDOS','TAINT','TITEGI','QLDDOS','NBASSGPE','NBASSEXT','NBGAR','DELINS','DELDEC','DELEDI','CONSCE']
# variables qualitatives 
VAR_QUALI_CREDIT_1 = ['COPROG','COOBJ','TOPPSC','LIBLGG']


# Séparation des variables quantitative float et quantative integer
# -------------------------------------
# variables quantitatives réelles (float)
VAR_QUANTI_CREDIT_1_f = ['MEDOS','TAINT','TITEGI'] 
# variables quantitatives entières (integer)
VAR_QUANTI_CREDIT_1_i = ['QLDDOS','NBASSGPE','NBASSEXT','NBGAR','DELINS','DELDEC','DELEDI','CONSCE'] 
