"""
Ce fichier permet de définir des transformations générales 
à appliquer sur les données CREDIT pour pouvoir les analyser.

Contient les fonctions :
>> fonctions pour certaines variables (cas particuliers) 
- special_treatment
    
>> fonctions pour les variables dates 
- type_date
- datetime_to_float, 
- float_to_datetime
- sort_without_nan
- interpolate_date
- sort_by_date
- df_datetime_to_float
- df_float_to_datetime

>> fonctions pour les variables réponses 
- compute_DELAI
- compute_NB_DOSS_DAY

>> fonctions pour les variables qualitatives 
- to_Categorical
"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs 
import numpy as np
import pandas as pd 
import datetime as dt
import phik
from phik import resources, report

import warnings
warnings.filterwarnings('ignore')


# ***********************************************************************
# Pré-processing général de certaines variables (cas particuliers)
# ***********************************************************************

def special_treatment(df_credits):
    """
    Cette fonction applique un traitement spécial à certaines variables
    fixées dans la fonction.
    
    Input:
    ------
    - df_credits (dataframe): dataframe crédits contenant les données.
    
    Output:
    ------
    - df (dataframe): dataframe transformée.  
    """
    df = df_credits.copy()
    
    # Suppression des colonnes non nécessaires
    # -----------------------------
    # Suppression de [COETB,LOBJ,COFAAP] car équivalents respectivement à [LIETB,COOBJ,LIBLGG] que l'on garde.
    # COPRO et LIPRLG (id locaux) peuvent être remplacés COPROG et LIPRO (id du groupe).
    # Suppression de LIPRO qui est équivalent à COPROG que l'on garde.
    # 'CODE_INDIC_PRM_ACCS' est utile pour un autre projet dataScience mais inutile ici.
    # -----------------------------
    DEL_CREDIT = ['COETB','COPRO','LIPRLG','LOBJ','COFAAP','LIPRO','CODE_INDIC_PRM_ACCS'] #variables à supprimer
    df.drop(DEL_CREDIT, axis=1,inplace=True)
  
    # Suprpression des lignes ne contenant que des valeurs NaN (il n'y en a pas pour le moment)
    # -----------------------------
    df.dropna(how='all',inplace = True)
  
    # Variable LIBLGG
    # -----------------------------
    # La variable LIBLGG est le nom du prescripteur. Comme il y a trop de modalités,
    # on transforme cette variable avec seulement la présence ou non d'un prescripteur. 
    # Attention, le fait de binariser cette variable crée des duplicates ce qui veut
    # dire qu'un même contrat (variable 'COCO') peut avoir différents prescripteurs. 
    # -----------------------------
    df["LIBLGG"][df["LIBLGG"].notna()] = 1 #présence d'un prescripteur
    df["LIBLGG"][df["LIBLGG"].isna()]  = 0 #valeur NaN donc pas de prescripteur
    
    # Suppression des lignes identiques (il y en a 971 pour le moment)
    # --------------------------------
    # Le fait d'avoir appliqué le traitement sur 'LIBLGG' ci-dessus
    # a créé des duplicates
    # --------------------------------
    df.drop_duplicates(inplace=True)
    
    # Variable CONSCE
    # -----------------------------
    # La variable CONSCE est le nb d'intéractions entre l'agence et le backOffice
    # lorsqu'un dossier n'est pas valide du premier coup. 
    # On peut donc mettre les valeurs NaN à 0 par défaut. 
    # -----------------------------
    df['CONSCE'][df['CONSCE'].isna()] = 0 #### METTRE DANS UNE FONCTION FILL_NA DEDIEE ? #####
    
    
    # Variable TOP_EDITION_BACK_OFFICE
    # -----------------------------
    # La variable TOP_EDITION_BACK_OFFICE indique si le dossier a transité ou non par le 
    # backOffice. Comme on veut prédire un délai pour le backOffice, on enlève toutes les 
    # lignes qui ne sont pas passées par le backOffice.
    # -----------------------------
    # indices des lignes qui ont TOP_EDITION_BACK_OFFICE = 0
    del_idx = df.index[df['TOP_EDITION_BACK_OFFICE'] == 0]
    # on supprime les indices del_idx
    df.drop(axis=0,index=del_idx,inplace=True)
    # on réindexe depuis 0
    df.reset_index(inplace=True,drop=True)
    # supprime la colonne TOP_EDITION_BACK_OFFICE
    df.drop(['TOP_EDITION_BACK_OFFICE'],axis=1,inplace=True)
    
    return df   



# ***********************************************************************
# Pré-processing général des variables dates 
# ***********************************************************************

def type_date(df,var):
    """
    Cette fonction transforme les variables var du dataframe df
    au format pd.Timestamp qui est l'équivalent pandas de 
    Datetime de python et est interchangeable avec lui dans 
    la plupart des cas.     

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variable dates à transformer

    Output:
    -------
    - df (dataframe): dataframe transformé.
    
    Reference:
    ----------
    https://stackoverflow.com/questions/31761047/
    what-difference-between-the-date-time-datetime-and-timestamp-types
    """  
    
    # Transformation au format datetime
    for v in var:
        df[v] = pd.to_datetime(df[v])
    return df


# python 2
def datetime_to_float(d):
    """
    Prend en entrée un timestamp et le transforme en float.
    """
    # '1970-01-01 00:00:00': date de référence: 
    epoch = dt.datetime.utcfromtimestamp(0) 
    # nb de secondes entre la date de référence et la date d (millisecond precision): 
    total_seconds =  (d - epoch).total_seconds() 
    return total_seconds

# python 3
'''def datetime_to_float(d):
    return d.timestamp()''' 

# python 2 & 3
def float_to_datetime(fl):
    """
    Prend en entrée un float et le transforme en timestamp.
    """
    # Transformation des secondes fl en dae (Timestamp)
    return dt.datetime.fromtimestamp(fl)


def sort_without_nan(df,v):
    """
    Cette fonction renvoie les indices des valeurs de la colonne v de la 
    dataframe df triés sans tenir compte des valeurs NaN. Les valeurs NaN 
    resteront donc au même endroit. 
    Ex: [2,np.nan,9,8,7,4] devient [2,np.nan,4,7,8,9]. 
    Un sort classique mettrait toutes les valeurs NaN à la fin.
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variable à trier

    Output:
    ------
    - idx_sorted (np.array): indices des éléments triés 
      sans les NaN. Dans l'exemple ci-dessus, on aurait
      [0,1,5,4,3,2]
    - save_idx (np.array): indices des éléments dans leur
      ordre d'origine. Dans l'exemple ci-dessus, on aurait
      [0,1,2,3,4,5]
    """
    #--- indices de df[v]
    save_idx = df[v].index
    #--- indices des éléments NaN dans df[v]
    idx_na=df[v][df[v].isna()].index
    #--- indices des éléments non NaN dans df[v]
    idx_no_na=df[v][df[v].notna()].index
    #--- indices des éléments non NaN **triés** dans df[v]
    # (on trie sans les NaN car sinon ils se retrouvent tous à la fin du tri)
    idx_sorted=df[v][idx_no_na].sort_values().index
    #--- On insère dans idx_sorted les indices idx_na aux 
    # localisations donnés idx_na. Ceci revient à avoir trié le tableau
    # sans avoir tenu compte des NaN que l'on a laissé au même endroit.
    for i in range(len(idx_na)):
        idx_sorted = np.insert(arr=idx_sorted,obj=idx_na[i],values=idx_na[i])
    return idx_sorted,save_idx



def interpolate_date(df,var):
    """
    Cette fonction interpole les dates de façon linéaire. Les dates sont 
    au préalable triées (sans tenir compte des nan) avec la fonction sort_without_nan.
    L'interpolation donne des dates précises commme : '2022-12-18 12:00:00'
    Or, nous n'avons pas besoin d'une telle précision donc nous 
    simplifierons en '2022-12-18'. 
    Avant d'appliquer cette fonction: 
        * Appliquer la fonction type_date.
        * S'assurer que les index commencent bien à 0 et que le pas est de 1
          (sinon crée une erreur)
    On utilise les fonctions datetime_to_float et float_to_datetime. 
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (liste): liste de(s) variable(s) dates à interpoler. 

    Output:
    ------
    - df (dataframe): dataframe transformé.
    
    Reference:
    ----------
    https://github.com/pandas-dev/pandas/issues/11312
    """
    
    for v in var:
        # Transformation du type timestamp au type float.
        s2 = df[v].apply(lambda x: datetime_to_float(x))
        # Avant d'interpoler, on récupère les indices des valeurs triées sans les NaN 
        idx_sorted,save_idx = sort_without_nan(df,v)
        # On interpole les floats de façon linéaire  
        df_interp = s2[idx_sorted].interpolate(method='linear') 
        # On remet dans l'ordre d'origine
        df_interp = df_interp[save_idx]
        # On transforme à nouveau le floats en timestamp
        df[v] = df_interp.apply(lambda x: float_to_datetime(x))
        # Réduit la précision au jour (et non pas à l'heure). Ceci met au format datetime 
        df[v] = df[v].apply(lambda x: x.date())
        # Remet met le format datetime au format timestamp
        df[v] = pd.to_datetime(df[v])

    return(df)



def sort_by_date(df):
    """
    Une fois que l'on a interpolé les dates avec la fonction interpolate_date,
    on peut trier df par 'DATEDI' croissant. Cela permet d'éviter des erreurs
    par la suite (notamment lorsque l'on va calculer le nb de 'CODOSB' par 
    jour ou par semaine).  
        
    Input:
    ------
    - df (dataframe): dataframe contenant les données.

    Output:
    ------
    - df (dataframe): dataframe trié par 'DATEDI' croissant.
    """
    # Une fois la date interpolée, on peut trier par 'DATEDI' croissant
    df.sort_values(by=['DATEDI'],axis=0,inplace=True)
    # On réordonne les index
    df.reset_index(inplace=True,drop=True)
    
    return df



def df_datetime_to_float(df,var):
    """
    Cette fonction transforme les variables dates du 
    dataframe df du format datetime au format float.
    Utilise la fonction datetime_to_float. 
    Le dataframe df ne doit pas contenir de NaN.
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variable dates à transformer

    Output:
    ------
    - df (dataframe): dataframe transformé.
    """
    # Parcourt des variables 
    for v in var:
        df[v] = df[v].apply(lambda x: datetime_to_float(x))
    return df



def df_float_to_datetime(df,var):
    """
    Cette fonction transforme les variables dates du 
    dataframe df du format float au format datetime.
    Utilise la fonction float_to_datetime. 
    Le dataframe df ne doit pas contenir de NaN.
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variable dates à transformer

    Output:
    ------
    - df (dataframe): dataframe transformé.
    """
    # Parcourt des variables 
    for v in var:
        df[v] = df[v].apply(lambda x: float_to_datetime(x))
    return df


# ***********************************************************************
# Ajout des variables réponses
# ***********************************************************************

def compute_DELAI(df):
    """
    Cette fonction calcule la variable réponse 'DELAI'. 
    C'est le délai (en jours) entre l'arrivée du crédit
    au backOffice (variable 'DATEDI') et l'accord du client (variable 
    'DTINS'). Ainsi, 'DELAI' = 'DATEDI' - 'DTINS'.
    A utiliser après la fonction type_date.
        
    Input:
    ------
    - df (dataframe): dataframe contenant les données.

    Output:
    ------
    - df (dataframe): dataframe augmenté avec la variable 'DELAI'
    """
    # Différence 'DATEDI'-'DTINS'
    delta = df['DATEDI']-df['DTINS']
    # delta est du type timedelta64[ns] et on le transforme en nb de jours
    delta = (delta/np.timedelta64(1, 'D')).astype(int)
    # On met 0 aux endroits où DATEDI'-'DTINS' < 0 (pour le moment il y en a 163)
    delta[delta<0] = 0 
    # On crée une nouvelle colonne 'DELAI' contenant le nombre de jours
    # entre 'DATEDI' et 'DTINS'
    df['DELAI'] = delta.values
    
    return df



def compute_NB_DOSS_DAY(df):
    """
    Cette fonction calcule la variable réponse 'NB_DOSS_DAY'.
    'NB_DOSS_DAY' est le nombre de dossiers qui arrivent chaque semaine 
    au backOffice. Pour cela, on compte le nombre de 'CODOSB' par jour. 
    A utiliser après la fonction type_date.
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.

    Output:
    ------
    - df (dataframe): dataframe augmenté avec la variable 'NB_DOSS_DAY'
    """ 
    # Dataframe ayant en indice le jour ('DATEDI') et en valeur le nb de 'CODOSB' par jour
    df_CODOSB_day = df.groupby([df['DATEDI'].dt.date]).count()['CODOSB']
    # Convertit les indices au format datetime64 pour pouvoir faire le merge avec la 
    # variable 'DATEDI' de df qui est au format datetime64.
    df_CODOSB_day.index = pd.to_datetime(df_CODOSB_day.index)
    # Merge le nb de 'CODOSB' avec df. 
    # On obtient df avec une colonne en plus qui est le nb de 'CODOSB' par jour.
    df = df.merge(df_CODOSB_day, on='DATEDI')
    # On renomme les colonnes 
    df.rename(columns={"CODOSB_x": "CODOSB", "CODOSB_y": "NB_DOSS_DAY"},inplace=True)
    
    return df



# ***********************************************************************
# Mettre les variables qualitatives au format categorical.
# ***********************************************************************

def to_Categorical(df,var):
    """
    Cette fonction transforme les variables qualitatives var 
    de la dataframe df et les met aui format pd.Categorical
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables quanlitatives à transformer
      en pd.Categorical.

    Output:
    ------
    - df (dataframe): dataframe transformé
    """
    credits_cat = df.copy()
    for v in var: 
        credits_cat[v] = pd.Categorical(df[v],ordered=False)
    return credits_cat