"""
Cette fichier contient les fonctions nécessaires au pré-processing 
des données CREDIT pour répondre au problème "task1".

Le problème dénommé "task1" cherche à prédire la variable 'NB_DOSS_DAY'
c'est-à-dire le nombre de dossiers qui arrivent au backOffice 
par jour (si on regroupe les données au jour) ou par semaine,
(si on regroupe les données à la semaine). 
Pour prédire 'NB_DOSS_DAY', on considère une approche par série temporelle
multivariée ('NB_DOSS_DAY' + variables explicatives) ou univariée 
(uniquement 'NB_DOSS_DAY').

Contient les fonctions:
- fill_na_1
- create_MTS_day
- create_TS
- plot_prediction
- transform_TS
- transform_inv_TS
- compute_rmse
- compute_mae
"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs 
import numpy as np
import numpy.linalg as npl
import pandas as pd 
import datetime as dt
from collections import Counter
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Graphes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings('ignore')



# ***********************************************************************
# Imputation des données pour "task1"
# ***********************************************************************

def fill_na_1(df):
    """
    Cette fonction remplit les valeurs manquantes pour le cas de la prédiction de NB_DOSS_DAY    
    selon les régles décrites ci-dessous : 
        *'NBGAR': toutes les variables manquantes sont mises à 0.
        *'COOBJ': toutes les variables manquantes sont imputées par la modalité majoritaire.
        *'DELINS' = 'DTINS'-'DDD0SP', où les valeurs manquantes de 'DTINS' ont été au préalable interpolées. 
        *'DELDEC' = 'DTDEC'-'DATDEC', où les valeurs manquantes de 'DATDEC' ont été au préalable interpolées. 
        * 'NBASSGPE' et 'NBASSEXT' (valeurs manquantes aux mêmes endroits). Pour un dossier dont l'assurance 
          est manquante, on tire aléatoirement (avec proba 1/2) une assurance entre NBASSGPE et NBASSEXT que 
          que l'on met à 0. Pour la variable non tiée, on remplit par la valeur moyenne.
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.

    Output:
    ------
    - df (dataframe): dataframe avec données manquantes imputées.
    """ 
    
    # Variable NBGAR
    # --------------------------------------
    df['NBGAR'][df['NBGAR'].isna()] = 0

    # Variable COOBJ
    # --------------------------------------
    # indices des valeurs manquantes pour la variable 'COOBJ'
    idx_na = np.where(df['COOBJ'].isna())[0]
    # dictionnaire contenant en key, la modalité et en value, son occurrence dans df
    d = Counter(df['COOBJ'][df['COOBJ'].notna()].values)
    # modalité majoritaire 
    majority_class = d.most_common(1)[0][0]
    # imputation par la classe majoritaire 
    df['COOBJ'].iloc[idx_na] = majority_class

    # Variable DELINS
    # --------------------------------------
    # indices des valeurs manquantes de 'DELINS'
    idx_na = np.where(df['DELINS'].isna())[0]
    # remplissage des valeurs manquantes
    df['DELINS'].iloc[idx_na] = df['DTINS'].iloc[idx_na]-df['DDDOSP'].iloc[idx_na]
    # transformation du type timedelta au type float
    df['DELINS'].iloc[idx_na] = df['DELINS'].iloc[idx_na]/pd.to_timedelta(1, unit='D')
    # mise au format entier
    df['DELINS'] = df['DELINS'].astype(int)

    # Variable DELDEC
    # --------------------------------------
    # indices des valeurs manquantes de 'DELINS'
    idx_na = np.where(df['DELDEC'].isna())[0]
    # remplissage des valeurs manquantes
    df['DELDEC'].iloc[idx_na] = df['DTDEC'].iloc[idx_na]-df['DATDEC'].iloc[idx_na]
    # transformation du type timedelta au type float
    df['DELDEC'].iloc[idx_na] = df['DELDEC'].iloc[idx_na]/pd.to_timedelta(1, unit='D')
    # l'imputation a créé des valeurs négatives que l'on met à 0
    df['DELDEC'].iloc[np.where(df['DELDEC'] < 0)[0]] = 0
    # mise au format entier
    df['DELDEC'] = df['DELDEC'].astype(int)

    # Variables NBASSGPE & NBASSEXT
    # --------------------------------------
    # indices des valeurs manquantes de 'NBASSGPE' et 'NBASSEXT' (qui ont des NaN aux mêmes endroits)
    idx_na = np.unique(np.where(df[['NBASSGPE','NBASSEXT']].isna())[0])

    # valeurs moyennes des variables NBASSGPE et NBASSEXT 
    val_moy_NBASSGPE = round(df['NBASSGPE'].mean()) #round car on veut des entiers 
    val_moy_NBASSEXT = round(df['NBASSEXT'].mean()) #round car on veut des entiers

    for i in range(len(idx_na)):
        # choix aléatoire: 1 -> on impute par 'NBASSGPE', 0 -> on impute par 'NBASSEXT'
        is_NBASSGPE = random.choice([0,1])

        # on met NBASSEXT=0 et NBASSGPE=valeur moyenne 
        if is_NBASSGPE:
            df['NBASSEXT'].iloc[idx_na[i]] = 0
            df['NBASSGPE'].iloc[idx_na[i]] = val_moy_NBASSGPE
        # on met NBASSGPE=0 et NBASSEXT=valeur moyenne 
        else:
            df['NBASSEXT'].iloc[idx_na[i]] = val_moy_NBASSEXT
            df['NBASSGPE'].iloc[idx_na[i]] = 0

    return df



# ***********************************************************************
# Création de la Série Temporelle (pd.DataFrame) pour la prédiction MULTIVARIEE 
# ***********************************************************************

def create_MTS_day(df,var_floats,var_int,var_quali,var_date,var_rep):
    """
    Cette fonction permet d'aggréger les lignes ayant des dates identiques 
    afin d'avoir le format d'une série temporelle. Pour l'aggrégation, on prend  la classe 
    majoritaire pour les variables qualitatives et la valeur moyenne pour les variables 
    quantitatives. Pour la variable réponse 'NB_DOSS_DAY' (nb dossiers/jours), on prend la valeur 
    médiane. En effet, comme nous avons calculé au préalable le 'NB_DOSS_DAY' pour chaque jour,
    les lignes correspondant à une même date ont la même valeur de 'NB_DOSS_DAY' et la médiane 
    est simplement la valeur de 'NB_DOSS_DAY' par jour. 

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var_floats (list): variables quantitatives réelles.
    - var_int (list): variables quantitatives entières.
    - var_quali (list): variables qualitatives.
    - var_date (list): variable date.
    - var_rep (list): variable réponse.

    Output:
    ------
    - df_finale (dataframe): dataframe transformée.

    Reference:
    ---------
    https://stackoverflow.com/questions/15222754/
    groupby-pandas-dataframe-and-select-most-common-value
    """ 
    
    # Variable réponse
    # -------------------------------------
    # On a déjà calculé 'NB_DOSS_DAY' (nb de dossiers/jours). 
    # Toutes les lignes correspondant à une même dates ont une même valeur de 'NB_DOSS_DAY'
    # df_rep = dataframe de 'NB_DOSS_DAY' regroupés par date. 
    # En indice: date, en values:valeur médiane = valeur réelle (car les mêmes jours ont déjà les mêmes valeurs)
    df_rep = df[var_date+var_rep].groupby('DATEDI').median().astype(int)

    # Variables quantitatives
    # -------------------------------------
    # dataframe des variables quantitatives floats regroupés par date. 
    # En indice: date, en values:valeur moyenne par date
    df_float = df[var_date+var_floats].groupby(var_date).mean()
    # dataframe des variables quantitatives integer regroupés par date. 
    # En indice: date, en values:valeur moyenne par date arrondie à l'unité
    df_int = df[var_date+var_int].groupby(var_date).mean().round(0).astype(int)
    # merge de df_float & df_int.
    # En indice: date, en values: valeur moyenne par date
    df_quanti = df_float.merge(df_int,on=var_date)

    # Variables qualitatives
    # -------------------------------------
    # dataframe des variables qualitatives regroupés par date. 
    # En indice: date, en values:valeur majoritaire par date
    df_quali = df[var_date+var_quali].groupby(by=var_date).agg(lambda x: x.value_counts().index[0])

    # Merge variables quantitatives & qualitatives
    # -------------------------------------
    # on merge les tables df_quali et df_quanti selon la date 'DATEDI'
    df_quanti_quali = df_quanti.merge(df_quali,on=var_date)
    # on merge les tables df_quanti_quali et df_rep
    df_finale = df_quanti_quali.merge(df_rep,on=var_date)

    # on recrée la variable 'DATEDI' (qui était passée en index)
    df_finale[var_date[0]] = df_finale.index
    # on réindexe en partant de 0 (l'indexe était 'DATEDI')
    df_finale = df_finale.reset_index(level=0, drop=True).reset_index(drop=True)
    
    return df_finale


# ***********************************************************************
# Création de la Série Temporelle (pd.Series) pour la prédiction UNIVARIEE 
# ***********************************************************************

def create_TS(df,by='day'):
    """
    Cette fonction crée une série temporelle du nombre de 'CODOSB' par jour 
    ou par semaine. La référence de temps est donnée par la variable 'DATEDI'. 
    Si on compte le nombre de 'CODOSB' par semaine, on obtient le regroupement suivant:
    ex: 2017-12-11   12
        2017-12-18   18
        2017-12-25   31
    Ce qui indique qu'il y a 31 'CODOSB' entre ]2017-12-18, 2017-12-25],  
    18 'CODOSB' entre ]2017-12-11, 2017-12-18]. 

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - by (string): indique si on veut regrouper le nombre de 'CODOSB'
      par jours (by='day'), par jours en exclant les dimanches (by='business_day')
      ou par semaine (by='week').

    Output:
    -------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, le nombre de 'CODOSB' par jour, buisiness day ou semaine.
      
    References:
    ---------
    * Grouper par semaine: https://www.statology.org/pandas-group-by-week/
    * Alias de groupage: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """
    # =============================
    # Nb de 'CODOSB' par jours
    # =============================
    #----- nb de 'CODOSB' par jours (calcule du lundi au dimanche)
    TS_day = df.groupby([pd.Grouper(key='DATEDI', freq='D')])['CODOSB'].count()
    if by=='day': 
        TS = TS_day
        
    elif by=='business_day':
        #----- nb de 'CODOSB' par jours en excluant le dimanche (calcule du lundi au samedi)
        # indices des jours n'étant pas des dimanches dans TS_day
        idx_B = np.where(TS_day.index.strftime("%A")!='Sunday')[0]
        # on de garde de TS_day, que les jours du lundi au samedi
        TS_day_B = TS_day[idx_B] 
        TS = TS_day_B
        
    # =============================
    # Nb de 'CODOSB' par semaine
    # =============================  
    elif by=='week':
        # Convertit la colonne date en datetime et soustrait une semaine
        df['DATEDI'] = pd.to_datetime(df['DATEDI']) - pd.to_timedelta(7, unit='d')
        # Calcule la somme des valeurs, regroupées par semaine. On obtient en index, 
        # les dates correspondant à des lundis et en valeur, le nb de 'CODOSB' par semaine. 
        TS_week = df.groupby([pd.Grouper(key='DATEDI', freq='W-MON')])['CODOSB'].count()
        TS = TS_week
        
    # On trie la série temporelle (normalement elle est déjà triée)    
    TS.sort_index(inplace=True)
    
    return TS



# ***********************************************************************
# Tracé de la prédiction (par le modèle MiniRocket, SARIMA, ARIMA...)
# ***********************************************************************

def plot_prediction(TS,TS_forecast,splits,conf_int=None,model_name=''):
    """
    Cette fonction trace la prédiction du modèle ARIMA,SARIMA 
    ou MiniRocket.
    
    Input:
    ------
    - TS (pd.series): série temporelle avec en index
      une date et en valeur, la variable d'intérêt.
    - TS_forecast (pd.series): valeurs prédites sur le jeu de test 
      (index: une date, valeur :variable d'intérêt).
    - splits (list): liste de 2 tableaux comprenant les indices
      tu set de train et du set de test.   
    - conf_int (np.array): intervalle de confiance pour 
      les prédictions au niveau alpha=0.05. Si conf_int=None,
      aucun intervalle de confiance n'est tracé. 
    - model_name (string): nom du modèle utilisé pour la 
      prédiction pour l'affichage du graphique.

    Output:
    ------
    - Graphique.
    """

    fig, axs = plt.subplots(3, 1, figsize=(10,13),gridspec_kw={'height_ratios':[2,1.5,1.5]})
    fig.subplots_adjust(hspace=.5)
    # =================================
    # Tracé de la prédiction
    # =================================
    # Tracé de la série temporelle de train originale
    axs[0].plot(TS.iloc[splits[0]],label='ST de train',lw=1,color='#1f77b4')
    # Tracé de la série temporelle de test originale
    axs[0].plot(TS.iloc[splits[1]],label='ST de test',lw=1,color='#ff7f0e')
    # Tracé de la série temporelle de test prédite
    axs[0].plot(TS_forecast,label='ST prédite',lw=1,color='#2ca02c')
    
    # Tracé de l'intervalle de confiance si confint!=None
    if np.any(conf_int):
        cf= pd.DataFrame(conf_int)
        axs[0].fill_between(TS_forecast.index,
                        cf[0],
                        cf[1],color='grey',alpha=.3,label='IC(alpha=0.05)')
    
    axs[0].set_title('Prédiction par le modèle '+ model_name)
    axs[0].set_xlabel('Temps')
    axs[0].set_ylabel('NB_DOSS_DAY')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(ls=':')
    axs[0].legend(prop={'size': 8},loc=2)
    
    # =================================
    # Tracé de la prédiction avec zoom sur les données de test
    # =================================
    # Tracé de la série temporelle de test originale
    axs[1].plot(TS.iloc[splits[1]],label='ST de test',lw=1,color='#ff7f0e')
    # Tracé de la série temporelle de test prédite
    axs[1].plot(TS_forecast,label='ST prédite',lw=1,color='#2ca02c')
    axs[1].set_title('Zoom prédiction par le modèle '+ model_name)
    axs[1].set_xlabel('Temps')
    axs[1].set_ylabel('NB_DOSS_DAY')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(ls=':')
    axs[1].legend(prop={'size': 8},loc=2)
    
    
    # =================================
    # Tracé de l'erreur entre valeur réelle & prédiction
    # =================================
    err = np.abs(TS - TS_forecast)
    axs[2].scatter(x=TS.index, y=err,label='abs(erreur)',s=8,color='#d62728')
    axs[2].set_title('abs(valeur réelle - valeur prédite)')
    axs[2].set_xlabel('Temps')
    axs[2].set_ylabel('NB_DOSS_DAY')
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].grid(ls=':')
    
    # =================================
    # Affichage de la valeur moyenne
    # =================================
    mean_TS = int(np.mean(TS))
    mean_err = int(np.mean(err))
    print("Valeur moyenne de la ST originale: ",mean_TS)
    print("Valeur moyenne de l'erreur: ",mean_err)
    axs[2].axhline(y=mean_TS,color='blue',ls=':',label='valeur moyenne ST originale')
    axs[2].axhline(y=mean_err,color='green',ls=':',label='valeur moyenne abs(erreur)')
    axs[2].legend(prop={'size': 8},loc=2)
    
    plt.show()



# ***********************************************************************
# Transformation et transformation iverse Box-Cox d'un signal univarié
# ***********************************************************************
def transform_TS(TS,lam,cut=20,plot_graph:bool=True):
    """
    Cette fonction permet d'appliquer des corréctions sur la série temporelle 
    (ST) afin d'améliorer les prédictions ARIMA. 

    Input:
    ------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - lam (float): Si lam=0, la transformation log est appliquée. 
      La transformation log ne prend pas de valeurs nulles donc on supprime
      les première valeurs de la ST regroupée par semaine. Pour la ST regroupée
      par jour, les valeurs nulles ne sont pas seulement au début mais tout au
      long de la série donc on ne peut pas appliquer le log.
      Si lam>0, la transformation Cox-Box[1] de paramètre lambda est appliquée.
    - cut (int): si cut=n, on supprime les n premières valeurs de la ST donc 
      on ne sélectionne que ST[n:].
    - plot_graph (bool): si True, un graphique comprenant la ST d'origine et 
      la ST transformée est affiché.

    Rq: Pour la transformation Cox-Box, une bonne valeur de  lambda est celle 
    qui fait que la taille de la variation saisonnière est à peu près la même
    dans toute la série.

    Output:
    ------
    - TS_trans (pd.series): série temporelle transformée
      (index: date, valeur: la variable d'intérêt)
    - lam (float): valeur de lambda appliquée.

    ========================== Notes ==========================
    Soit Yt la ST à transformer. Pour l'étude avec ARIMA, 
    on suit les étapes suivantes:
    - On applique la transformation désirée à Yt: Wt = transformation(Yt)
    - On calcule la décomposition POSITIVE tendance+saisonnalité+résidus à Wt:
      Wt = mt+st+rt. On utilise de préférence la méthode STL. 
      (Rq: si transformation=log, alors une décomposition postive reviendra à 
      faire une décomposition multiplicative. Pour plus d'infos, se référer à
      la doc de la fonction 'decompose'). 
    - On retire la saisonnalité à Wt: Wt = Wt-st
    - On prédit les valeurs de test avec le modèle ARIMA sur Wt: on obtient W't.
    - On applique la transformée inverse pour obtenir les "véritables" valeurs 
      prédites: Y't = transformation_inv(W't)
    ===========================================================

    Référence:
    ----------
    [1] https://otexts.com/fpp2/transformations.html 
    """
    # On retire les cut=n premières valeurs qui sont négatives ou nulles
    # et/ou qui semblent incohérentes. 
    TS_trans  = TS[cut:] #retire les cut=n premières semaines
    if lam==0:
        # Transformation log pour que décomposition positive <=> décomposition multiplicative 
        TS_trans = np.log(TS_trans)
    else: 
        # Transformation Box-Cox pour que décomposition positive <=> décomposition multiplicative 
        TS_trans = np.multiply(np.sign(TS_trans),((np.abs(TS_trans))**lam - 1)/lam)
        
    # Tracé de la ST d'origine et de la ST transformée pour comparaison    
    if plot_graph:
        fig,axs = plt.subplots(1,2,figsize=(15,5))
        # ST originale
        axs[0].plot(TS)
        axs[0].set_title("ST d'origine")
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].grid(ls=':')
        # ST transformée
        axs[1].plot(TS_trans)
        axs[1].set_title("ST transformée Box-Cox\nlambda={}, cut={}".format(lam,cut))
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].grid(ls=':')
        plt.show()
        
    return TS_trans,lam


def transform_inv_TS(TS,lam):
    """
    Cette fonction applique la transformation inverse 
    de la fonction transform_TS.
    
    Input:
    ------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - lam (int): si lam=0, la transformation inverse du log est appliquée, 
      si lam>0, la transformation inverse Cox-Box de paramètre lambda 
      est appliquée.

    Output:
    ------
    - TS_inv (pd.series): série temporelle inverse transformée
      (index: date, valeur: la variable d'intérêt)
    """
    if lam==0:
        TS_inv = np.exp(TS)
    else:
        TS_inv = np.multiply(np.sign(lam*TS+1),(np.abs(lam*TS+1))**(1/lam))
    return TS_inv


# ***********************************************************************
# Calcul du rmse sur les données de test
# ***********************************************************************
def compute_rmse(y_test,y_pred):
    """
    Cette fonction calcule le RMSE.
    
    Input:
    -----
    - y_test (pd.Series): série temporelle de test contenant
      les valeurs réelles avec en index une date et en valeur, 
      la variable d'intérêt.
      
    - y_pred (pd.Series): série temporelle prédite sur le jeu 
      de test avec en index une date et en valeur, 
      la variable d'intérêt.
    
    Ouput:
    -----
    - rmse (float): valeur du RMSE
    """
    rmse = mean_squared_error(y_test,y_pred,squared=False)
    print('rmse sur les données de test :', round(rmse,5))
    return rmse


# ***********************************************************************
# Calcul du mae sur les données de test
# ***********************************************************************
def compute_mae(y_test,y_pred):
    """
    Cette fonction calcule le MAE.
    
    Input:
    -----
    - y_test (pd.Series): série temporelle de test contenant
      les valeurs réelles avec en index une date et en valeur, 
      la variable d'intérêt.
      
    - y_pred (pd.Series): série temporelle prédite sur le jeu 
      de test avec en index une date et en valeur, 
      la variable d'intérêt.
    
    Ouput:
    -----
    - mae (float): valeur du MAE
    """
    mae = mean_absolute_error(y_test,y_pred)
    print('mae sur les données de test :', round(mae,5))
    return mae