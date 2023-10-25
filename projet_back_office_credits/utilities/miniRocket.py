"""
Ce fichier permet de définir les fonctions nécessaires à l'exécution
de l'algorithme MiniRocket sur les données CREDIT pour répondre au 
problème "task1" dans le cas univarié et multivarié.

Le problème dénommé "task1" cherche à prédire la variable 'NB_DOSS_DAY'
c'est-à-dire le nombre de dossiers qui arrivent au backOffice 
par jour (si on regroupe les données au jour) ou par semaine,
(si on regroupe les données à la semaine). 
Pour prédire 'NB_DOSS_DAY', on considère une approche par série temporelle
multivariée ('NB_DOSS_DAY' + variables explicatives) ou univariée 
(uniquement 'NB_DOSS_DAY').

Contient les fonctions :
- one_hot_encode
- run_miniRocket_multi
- run_miniRocket_uni
"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs
import numpy as np
import pandas as pd

# MiniRocket
from tsai.basics import *

from tsai.models.MINIROCKET import *
from sklearn.metrics import mean_squared_error, make_scorer
    
import sktime
import sklearn

print(my_setup(sktime, sklearn))

import warnings
warnings.filterwarnings('ignore')
# Désactive warning SettingWithCopyWarning
# ref: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'

# ***********************************************************************
# One-hot encoding des variables qualitatives 
# ***********************************************************************
def one_hot_encode(df,num_cols,cat_cols):
    """
    Cette fonction permet de one-hot encoder le jeu 
    d'entraînement et le jeu de test. 

    Input:
    -------
    - df (dataframe): jeu de données entier (données de
      train et de test non séparées pour le moment).
    - cat_cols (list): variables qualitatives.
    - num_cols (list): variables quantitatives.

    Output:
    ------
    - df_oh (dataframe): jeu de données one-hot encodé.  
    """

    # Noms des colonnes qualitatives et quantitatves 
    # num_cols = df.select_dtypes(exclude=['category']).columns.tolist()
    # cat_cols = df.select_dtypes(include=['category']).columns.tolist()

    # dataframe réduite aux variables quantitatives
    df_num = df[num_cols]
    # dataframe réduite aux variables qualitatives + one-hot encoding
    df_cat = pd.get_dummies(df[cat_cols],drop_first = True)
    # concaténation des datafrmes quantitatifs et qualitatifs
    df_oh = pd.concat([df_num,df_cat],axis=1)
    
    return df_oh



# ***********************************************************************
# Modèle MiniRocket pour les données MULTIVARIEES
# ***********************************************************************
def run_miniRocket_multi(df,var_rep,window_length=9,n_estimators=5,valid_size=.2):
    """
    Cette fonction permet d'appliquer le modèle MiniRocket dans le cas 
    de données multivariées, après que celles-ci aient été préalablement
    one-hot encodées. Cependant, il n'est pas nécessaire de renormaliser 
    les variables.
    La fonction sépare le jeu de données en jeu de train et jeu de test. 
    Il n'y a donc pas besoin de séparer en train et en test au préalable.  

    Input:
    ------
    - df (dataframe): dataframe contenant les données (une sélection de 
      variable retenant les variables explicatives les plus pertinentes 
      peut être appliquée). 
    - var_rep (liste): variable réponse ['NB_DOSS_DAY'] (sous forme de liste)
    - window_length (int): taille de la fenêtre glissante. 
    - n_estimators (int): nombre de modèles MiniRockets qui sont fit.
      Les prédictions sont une moyenne de tous les modèles MiniRockets entrainés.
    - valid_size (float): taille du dataset de test.
    
    Output:
    -------
    - y_pred (np.array): valeurs réelles et valeurs prédites.
      On renvoie aussi les valeurs réelles pour les graphes + tard.
    - rmse (float): valeur du rmse calculé sur les donné de test 
    - splits (liste): liste de tableaux contenant les indices du jeu 
      de train et du jeu de test. 
    
    ================================== Notes =====================================
    
    Les variables explicatives sont X=[[t t t t], (toutes les variables du jeu de   
                                       [t t t t],  données credits préalablement 
                                       [t t t t],  sélectionnées sauf 'NB_DOSS_DAY')
                                       [v v v v],
                                       [v v v v]]
    
    et la variable à expliquer est y='NB_DOSS_DAY'=[t t t v v] où 
    't' représente les données de train et 'v' les données de test (validation).
    
    Afin que les données soient dans le bon format pour l'application de 
    MiniRocket, on applique une fenêtre glissante (comme indiqué par [1]). 
    La fenêtre glissante permet de passer d'un format X=(n_timesteps, n_features)
    à un format X=(n_samples, n_features, n_steps) où n_steps = window_length et 
    n_samples = n_windows = (n_timesteps - window_length)//stride + 1*(len(df)%2==0)
    De même, la réponse passe du format y=(n_timesteps) au format y=(n_samples).
    
    Pour constituer le vecteur y, la fonction SlidingWindow prend une valeur de y 
    toutes les window_length valeurs: 
    par exemple, si target = [0,1,2,3,4,5,6,7,8,9,10,11,12] et que window_length=3 
    alors y = [2,5,8,11]

    Après plusieurs test, un stride=1 est adopté pour la fenêtre glissante.
    Si stride=n>1, alors on obtiendra des valeurs prédites tous les
    n jours si les données sont regroupées par jour (ou toutes les n semaines
    si les données sont regroupées par semaines).
    Avec stride=1 ou n>1, les window_length premières valeurs sont omises. 
    
    
    La taille de la fenêtre est customisable et doit être >= 7. Après 
    plusieurs tests, de petites valeurs pour la taille de la fenêtre 
    montrent de meilleurs résultats.
    
    >>> Nous n'avons pas besoin de renormaliser les données avant d'appliquer
        MiniRocket.        
    >>> MiniRocketRegressor est recommandé pour des séries temporelles allant 
        jusqu'à 10k. Pour un jeu de données plus important, on peut utiliser 
        MINIROCKET (dans Pytorch avec GPU).
    ==============================================================================
        
    References:
    ----------
    - MiniRocket tutorial: 
        https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb
        https://github.com/timeseriesAI/tsai/tree/main/tsai/models
    - librairie tsai: 
        https://timeseriesai.github.io/tsai/tslearner.html
        https://timeseriesai.github.io/tsai/tslearner.html#tsclassifier-api
        https://timeseriesai.github.io/tsai/learner.html
    - données utilisées dans la doc de la librarie tsai:
        https://timeseriesai.github.io/tsai/data.external.html
    - comment utiliser les données présentes dans la doc de tsai
        [1] https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/00c_Time_Series_data_preparation.ipynb#scrollTo=5f1tQ1G2-GxY
    - la fonction slice (replace colon in index):
        https://stackoverflow.com/questions/7813305/array-assignment-in-numpy-colon-equivalent
    - explication des sliding windows: 
        https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
    - différence univarié/multivarié: 
        https://www.datacamp.com/tutorial/tutorial-time-series-forecasting
    """
    
    # liste des variables explicatives 
    exp_var = list(df.columns).copy()
    # on supprime de la liste la variable à prédire 
    exp_var.remove(var_rep[0])

    # Définition des paramètres de la fenêtre glissante
    # ------------------------------------------
    window_length = window_length # window_length is usually selected based on prior domain knowledge or by trial and error
    stride = 1 # None for non-overlapping (stride = window_length) (default = 1). 
               # This depends on how often you want to predict once the model is trained 
    start = 0  # use all data since the first time stamp (default = 0)
    get_x = exp_var # Indicates which are the columns that contain the x data.
    get_y = var_rep[0] # In multivariate time series, you must indicate which is/are the y columns
    horizon = 0  # 0 means y is taken from the last time stamp of the time sequence (default = 0)
    seq_first = True

    # Application de la fenêtre glissante
    # ------------------------------------------
    X, y = SlidingWindow(window_length, stride=stride, start=start, get_x=get_x,  
                         get_y=get_y, horizon=horizon, seq_first=seq_first)(df)

    # Calcul des indices des données de test de de train. Avec shuffle=False, on prend
    # les 80% des premières valeurs comme train et les 20% des dernières valeurs comme test.
    # ------------------------------------------
    splits = get_splits(y, valid_size=valid_size, stratify=True, random_state=23, shuffle=False)

    # Définition des set de train et de tests
    # ------------------------------------------
    X_train = X[splits[0]] ;  y_train = y[splits[0]]
    X_test  = X[splits[1]] ;   y_test = y[splits[1]]

    # Définition de la métrique et du modèle MiniRocket
    # ------------------------------------------
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    model = MiniRocketVotingRegressor(n_estimators=n_estimators, scoring=rmse_scorer)

    # On entraîne le modèle sur les données de train
    # ------------------------------------------
    timer.start(False) #début du timer
    print("début de l'entraînement...")
    model.fit(X_train, y_train) #entraînement du modèle
    print("...fin !")
    t = timer.stop() #fin du timer

    # On calcule les prédicitons sur les données de test
    # ------------------------------------------
    y_pred = model.predict(X_test)
    # on transforme les prédictions en entiers (car c'est le nb de crédits)
    y_pred = y_pred.astype(int)
    # on transforme les prédiction négatives en 0
    y_pred[np.where(y_pred<0)[0]] = 0
    # calcul & affichage du rmse
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'rmse sur les données de test : {rmse:.5f} temps d\'exécution: {t}')
    
    # Comme on a appliqué des fenêtres glissantes avec stride=1, les splits sont 
    # décalés de window_length, et on doit les redéfinir.
    # ------------------------------------------
    if stride==1:
        # valeur à ajouter 
        to_add = len(df)-(len(splits[0])+len(splits[1])) 
        # pour avoir des données qui correspondent à y_test et y_pred,
        # quand on fait TS[splits[1]], on ajoute window_length à tous 
        # les termes de splits[1] 
        s1 = np.array(splits[1])+to_add
        # pour avoir des données qui correspondent aux données de train
        # quand on fait TS[splits[0]], on ajoute window_length à tous 
        # les termes de splits[0] et on ajoute une liste [0,1,...window_length]
        # pour avoir le début de la TS. 
        s0 = np.r_[np.arange(to_add),np.array(splits[0])+to_add]
        splits = (s0,s1)

    return y_pred,rmse,splits




# ***********************************************************************
# Modèle MiniRocket pour les données UNIVARIEES
# ***********************************************************************
def run_miniRocket_uni(TS_,window_length=9,n_estimators=5,valid_size=.2):
    """
    Cette fonction permet d'appliquer le modèle MiniRocket dans le cas 
    de données univariées.
    La fonction sépare le jeu de données en jeu de train et jeu de test. 
    Il n'y a donc pas besoin de séparer en train et en test au préalable.  

    Input:
    ------
    - TS (pd.series): série temporelle avec en index
      une date et en valeur, la variable d'intérêt.
    - window_length (int): taille de la fenêtre glissante. 
    - n_estimators (int): nombre de modèles MiniRockets qui sont fit.
      Les prédictions sont une moyenne de tous les modèles MiniRockets entrainés.
    - valid_size (float): taille du dataset de test.
    
    Output:
    -------
    - y_pred (np.array): valeurs réelles et valeurs prédites.
      On renvoie aussi les valeurs réelles pour les graphes + tard.
    - rmse (float): valeur du rmse calculé sur les donné de test 
    - splits (liste): liste de tableaux contenant les indices du jeu 
      de train et du jeu de test. 
    
    ================================== Notes =====================================
    
    La série temporelle est TS=[[t]     
                                [t],  
                                [t],  
                                [v],
                                [v]]
    
    avec TS='NB_DOSS_DAY où 't' représente les données de train et
    'v' les données de test (validation).
    
    Afin que les données soient dans le bon format pour l'application de 
    MiniRocket, on applique une fenêtre glissante (comme indiqué par [1]). 
    La fenêtre glissante permet de passer d'un format TS=(n_timesteps)
    à un format X=(n_samples, 1 , n_steps) où n_steps = window_length et 
    n_samples = n_windows = (n_timesteps - window_length)//stride + 1*(len(df)%2==0).
    De même, la réponse devient est au format y=(n_samples).
    
    Pour constituer le vecteur y, la fonction SlidingWindow prend une valeur de TS 
    toutes les window_length valeurs: 
    par exemple, si TS = [0,1,2,3,4,5,6,7,8,9,10,11,12] et que window_length=3 
    alors y = [2,5,8,11]

    Après plusieurs test, un stride=1 est adopté pour la fenêtre glissante.
    Si stride=n>1, alors on obtiendra des valeurs prédites tous les
    n jours si les données sont regroupées par jour (ou toutes les n semaines
    si les données sont regroupées par semaines).
    Avec stride=1 ou n>1, les window_length premières valeurs sont omises. 
    
    La taille de la fenêtre est customisable et doit être >= 7. Après 
    plusieurs tests, de petites valeurs pour la taille de la fenêtre 
    montrent de meilleurs résultats.
    
    >>> Nous n'avons pas besoin de renormaliser les données avant d'appliquer
        MiniRocket.        
    >>> MiniRocketRegressor est recommandé pour des séries temporelles allant 
        jusqu'à 10k. Pour un jeu de données plus important, on peut utiliser 
        MINIROCKET (dans Pytorch avec GPU).
    ==============================================================================
        
    References:
    ----------
    - MiniRocket tutorial: 
        https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb
        https://github.com/timeseriesAI/tsai/tree/main/tsai/models
    - librairie tsai: 
        https://timeseriesai.github.io/tsai/tslearner.html
        https://timeseriesai.github.io/tsai/tslearner.html#tsclassifier-api
        https://timeseriesai.github.io/tsai/learner.html
    - données utilisées dans la doc de la librarie tsai:
        https://timeseriesai.github.io/tsai/data.external.html
    - comment utiliser les données présentes dans la doc de tsai
        [1] https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/00c_Time_Series_data_preparation.ipynb#scrollTo=5f1tQ1G2-GxY
    - la fonction slice (replace colon in index):
        https://stackoverflow.com/questions/7813305/array-assignment-in-numpy-colon-equivalent
    - explication des sliding windows: 
        https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
    - différence univarié/multivarié: 
        https://www.datacamp.com/tutorial/tutorial-time-series-forecasting
    """
    
    # La TS passée en argument peut contenir des dates en index.
    # On réinitialise les indices
    # ------------------------------------------
    TS = TS_.copy()
    TS.reset_index(drop=True,inplace=True)
    
    # Définition des paramètres de la fenêtre glissante
    # ------------------------------------------
    window_length = window_length # window_length is usually selected based on prior domain knowledge or by trial and error
    stride = 1 # None for non-overlapping (stride = window_length) (default = 1). 
               # This depends on how often you want to predict once the model is trained 
    horizon = 1 

    # Application de la fenêtre glissante
    # ------------------------------------------
    X, y = SlidingWindow(window_length, stride=stride, horizon=horizon)(TS)

    # Calcul des indices des données de test de de train. Avec shuffle=False, on prend
    # les 80% des premières valeurs comme train et les 20% des dernières valeurs comme test.
    # ------------------------------------------
    splits = get_splits(y, valid_size=valid_size, stratify=True, random_state=23, shuffle=False)
    
    # Définition des set de train et de tests
    # ------------------------------------------
    X_train = X[splits[0]] ;  y_train = y[splits[0]]
    X_test  = X[splits[1]] ;   y_test = y[splits[1]]

    # Définition de la métrique et du modèle MiniRocket
    # ------------------------------------------
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    model = MiniRocketVotingRegressor(n_estimators=n_estimators, scoring=rmse_scorer)

    # On entraîne le modèle sur les données de train
    # ------------------------------------------
    timer.start(False) #début du timer
    print("début de l'entraînement...")
    model.fit(X_train, y_train) #entraînement du modèle
    print("...fin !")
    t = timer.stop() #fin du timer

    # On calcule les prédicitons sur les données de test
    # ------------------------------------------
    y_pred = model.predict(X_test)
    # on transforme les prédictions en entiers (car c'est le nb de crédits)
    y_pred = y_pred.astype(int)
    # on transforme les prédiction négatives en 0
    y_pred[np.where(y_pred<0)[0]] = 0
    # calcul & affichage du rmse
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'rmse sur les données de test : {rmse:.5f} temps d\'exécution: {t}')
    
    # Comme on a appliqué des fenêtres glissantes avec stride=1, les splits sont 
    # décalés de window_length, et on doit les redéfinir.
    # ------------------------------------------
    if stride==1:
        # valeur à ajouter 
        to_add = len(TS)-(len(splits[0])+len(splits[1])) 
        # pour avoir des données qui correspondent à y_test et y_pred,
        # quand on fait TS[splits[1]], on ajoute window_length à tous 
        # les termes de splits[1] 
        s1 = np.array(splits[1])+to_add
        # pour avoir des données qui correspondent aux données de train
        # quand on fait TS[splits[0]], on ajoute window_length à tous 
        # les termes de splits[0] et on ajoute une liste [0,1,...,window_length]
        # pour avoir le début de la TS. 
        s0 = np.r_[np.arange(to_add),np.array(splits[0])+to_add]
        splits = (s0,s1)

    return y_pred,rmse,splits 