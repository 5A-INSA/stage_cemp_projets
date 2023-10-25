"""
Ce fichier permet de définir les fonctions nécessaires pour 
l'exécution des algorithmes ARIMA et SARIMA sur les données
CREDIT pour répondre au problème "task1" dans le cas univarié
seulement.

Le problème dénommé "task1" cherche à prédire la variable 'NB_DOSS_DAY'
c'est-à-dire le nombre de dossiers qui arrivent au backOffice 
par jour (si on regroupe les données au jour) ou par semaine,
(si on regroupe les données à la semaine). 
Pour prédire 'NB_DOSS_DAY', on considère une approche par série temporelle
multivariée ('NB_DOSS_DAY' + variables explicatives) ou univariée 
(uniquement 'NB_DOSS_DAY').

Contient les fonctions :
- plot_FFT
- decompose
- correct_season
- split_train_test_uni
- plot_autocorr_TS
- diff_TS
- train_SARIMA
"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs
import numpy as np
import pandas as pd

# Graphes
import matplotlib.pyplot as plt

# Transformée de Fourier
from scipy.fft import fft, fftfreq
from scipy import signal as sig
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

# ACF et PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statistics import stdev

# modèle ARIMA/SARIMA
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


# ***********************************************************************
# Calcul de la transformée de Fourier d'un signal univarié
# ***********************************************************************
def plot_FFT(TS,grouped_by,thres=1e4,fig_size=[10,9]):
    """
    Cette fonction trace le signal entrée en argument ainsi que 
    la Transformée de Fourier discrète du signal.
    
    Cette fonction est utile pour déterminer la période de la 
    saisonnalité. Par exemple, un pic de fréquence à 
    f= 0.10[1/Jour] donne une période de 1/f = 10 jours. 

    Input:
    ------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - grouped_by (string): Indique si les données sont regroupées par 
      jour ou par semaine pour l'affichage du label des axes des plots.
    - thres (float): les fréquences au dessus du seuil 'thres' seront 
      indiquées avec un point rouge.
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 
 
    Output:
    -------
    - Graphiques.

    Reference:
    ----------
    FFT: https://docs.scipy.org/doc/scipy/tutorial/fft.html#d-discrete-fourier-transforms
    Trouver la saisonnalité(= période) à partir de FFT: https://github.com/Kommandat/seasonality-
    fourier-analysis/blob/master/notebooks/Part%201%20-%20Seasonality%20Analysis%20with%20scipy-fft.ipynb
    """

    # ===================================
    # Calcul de la Transformée de Fourier Discrete (DFT)
    # avec l'algorithme Fast Fourier Transform (FFT)
    # ===================================
    # Number of sample points
    N = len(TS)
    # Sample spacing
    T = 1 # 1 jour ou 1 semaine selon que les données sont aggrégées au jour ou à la semaine

    # Valeurs de la DFT (= nombre complexe a+jb)
    fft_output = fft(TS) 
    # Amplitude de la DFT (= |a+jb|)
    power = np.abs(fft_output) 
    # Lieu des fréquences calculées
    freq = fftfreq(N, T)

    # On ne garde que les fréquences positives. La FFT produit 
    # des fréquences positives et négatives mais seules les 
    # fréquences positives ont une réalité physique.
    mask = freq >= 0 
    freq = freq[mask]
    power = power[mask]

    # ===================================
    # Tracé des graphiques 
    # ===================================
    fig, axs = plt.subplots(3,1, figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(hspace=0.5)

    # Graphique de la série temporelle d'origine
    #--------------------------------
    axs[0].plot(TS)
    axs[0].set_title("Signal regroupé par {}".format(grouped_by))
    axs[0].set_xlabel("Temps")
    axs[0].set_ylabel("NB_DOSS_DAY")
    axs[0].grid(linestyle=':')

    # Graphique de la Transformée de Fourier 
    #--------------------------------
    axs[1].plot(freq, power)
    axs[1].set_title("Toutes les fréquences du signal")
    axs[1].set_xlabel("Fréquence [1/{}]".format(grouped_by))
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(linestyle=':')

    # Graphique de la Transformée de Fourier où on ne garde que
    # les fréquences qui ont une amplitude >= 0.25 et qui sont
    # donc les fréquences dominantes du signal. Les fréquences 
    # nulles sont omises car elles représentent la baseline et
    # sont inutiles pour l'analyse.
    #--------------------------------
    mask = (freq > 0) & (freq <= 0.25) 
    axs[2].plot(freq[mask], power[mask])
    axs[2].set_title('Fréquences dans ]0, 0.25]')
    axs[2].set_ylabel( 'Amplitude' )
    axs[2].set_xlabel( 'Fréquence [1/{}]'.format(grouped_by))
    axs[2].grid(linestyle=':')
    # placer un marqueur aux fréquences dominances
    peaks = sig.find_peaks(power[freq >=0], prominence=thres)[0]
    peak_freq =  freq[peaks]
    peak_power = power[peaks]
    axs[2].plot(peak_freq, peak_power, 'ro')

    plt.tight_layout()
    plt.xticks(rotation=90)

    plt.show()



# ***********************************************************************
# Décomposition d'une série temporelle univariée en tendance, saisonnalité, résidu
# ***********************************************************************

def decompose(TS,decomposition='Naive',period=None,model='additive',fig_size=[10,9]):
    """
    Cette fonction permet de décomposer un signal Yt selon 3 paramètres: 
    tendance (mt), saisonnalité (st) et résidu (rt). 

    Input:
    ------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - decomposition (string): type de décomposition 'Naive' ou 'STL' (voir les 
      détails dans les Notes plus bas).
    - period (integer): La période est le nombre de fois que le cycle saisonnier
      se répète en un an. Si period=None, la période est prise comme la période de
      la TS (obentue avec TS.index.freq). Ainsi, si TS est regroupé par semaine period=52.
      On peut également utiliser la fonction plot_FFT pour déterminer la saisonnalité. 
    - model (string): 'additive' ou 'multiplicative' pour déterminer le type de 
      décomposition (voir détails dans la Notes plus bas).
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 

    Output:
    -------
    - decompose.seasonal, decompose.trend, decompose.resid (pd.series): séries temporelles avec 
      en index une date et en valeur, respectivement la saisonalité, la tendance et les résidus.

    ======================================= Notes ============================================
    I) Si la décomposition est 'additive', alors le signal est décomposé comme:
    Yt = mt+st+rt. Pour retirer la saisonnalité de Yt, on calcule: Yt - st
    > Approprié si la variation des fluctuations saisonnières 
      ne varie pas avec le niveau de la série temporelle [1]. 

    II) Si la décomposition est 'multiplicative', alors le signal est décomposé comme: 
    Yt = mt*st*rt. Pour retirer la saisonnalité de Yt, on calcule: Yt/st. 
    > Approprié si la variation des fluctuations saisonnières semble 
      être proportionnelle au niveau de la série temporelle
      (couramment utilisé pour les séries temporelles économiques) [1].

    III) Si la décomposition semble être multiplicative mais que le signal comprend des valeurs nulles
    (comme c'est le cas ici), on utilise une décomposition pseudo-additive comme mentionné dans [1].
    Pour ce faire, on utilise la transformation de Box-Cox avec labmda > 0 pour transformer
    Yt en Wt. On effectue ensuite une décomposition additive sur Wt: Wt = mt+st+rt. On retire 
    la saisonnalité à Wt: Wt = Wt-st. On calule les prédictions sur les données transformées Wt. 
    A la fin de l'étude, on applique la transformée inverse de Cox-Box sur les données prédites.
    La valeur lambda doit être choisie de sorte à ce que la saisonnalité soit la plus régulière 
    possible. 

    ----- Méthode 1: -----
    La première méthode de décomposition est 'seasonal_decompose' de 'statsmodels'. 
    Cette décomposition est naive et des méthodes plus sophistiquées doivent être privilégiées.

    ----- Méthode 2: -----
    La deuxième méthode de décomposition est 'STL' de 'statsmodels'. Cette méthode est recommandée
    par [1] pour l'étude des données à la semaine. Cependant, elle ne supporte pas de décomposition
    multiplicative. Si les données sont non nulles, on peut utiliser une décomposition additive sur
    log(Yt) car Yt = mt*st*rt <=> log(Yt)=log(mt)+log(st)+log(rt). Si la série temporelle présente 
    des données nulles, on peut utiliser la méthode décrite dans III). 
    ===========================================================================================

    Références:
    -----------
    - Livre de référence pour cette étude [1] : 
        https://otexts.com/fpp2/stl.html
    - Fonctionnement de 'seasonal_decompose': 
        https://github.com/statsmodels/statsmodels/issues/3872
        https://stats.stackexchange.com/questions/285718/seasonal-decomposition
    - Semi-additive models: 
        https://www.abs.gov.au/websitedbs/d3310114.nsf/home/time+series+analysis:+the+basics
        (je n'ai pas réussi à trouver une foncion python qui l'implémentait)
    - Fonctionnement de 'STL' : 
        https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
        https://stackoverflow.com/questions/66067471/how-to-choose-the-correct-arguments-of-statsmodels-stl-function
    """

    # ===================================================
    # Calcul de la décomposition (tendance,saisonnalité,résidu)
    # ===================================================
    #----- Method 1: Seasonal decomposition using moving averages.
    if decomposition=='Naive':
        decompose = seasonal_decompose(TS,model=model, period=period, extrapolate_trend='freq') 
        # Attention, paramètre "freq" <=> "period". 
        # Si period=n, alors on considère qu'il y a une saisonnalité toutes les n semaines si les données sont 
        # regroupées par semaines ou tous le n jours si les données sont regroupées par jours. 

    #----- Method 2: Seasonal decomposition using STL.
    elif decomposition=='STL':
        stl = STL(TS,period=period)
        decompose = stl.fit()

    # ===================================================
    # Tracé des graphiques 
    # ===================================================
    fig_size = [10,15]

    fig, axs = plt.subplots(4,1, figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(hspace=0.4)

    # Graphique de la série temporelle
    axs[0].plot(TS)
    axs[0].set_title("Signal")
    axs[0].set_xlabel("Temps",fontsize=8)
    axs[0].set_ylabel("NB_DOSS_DAY",fontsize=8)
    axs[0].grid(linestyle=':')

    # Graphique de la tendance
    axs[1].plot(decompose.trend)
    axs[1].set_title("Tendance")
    axs[1].set_xlabel("Temps",fontsize=8)
    axs[1].set_ylabel("NB_DOSS_DAY",fontsize=8)
    axs[1].grid(linestyle=':')

    # Graphique de la saisonnalité
    axs[2].plot(decompose.seasonal)
    axs[2].set_title("Saisonnalité")
    axs[2].set_xlabel("Temps",fontsize=8)
    axs[2].set_ylabel("NB_DOSS_DAY",fontsize=8)
    axs[2].grid(linestyle=':')

    # Graphique de la saisonnalité
    axs[3].scatter(x=TS.index,y=decompose.resid,s=8)
    axs[3].set_title("Résidu")
    axs[3].set_xlabel("Temps",fontsize=8)
    axs[3].set_ylabel("NB_DOSS_DAY",fontsize=8)
    axs[3].grid(linestyle=':')

    return decompose.seasonal,decompose.trend,decompose.resid





# ***********************************************************************
# Fonction permettant d'enlever la saisonnalité d'un signal univarié
# ***********************************************************************

def correct_season(TS,TS_season,model='additive',display_graph:bool=True):
    """
    Cette fonction corrige (= retire) la saisonnalité de la série 
    temporelle passée en argument. 

    Input:
    ------
    - TS (pd.series): série temporelle originale Yt avec en index une date 
      et en valeur, la variable d'intérêt.
    - TS_season (pd.series): série temporelle se la saisonnalité St avec en 
      index une date et en valeur, la variable d'intérêt.
    - model (string): 'additive' ou 'multiplicative' pour spécifier le type de 
      décomposition. Si model='additive', la série temporelle Yt est corrigée par 
      Yt = Yt - St, si model='additive', la série temporelle Yt est corrigée par 
      Yt = Yt/St.
    - display_graph (bool): si True, affiche un graphe avec la série temporelle
      et la série temporelle corrigée. 

    Output:
    -------
    - TS_cor (pd.series): série temporelle corrigée de la saisonnalité.
    - Graphique si display_graph=True
    """
    
    if model=='additive':
        TS_cor = TS.subtract(TS_season)
        # avec la soustraction, certaines variables deviennent NaN
    elif model=='multiplicative':
        TS_cor = TS.divide(TS_season)
        
    if display_graph==True:
        fig, axs = plt.subplots(1,1, figsize=(10,5))
        axs.plot(TS,label="ST originale")
        axs.plot(TS_cor,label="ST corrigée de la saisonnalité")
        axs.set_title("Comparaison ST originale et ST corrigée de la saisonnalité")
        axs.set_xlabel("Temps")
        axs.set_ylabel("NB_DOSS_DAY")
        plt.legend()
        plt.grid(linestyle=':')
    return TS_cor


# ***********************************************************************
# Fonction permettant de séparer le jeu de données en jeu de train et jeu de test
# ***********************************************************************
def split_train_test_uni(TS,k=0.8):
    """
    Divise la série temporelle en set de train et de test 
    pour l'étude univariée.
    
    Input:
    ------
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - k (float): pourcentage du set de train.   

    Output:
    -------
    - TS_train,TS_test (pd.series): série temporelle de train  
      et de test avec en index une date et en valeur, la variable 
      d'intérêt.
    - splits (list): liste de 2 tableaux comprenant les indices
      tu set de train et du set de test.   
    """
    # indices du train set et du test set
    splits = [np.arange(0,int(k*len(TS))),np.arange(int(k*len(TS)),len(TS))]
    TS_train = TS[splits[0]]
    TS_test  = TS[splits[1]]
    return TS_train,TS_test,splits




# ***********************************************************************
# Trace l'autocorrélation d'une série temporelle univariée
# ***********************************************************************
def plot_autocorr_TS(TS):
    """
    Cette fonction trace les plots ACF et PACF permettant de déterminer
    si la série teporelle (ST) est stationnaire et les paramètres des modèles
    ARIMA ou SARIMA. Si la ST n'est pas stationnaire, on peut la différencier 
    pour la rendre plus stationnaire.
    
    La fonction réalise aussi le test ADF (Augmented Dickey-Fuller) où 
    l'hypothèse nulle est que la ST est stationnaire. Ainsi, une  
    p-valeur > 0.05 indique que l'on rejette PAS l'hypothèse nulle 
    et donc que la ST est non-stationnaire (Hypothèse nulle : 
    la série temporelle n'est PAS stationnaire). 

    Input:
    -----
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.

    Output:
    ------
    - p_val (float): p-valeur du test ADF
    - ecart_type (float): écart type de la série temporelle différenciée
    - Affichage graphiques & p-valeur du test ADF
    
    ############################## Notes ##################################
    
    - Le log peut aider à stabiliser la variance d'une ST.
    - La différentiation peut aider à stabiliser la moyenne d'une ST en
    éliminant (ou en réduisant) la tendance et la saisonnalité.

    ###### ACF  
    Le graphique ACF montre les autocorrélations qui mesurent la relation entre Yt et
    Yt-k pour différentes valeurs de k. Or si Yt et Yt-1 sont corrélés alors Yt-1 et Yt-2
    sont probablement aussi corrélés. Cependant, Yt et Yt-2 peuvent être corrélés car ils
    sont tous deux connectés à Yt-1 plutot que parce que Yt-2 contient une nouvelle 
    information qui pourrait être utilisée pour la prédiction.
    
    ###### PACF
    Pour surmonter ce problème, nous pouvons utiliser PACF. Celles-ci mesurent la relation 
    entre Yt et Yt-k en en enlevant les effets des lags 1,2,3,...k-1.
    
    *************************************************************
              IDENTIFIER PARAMETRES ARIMA(p,d,q) 
    *************************************************************
    ==================================================
    Identifier l'Ordre de différentation (d) et la constante: 
    ==================================================
    Règle 0: Si la série originale est stationnaire, l'ACF tombe à zéro rapidement,alors que si la
    série n'est pas stationnaire, l'ACF diminue lentement. Le variance et la moyenne des séries
    stationnaires ne changent pas avec le temps. Pour savoir si une série est stationnaire,
    on peut aussi calculer les moyennes et variances de la série et de la série différenciée.

    Règle 1: ACF: si la ST a des autocorrélations positives jusqu'à un grand nombre de lags (10 ou +), 
    alors elle a besoin d'un ordre de différentiation plus élevé.

    Règle 2: ACF: Si l'autocorrélation à lag-1 est négative ou nulle, ou si les autocorrélations sont
    toutes petites et sans schéma, alors la ST n'a pas besoin d'un ordre de différentiation plus élevé. 
    Si l'autocorrélation lag-1 est de -0,5 ou plus négative, la série peut être surdifférenciée. 
    /.\ Attention à la surdifférenciation !

    Règle 3: L'ordre optimal de différentiation est souvent l'ordre de différentiation auquel l'écart-type
    est le plus faible. (Pas toujours, cependant). Trop ou pas assez de différentiation peut également 
    être corrigée avec des termes AR ou MA (voir les règles 6 et 7).

    Règle 4: Un modèle **sans** ordre de différentiation suppose que la série originale est stationnaire.
    Un modèle avec **un** ordre de différentiation suppose que la série originale a une tendance moyenne 
    constante. Un modèle avec **deux** ordres de différentiation suppose que la série originale a une 
    tendance variable dans le temps.

    Règle 5: Un modèle **sans** ordre de différentiation comprend normalement un terme constant 
    (qui implique une moyenne non nulle). Un modèle avec **deux** ordres de différentiation n'inclut
    normalement pas de terme constant. Dans un modèle avec **un** ordre de différentiation, un terme 
    constant doit être inclus si la série a une tendance moyenne non nulle.

    ==================================================
    Identifier le nombre de termes AR(p) et MA(q) :
    S'applique une fois que la ST est STATIONNAIRE (= différenciée)
    ==================================================

    ###### Les données suivent un modèle ARIMA(p,d,0)=AR(p) si : 

    Règle 6: (ignorer la valeur au lag 0)
    - PACF: si la série différentiée présente une coupure nette et/ou si l'autocorrélation au lag-1 est positive,
      i.e. si la série semble légèrement "sous-différenciée", il faut envisager d'ajouter un ou plusieurs termes 
      AR au modèle. Le décalage au-delà duquel le PACF se coupe est le nombre de termes AR à ajouter.
    - PACF: il y a un pic significatif à lag p (au delà de l'intervalle de confiance) mais aucun au-delà 
      du lag p (pic significatif positif ou négatif).
    - PACF: le graphe décroit rapidement vers zéro (de façon exponentielle ou sinusoïdale)
    - ACF: décroît plus graduellement.

    ###### Les données suivent un modèle ARIMA(0,d,q) = MA(q) si : 

    Règle 7: (ignorer la valeur au lag 0)
    - ACF: si la série différentiée présente une coupure nette et/ou si l'autocorrélation en lag-1 est négative,
      i.e. si la série semble légèrement "surdifférenciée", il faut envisager d'ajouter un terme MA au modèle.
      Le lag au-delà duquel l'ACF se coupe est le nombre de termes MA à ajouter.
    - ACF: il y a un pic significatif à lag q (au delà de l'intervalle de confiance) mais aucun au-delà
      du lag q (pic significatif positif ou négatif).
    - AFC: le graphe décroit rapidement vers zéro (de façon exponentielle ou sinusoïdale)
    - PACF: décroît plus graduellement.

    Règle 8: Il est possible qu'un terme AR et un terme MA annulent leurs effets respectifs, donc si un modèle
    mixte AR-MA semble convenir aux données, essayez également un modèle avec un terme AR et un terme MA en moins,
    en particulier si les estimations des paramètres dans le modèle original nécessitent plus de 10 itérations pour 
    converger. /.\ Attention à l'utilisation de plusieurs termes AR et de plusieurs termes MA dans le même modèle !

    Règle 9: Si la somme des coefficients AR du modèle est presque égale à 1 (racine unitaire), vous devez réduire
    le nombre de termes AR d'une unité et augmenter l'ordre de différentiation d'une unité.

    Règle 10: Si la somme des coefficients MA du modèle est presque égale à 1 (racine unitaire), vous devez réduire
    le nombre de termes MA d'une unité et l'ordre de différentiation d'une unité.

    Règle 11: Si les prévisions à long terme semblent erratiques ou instables, il peut y avoir une racine 
    unitaire dans les coefficients AR ou MA.
    
    *************************************************************
            IDENTIFIER PARAMETRES SARIMA(p,d,q)(P,D,Q,m) 
    *************************************************************
    ==================================================
    Identifier la partie saisonnière du modèle (m) :
    ==================================================
    Règle 12: Si nous voulons utiliser un modèle SARIMA, nous avons besoin de connaitre la 
    saisonnalité. On peut tout à fait utiliser un modèle SARIMA sur une ST dont 
    on a retiré la saisonnalité, et c'est même ce qui est recommandé de faire selon [1]
    (où on recommande d'étudier la ST sans saisonnalité et la saisonnalité de façon
    séparée pour mieux déterminer les paramètres du modèle SARIMA.)
    Ainsi, si par exemple, nos données ont été collectées sur une base mensuelle et que
    nous avons une saisonnalité annuelle, on a une saisonnalité lag=12 et on devrait
    observer un pic au lag 12 dans les plots ACF et PACF. 

    Règle 13: Si la série présente une tendance saisonnière forte et cohérente, alors vous devez utiliser un ordre
    de différenciation saisonnière (sinon le modèle suppose que la tendance saisonnière s'estompera avec le temps).
    Cependant, n'utilisez jamais plus d'un ordre de différenciation saisonnière ou plus de deux ordres de 
    différenciation totale (saisonnière + non saisonnière).
    
    ==================================================
    Identifier l'Ordre de différentation saisonnière (D)
    ==================================================
    Règle 14: Si nous avons utilisé la différenciation saisonnière pour rendre la ST stationnaire 
    (par exemple, la valeur réelle (Yt) soustraite par m=12 mois précédents (Yt-12)), nous ajoutons
    1 terme à la différenciation saisonnière.

    ==================================================
    Identifier le nombre de termes AR(P) et MA(Q) :
    S'applique une fois que la ST différenciée.
    ==================================================
    ###### Termes AR(P) : 
    Règle 15: 
    - PACF: Au lieu de compter le nombre de pics qui sortent de l'intervalle de confiance, 
      on compte le nombre de pics saisonniers qui en sortent.
      Par exemple, si m=12, on regarde si le pic au lag 12 est hors de l'intervalle de confiance. 
      Dans ce cas, on ajoute un terme AR (SAR).
    - ACF: Si l'autocorrélation de la série différentiée de manière appropriée est positive au lag m, 
      où m est le nombre de périodes dans une saison, envisager d'ajouter un terme SAR au modèle. 
      Cette situation est susceptible de se produire si une différence saisonnière n'a pas été utilisée, 
      ce qui ne serait approprié que si le modèle saisonnier n'est pas stable dans le temps. 
     
    ###### Termes MA(Q) : 
    Règle 16: 
    - ACF: Au lieu de compter le nombre de pics qui sortent de l'intervalle de confiance, 
      on compte le nombre de pics saisonniers qui en sortent.
      Par exemple, si m=12, on regarde si le pic au lag 12 est hors de l'intervalle de confiance. 
      Dans ce cas, on ajoute un terme MA (SMA).
    - ACF: Si l'autocorrélation de la série différentiée est négative au lag m, envisager d'ajouter
      un terme SMA au modèle. Cette situation est susceptible de se produire si une différence saisonnière
      a été utilisée, ce qui devrait être fait si les données ont un modèle saisonnier stable et logique.
      
    /.\ Eviter d'utiliser plus d'un ou deux paramètres saisonniers (SAR+SMA)  dans le même modèle, 
    car cela risque d'entraîner un ajustement excessif des données et/ou des problèmes d'estimation.
      
    ############################## FIN Notes ##############################

    Références:
    ----------
    Choisir les paramètres ARIMA : 
    - https://otexts.com/fpp2/non-seasonal-arima.html
    - https://www.justintodata.com/arima-models-in-python-time-series-prediction/
    - https://medium.com/analytics-vidhya/arima-fc1f962c22d4
    - https://people.duke.edu/~rnau/arimrule.htm
    Choisir les paramètres SARIMA:
    - https://arauto.readthedocs.io/en/latest/how_to_choose_terms.html
    - [1] https://stats.stackexchange.com/questions/385657/determine-paramaters-for-sarima-model
    
    Autocorrélation négative:
    - http://www.pmean.com/09/NegativeAutocorrelation.html
    """
    # =======================================
    # Graphiques ACf et PACF
    # =======================================
    # ACF et PACF plots qui sont utiles pour déterminer les valeurs p et q
    # Si p,q sont tous deux >0, ces plots n'aident pas à trouver leurs valeurs.
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.3)
    x_lines = np.arange(0,26) #locations des lignes verticales
    
    #----- ACF
    acf_original = plot_acf(TS,ax=axs[0],title = "Autocorrelation (ACF)")
    axs[0].set_xlabel("lag"); axs[0].set_ylabel("ACF")
    for v in x_lines: #ajout de lignes verticales pour la précision
        axs[0].axvline(x=v,color='grey',alpha=.6, ls=':',lw=1)

    #----- PACF
    pacf_original = plot_pacf(TS,ax=axs[1],title = "Partial Autocorrelation (PACF)")
    axs[1].set_xlabel("lag"); axs[1].set_ylabel("PACF")
    for v in x_lines: #ajout de lignes verticales pour la précision
        axs[1].axvline(x=v,color='grey',alpha=.5, ls=':',lw=1)
        
    plt.show()
    
    # =======================================
    # Test ADF
    # =======================================
    adf_test = adfuller(TS)
    p_val = adf_test[1] #p-valeur
    print('p-value test ADF : {}'.format(p_val))
    if p_val >= 0.05:
        print('p-value > 0.05 : ST non stationnaire')
    else: 
        print('p-value < 0.05 : ST stationnaire')
        
    # =======================================
    # Ecart type de la série temporelle 
    # =======================================
    ecart_type = stdev(TS)
    print('Ecart type de la ST : ',round(ecart_type,3))
    return p_val, ecart_type



# ***********************************************************************
# Différenciation d'une série temporelle univariée
# ***********************************************************************
def diff_TS(TS,period=1,show_graph:bool=True):
    """
    Cette fonction permet de différencier une série temporelle
    Y_t afin de la rendre plus startionnaire:
    Y_t = Y_t - Y_(t-k) où k est l'ordre de la différenciation
    (argument period). 
    Pour une ST avec une saisonnalité, on peut commencer par 
    d'abord différencier avec la valeur de la saisonnalité
    puis ensuite différencier avec k=1,2,... (autant de fois
    que nécessaite pour rendre la ST stationnaire)
    
     Input:
    -----
    - TS (pd.series): série temporelle avec en index une date 
      et en valeur, la variable d'intérêt.
    - period (integer): ordre de la différenciation 
    - show_graph (bool): si True, la ST différenciée 
      est affichée.

    Output:
    ------
    - TS_diff (pd.series): série temporelle différenciée
    - Graphique de la série temporelle différenciée si
      show_graph=True
    """

    TS_diff = TS.diff(period).dropna()
    
    if show_graph:
        fig, axs = plt.subplots(1,1, figsize=(10,5))
        axs.plot(TS_diff)
        axs.set_title("ST différenciée (period={})".format(str(period)))
        axs.set_xlabel("Temps")
        axs.set_ylabel("NB_DOSS_DAY")
        axs.axhline(y=0,c="black",alpha=.5, ls=':',lw=1)
        plt.show ()
    
    return TS_diff



# ***********************************************************************
# Entraînement modèle ARIMA ou SARIMA
# ***********************************************************************
def train_SARIMA (TS_train,TS_test=None,order=None,seasonal_order=None,trend=None,all_pred:bool=False,verbose:bool=True):
    """
    Input:
    ------
    - TS_train (pd.Series): série temporelle de train  
      avec en index une date et en valeur, la variable 
      d'intérêt.
    - TS_test (pd.Series) série temporelle de test  
      avec en index une date et en valeur, la variable 
      d'intérêt.
    - order (tuple): liste des 3 paramètres de ARIMA: order=(p,d,q)
      Si order=None, les paramètres sont déterminés avec le modèle 
      auto_arima de pmdarima.
    - seasonal_order (tuple): liste des 4 paramètres de SARIMA: seasonal_order=(P,D,Q,M)
      Si seasonal_order=None, on applique un modèle ARIMA.
    - trend (string ou liste): contrôle la tendance déterministe. Peut prendre les valeurs: 
      * 'n': pas de tendance
      * 'c': terme constant (valeur par défaut pour les modèles sans différenciation (d=0))
      * 't': tendance linéaire dans le temps, 
      * 'ct': inclut les deux. 
      * Peut également être une liste définissant un polynôme : [1,1,0,1] indique a+bt+ct^3 
      La valeur par défaut est 'c' pour les modèles sans différentiation,
      et aucune tendance pour les modèles avec différentiation.
      Si trend=None, les paramètres sont déterminés avec le modèle 
      auto_arima de pmdarima.
    - all_pred: si True, calcule les prédictions sur le jeu de test et train avec 'get_prediction'
      de statsmodels ARIMA, si False, calcule uniqumenet le forecast sur le jeu de test avec 
      'get_forecast'. 
    - verbose (bool): si True, affiche SARIMA.summary() et le plot
      des résidus.

    Output:
    ------
    - forecast_test (pd.series): valeurs prédites sur le jeu de test 
      (index: une date, valeur :variable d'intérêt).
    - conf_int (pd.DataFrame ou np.array): intervalle de confiance pour 
      les prédictions au niveau alpha=0.05.

    ======================== Notes ========================
    Le modèle ARIMA(p,d,q) sécrit avec :
    - p = ordre de la partie autoregressive AR(p)
    - d = degré de la différenciation
    - q = odre de la partie moving average MA(q)

    La prédiction ARIMA(p,d,q) s'écrit :
    Y't = c + ar.L1*Y_(t-1) + ar.Lp*Y_(t-p) + ma.L1*e_(t-1) + ma.Lq*e_(t-q) + e_t
    Les résidus du modèle sont:
    r_t = Y't - Yt où Y't*: valeur prédite et Yt: valeur réelle. 
    
     Le modèle SARIMA(p,d,q)(P,D,Q,M) sécrit avec :
    >>>> Partie non saisonnière: 
    - p = ordre de la partie autoregressive AR(p) 
    - d = degré de la différenciation
    - q = odre de la partie moving average MA(q)
    >>>> Partie saisonnière: 
    - P = ordre de la partie autoregressive saisonnière AR(P) 
    - D = degré de la différenciation saisonnière
    - Q = odre de la partie moving average saisonnière MA(Q)
    - M = saisonnalité 
    
    La prédiction SARIMA(p,d,q)(P,D,Q,M) est complexe et nous n'allons 
    pas l'écrire maintenant.

    Analyse de quelques éléments de la sortie model_fit.summary():

    ----- 2ème encadré: Test de significativité des coefficients. 

        - ar.L1,ar.L2: coefficients AR dans le modèle ARIMA
        - ma.L1, ma.L2: coefficients MA dans le modèle ARIMA
        - sigma2: terme d'erreur (e_t).
        - P>|z|: p-valeur du test de significativité. Si p-value < 0.05 
          alors le coefficient est significatif (rejette hypothèse nulle).

    ----- 3ème encadré: Autres tests
    ARIMA suppose que les résidus sont gaussiens, centrés, non corrélées entre
    eux (= bruit blanc) et qu'ils ont la même variance (homoscédasticité)

        - Ljung-Box: Ljung Box test, teste que les résidus soient un bruit blanc.
          Si p-value < 0.05, on rejette l'hypothèse nulle et 
          les résidus ne sont PAS du bruit blanc. 
        - Heteroscedasticity: White’s test, teste que les résidus de l'erreur ont la 
          même variance. Si p-value < 0.05, on rejette l'hypothèse nulle et 
          les résidus n'ont PAS la même variance. 
        - Jarque-Bera: Jarque-Bera test, teste la normalité des résidus. 
          Si p-value < 0.05, on rejette l'hypothèse nulle et 
          les résidus ne sont PAS gaussiens. 
    =======================================================


    Références:
    ----------
    - interprétation ARIMA summary: https://analyzingalpha.com/interpret-arima-results
    - livre de référence: https://otexts.com/fpp2/residuals.html
    - arima doc: https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html
    - prédiction: https://www.justintodata.com/arima-models-in-python-time-series-prediction/
    """

    # ==============================
    # Modèle ARIMA trouvé automatiquement
    # ==============================
    if order==None:
        # définition du modèle auto arima 
        model = pm.auto_arima(TS_train,stepwise=False, seasonal=False)
        # Fit le modèle sur les données d'entraînement
        model_fit = model.fit(TS_train)
        # prédiction et intervalle de confiance sur les données de test
        if all_pred: 
             forecast_test,conf_int = model_fit.predict_in_sample(start=0,end=len(TS_test)+len(TS_train)-1,
                                                                  return_conf_int=True)
        else:
            forecast_test,conf_int = model_fit.predict(len(TS_test),return_conf_int=True)
                              
        # Calcul des résidus 
        residuals = model_fit.resid()

    # ==============================
    # Modèle ARIMA trouvé manuellement
    # ==============================
    else:
        # définition du modèle auto arima 
        model = ARIMA(TS_train, order=order,seasonal_order=seasonal_order,trend=trend)    
        # Fit le modèle sur les données d'entraînement
        model_fit = model.fit(method = 'innovations_mle') #autre option: method='statespace'
        # prédiction set intervalle de confiance sur les données de test
        if all_pred: 
            # prédiction sur le train et le test 
            m = model_fit.get_prediction(start=0,end=len(TS_test)+len(TS_train)-1)
        else:
            # prédiction sur le test uniquement
            m = model_fit.get_forecast(steps=len(TS_test))
            
        forecast_test = m.predicted_mean
        conf_int = m.conf_int(alpha=0.05)
        conf_int = np.array(conf_int) #transformation de df à array
        # Calcul des résidus 
        residuals = model_fit.resid

    if verbose: 
        print(model_fit.summary())

        # ==============================
        # Graphique des résidus
        # ==============================
        fig = model_fit.plot_diagnostics(figsize=(14,10))   
        plt.show()

    return forecast_test,conf_int