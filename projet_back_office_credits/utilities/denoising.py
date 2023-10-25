"""
Ce fichier permet de définir les fonctions nécessaires 
au débruitage d'un signal univarié à partir de la transformée
en Ondelettes.

Alors que la transformée de Fourier crée une représentation du signal dans
le domaine des fréquences, la transformée en Ondelettes crée une représentation
du signal à la fois dans le domaine temporel et dans le domaine des fréquences, 
permettant ainsi un accès efficace à des informations localisées sur le signal. 
Le débruitage en Ondelettes permet aussi de débruiter des signaux plus irréguliers.

Contient les fonctions :
- psnr
- EstimEcartTypeBruit
- Debruit
- DebruitTrans
- plot_dash_debruitTrans
- plot_DebruitTrans

Contient les variables  :
- arg_dash
"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs
import numpy as np
import pandas as pd

# Graphes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Transformée en Ondelette
import pylab as pyl
import pywt
import scipy as scp

# Visualisation panel intéractif
import panel as pn
pn.extension()
from ipywidgets import interact, fixed
import ipywidgets as widgets

# ***********************************************************************
# Calcul du PSNR
# ***********************************************************************
def psnr(Sref, Sd):
    """
    Cette fonction calcule le PSNR permettant de quantifier 
    la qualité de reconstruction du signal compressé par 
    rapport au signal originale.
    Un PSNR plus élevé indique généralement que la reconstruction
    est de meilleure qualité. La valeur maximale est 100. 
    
    Input:
    -----
    - Sref (np.array): signal original
    - Sd (np.array): signal débruité. 
    
    Output:
    ------
    - PSNR (float): valeur du PSNR
    """
    mse = np.mean((Sref - Sd) ** 2)
    if mse == 0:
        return 100
    Val_MAX = max(Sref)
    PSNR = 20 * np.log10(Val_MAX / np.sqrt(mse))
    return PSNR


# ***********************************************************************
# Estimation Ecart type du bruit
# ***********************************************************************
def EstimEcartTypeBruit(sb,qmf):
    """

    Cette fonction estime l'écart-type du bruit d'un signal. 
    Cette méthode suppose un bruit gaussien. 
    
    
    Input:
    ------
    - sb (np.array): signal bruité
    - qmf (string): base d'ondelettes de PyWavelets ('haar','db2','db3'...)
      Pour le débruitage, les bases Symlet ('sym') ou Daubechies ('db') sont recommandée [2].
      Le numéro 'N' dans 'dbN' est le nombre de moments nuls de la base d'ondelettes.
      Plus il y a de moments nuls, plus la base d'ondelettes peut représenter des fonctions
      complexes avec peu de coefficients d'ondelette.
      (voir toutes les ondelettes disponibles: pywt.families() et pywt.wavelist('db'))
    
    Output:
    -------
    - sigma (float): estimation de l'écart type des coefficients
      de bruits.

    =================== Notes issues de [1] ===================
    On peut estimer la variance du bruit d'un signal en utilisant les coefficients d'ondelettes.
    En effet on peut exploiter le fait qu'aux fines échelles, l'essentiel des coefficients 
    sont dus au bruit. La moyenne des valeurs absolue des coefficients du signal bruité
    peut être lourdement impacté par les coefficients du signal mais pas la médiane. 
    Si X est une variable gaussienne centrée réduite, la médiane m de sa valeur absolue 
    vérifie P(X>m)=0.25. L'espérance de cette médiane des coefficients du bruit 
    est donnée par np.sqrt(2)*scp.special.erfinv(0.5) où scp.special.erfinv est 
    la l'inverse de la error function.
    ==========================================================

    Références:
    ---------
    Estimation du bruit:
    - Cours signal Ondelettes INSA 4A [1].
    - https://www.scielo.br/j/bcg/a/Sk5YwZC3vwNbBH6R3mJrQyC/?lang=en
    - https://ieeexplore.ieee.org/document/9321739 (télécharger le pdf ici:
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=
    2ahUKEwj846T43pT9AhWRUqQEHYpFA9EQFnoECAoQAQ&url=https%3A%2F%2Fnightlessbaron
    .github.io%2Ffiles%2Fpublications%2FDIP_IEEE.pdf&usg=AOvVaw0AmaueY3uK2N0e6b-1KXJo)
    - https://en.wikipedia.org/wiki/Error_function
    
    Quelle base d'ondelette choisir: 
    - https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html [2]
    - https://math.stackexchange.com/questions/128165/what-is-a-vanishing-moment
    """
    # niveau maximal de la décomposition en ondelettes
    Lmax=pywt.dwt_max_level(len(sb),pywt.Wavelet(qmf).dec_len)
    # décomposition en ondelettes
    wsb=pywt.wavedec(sb, qmf, mode='per', level=Lmax)
    # Espérance de la médiane des coefficients de bruit
    mt=np.sqrt(2)*scp.special.erfinv(0.5)
    # ecart-type du bruit
    sigma = np.median(np.abs(wsb[Lmax]))/mt
    return sigma



# ***********************************************************************
# Seuillage doux en Ondelettes
# ***********************************************************************
def SeuillageOndelette(SB,qmf,Seuil,L=None):
    """
    Cette fonction permet de faire un seuillage doux en Ondelettes. 
    
    Input:
    -----
    - SB (np.array): signal à débruiter.
    - qmf (string): base d'ondelettes de PyWavelets ('haar','db2','db3'...)
      Pour le débruitage, les bases Symlet ('sym') ou Daubechies ('db') sont recommandée [2].
      Le numéro 'N' dans 'dbN' est le nombre de moments nuls de la base d'ondelettes.
      Plus il y a de moments nuls, plus la base d'ondelettes peut représenter des fonctions
      complexes avec peu de coefficients d'ondelette.
      (voir toutes les ondelettes disponibles: pywt.families() et pywt.wavelist('db'))
    - L (int): niveau maximal de la transformée en Ondelettes. Si L=None, ce 
      niveau est caculé
    - Seuil (float): Seuil pour le seuillage doux en Ondelettes.
    
    Output:
    ------
    - Srec (np.array): signal débruité.
    
    ======================= Notes =======================
    Le bruit d'un signal est généralement concentré sur les coefficients
    d'ondelettes de plus haut niveau (les plus précis).
    On calcule les coefficients d'ondelettes du signal d'entrée SB. 
    On conserve uniquement les coefficients qui sont supérieurs à
    un certain seuil en valeur absolue. On fait la transformée 
    en Ondelettes inverse pour reconstruire le signal. 
    Le signal reconstruit est débruité.
    ====================================================
    
    Références:
    ---------
    - Cours signal Ondelettes INSA 4A.
    Quelle base d'ondelette choisir: 
    - https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html [2]
    - https://math.stackexchange.com/questions/128165/what-is-a-vanishing-moment
    """
    # Multilevel 1D Discrete Wavelet Transform of data.
    WTB= pywt.wavedecn(SB, qmf, mode='per', level=L)
    # Arrange a wavelet coefficient list from wavedecn into a single array.
    arr, coeff_slices = pywt.coeffs_to_array(WTB)
    # Selection des coefficients d'ondelettes supérieurs en valeur absolue au Seuil
    WTS=arr*(np.abs(arr)>Seuil)
    # Convert a combined array of coefficients back to a list compatible with waverecn.
    coeffs_from_arr = pywt.array_to_coeffs(WTS, coeff_slices)
    # Multilevel nD Inverse Discrete Wavelet Transform.
    Srec=pywt.waverecn(coeffs_from_arr,qmf,mode='per')
    # Si len(SB) est impair, alors la dernière valeur de Srec est doublée 
    if len(SB)%2 !=0:
        Srec=Srec[:-1] #on retire la dernière valeur 
    return Srec


# ***********************************************************************
# Débruitage à partir du Seuillage doux en Ondelettes
# ***********************************************************************
def Debruit(SB,qmf,T):
    """
    Cette fonction permet de faire le débruitage d'un signal 
    SB en utilisant les coefficients d'ondelette.
    Utilise la fonction 'EstimEcartTypeBruit' pour estimer le
    bruit du signal afin de calculer le seuil intervant dans 
    le calcul du seuillage doux en ondelettes de la fonction
    'SeuillageOndelette'.
    
    Input:
    -----
    - SB (np.array): signal à débruiter.
    - qmf (string): base d'ondelettes de PyWavelets ('haar','db2','db3'...)
      Pour le débruitage, les bases Symlet ('sym') ou Daubechies ('db') sont recommandée [2].
      Le numéro 'N' dans 'dbN' est le nombre de moments nuls de la base d'ondelettes.
      Plus il y a de moments nuls, plus la base d'ondelettes peut représenter des fonctions
      complexes avec peu de coefficients d'ondelette.
      (voir toutes les ondelettes disponibles: pywt.families() et pywt.wavelist('db'))
    - T (float): Le seuil du seuillage doux en ondelettes est calculé comme Seuil=T*sigma
      où sigma a été estimée par 'EstimEcartTypeBruit'.
    
    Output:
    ------
    - Srec (np.array): signal débruité.
    - PSNR (float): valeur du psnr
    
    Références:
    ---------
    - Cours signal Ondelettes INSA 4A.
    Quelle base d'ondelette choisir: 
    - https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html [2]
    - https://math.stackexchange.com/questions/128165/what-is-a-vanishing-moment
    """
    # Calcul du niveau maximal de décomposition en ondelettes 
    Lmax=pywt.dwt_max_level(len(SB),pywt.Wavelet(qmf).dec_len)
    # Estimation du bruit du signal avec les coefficients d'ondelette
    sigma = EstimEcartTypeBruit(SB,qmf)
    # Calcul du seuil pour le seuillage doux en ondelettes
    Seuil=T*sigma
    # Seuillage doux en ondelettes donnant le signal débruité
    Srec=SeuillageOndelette(SB,qmf,Seuil,Lmax)
    # Calcul du psnr entre le signal d'origine et le signal débruité
    PSNR=psnr(SB,Srec)
    return Srec,PSNR

# ***********************************************************************
# Débruitage avec Translations à partir du Seuillage doux en Ondelettes
# ***********************************************************************
def DebruitTrans(SB,qmf,T,trans):
    """
    Cette fonction permet de faire le débruitage d'un signal 
    SB en utilisant les coefficients d'ondelette.
    Utilise la fonction 'EstimEcartTypeBruit' pour estimer le
    bruit du signal afin de calculer le seuil intervant dans 
    le calcul du seuillage doux en ondelettes de la fonction
    'SeuillageOndelette'.

    De plus, la performance du débruitage par seuillage doux en Ondelettes 
    peut être améliorée en utilisant le fait que la transformée en ondelettes 
    n'est pas invariante par translation. Ainsi si on effectue un shift 
    circulaire sur les composantes d'un vecteur, on modifie l'amplitude
    des coefficients d'ondelettes. On peut exploiter cette propriété pour
    le débruitage en effectuant un débruitage d'un même signal dans des 
    bases d'ondelettes translatées.

    Input:
    -----
    - SB (np.array): signal à débruiter.
    - qmf (string): base d'ondelettes de PyWavelets ('haar','db2','db3'...)
      Pour le débruitage, les bases Symlet ('sym') ou Daubechies ('db') sont recommandée [2].
      Le numéro 'N' dans 'dbN' est le nombre de moments nuls de la base d'ondelettes.
      Plus il y a de moments nuls, plus la base d'ondelettes peut représenter des fonctions
      complexes avec peu de coefficients d'ondelette.
      (voir toutes les ondelettes disponibles: pywt.families() et pywt.wavelist('db'))
    - T (float): Le seuil du seuillage doux en ondelettes est calculé comme Seuil=T*sigma
      où sigma a été estimée par 'EstimEcartTypeBruit'.
    - trans (integer): valeur de la translation.

    Output:
    ------
    - Srec (np.array): signal débruité.
    - PSNR (float): valeur du psnr

    Références:
    ---------
    - Cours signal Ondelettes INSA 4A.
    Quelle base d'ondelette choisir: 
    - https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html [2]
    - https://math.stackexchange.com/questions/128165/what-is-a-vanishing-moment
    """
    # Calcul du niveau maximal de décomposition en ondelettes 
    Lmax=pywt.dwt_max_level(len(SB),pywt.Wavelet(qmf).dec_len)
    # Estimation du bruit du signal avec les coefficients d'ondelette
    sigma = EstimEcartTypeBruit(SB,qmf)
    # Calcul du seuil pour le seuillage doux en ondelettes
    Seuil=T*sigma
    SSum=0*SB
    # Vecteur contenant les psnr pour chaque translation
    P=np.zeros(trans)
    for k in np.arange(0,trans):
        # Translation du signal SB de k valeurs
        SBtemp=np.roll(SB,k)
        # Seuillage doux en ondelettes du signal translaté
        # donnant le signal reconstitué
        Srectemp=SeuillageOndelette(SBtemp,qmf,Seuil,Lmax)
        # Translation inverse du signal reconstitué de k valeurs
        Srectemp2=np.roll(Srectemp,-k)
        # Somme des signaux reconstitués
        SSum=SSum+Srectemp2
        # Calcule une moyenne des signaux reconstitués
        Srec=SSum/(k+1)
        # Calcule le psnr de la reconstitution
        P[k]=psnr(SB,Srec)
    # Calcul du PSNR moyen
    PSNR = np.mean(P)
    return Srec,PSNR


# ***********************************************************************
# Panel intéractif pour choisir les paramètres de débruitage
# ***********************************************************************
"""
Définition des arguments nécessaires pour l'affichage du 
dachbord intéractif.
Pour utiliser le dashbord intéractif, exécuter dans le notebook :
import denoising as dn
from denoising import arg_dash
from ipywidgets import interact, fixed
import ipywidgets as widgets
_ = interact(dn.plot_dash_debruitTrans, **arg_dash, TS_week=fixed(TS_week), TS_day=fixed(TS_day))


Reference:
----------
https://levelup.gitconnected.com/create-interactive-and-beautiful-dashboards-with-python-dfb760b416e6
"""

wavelist = pywt.wavelist('haar') + pywt.wavelist('db') + pywt.wavelist('sym')

arg_dash = dict(
    
    signal = widgets.ToggleButtons(
        name = 'signal :',
        value = 'TS_week',
        options = ['TS_week','TS_day'],
        disabled=False,
        button_style = 'success'),

    wave = widgets.Dropdown(
        name = 'base d\'ondelette :',
        value = 'haar',
        options = wavelist),

    seuil = widgets.FloatSlider(
        name = 'seuil :',
        value = 3,
        min = 0,
        max = 8,
        step = 0.1),

     translation = widgets.IntSlider(
         name = 'translation :',
         value = 3,
         min = 0,
         max = 8,
         step = 1),

    frame_day = widgets.IntRangeSlider(
        name = 'indices de début et fin',
        value = [0,2000], 
        min = 0,
        max = 2000,
        step = 5) ,
    
    frame_week = widgets.IntRangeSlider(
        name = 'indices de début et fin',
        value = [0,200], 
        min = 0,
        max = 200,
        step = 5) ,
)


def plot_dash_debruitTrans(TS_week,TS_day,signal,wave,seuil,translation,frame_week=[None,None],frame_day=[None,None]):
    """    
    Cette fonction permet d'afficher le débruitage d'un signal sous forme 
    d'un dashbord intéractif à l'aide de la fonction 'DebruitTrans'. 
    Pour plus d'info, voir la doc de 'DebruitTrans'.

    Pour utiliser le dashbord intéractif, exécuter dans le notebook :
    import denoising as dn
    from denoising import arg_dash
    from ipywidgets import interact, fixed
    import ipywidgets as widgets
    _ = interact(dn.plot_dash_debruitTrans, **arg_dash, TS_week=fixed(TS_week), TS_day=fixed(TS_day))

    Input:
    -----
    - TS_week (pd.Series):  série temporelle avec en index une date 
      et en valeur, le nombre de 'CODOSB' par semaine.
    - TS_day (pd.Series):  série temporelle avec en index une date 
      et en valeur, le nombre de 'CODOSB' par jour.
    - signal (string): signal à débruiter. Peut prendre les valeurs "TS_week"
      et "TS_day".
    - wave (string): base d'ondelettes de PyWavelets.
    - seuil (float): seuil du seuillage doux en ondelettes est calculé comme Seuil=T*sigma
      où sigma a été estimée par 'EstimEcartTypeBruit'.
    - translation (integer): valeur de la translation.   
    - frame_week (liste): bornes des abscisses sur lesquelles afficher le plot
      si signal = "TS_week" .
      Si frame_week=[None,None], toute la série temporelle est affichée.
    - frame_day (liste): bornes des abscisses sur lesquelles afficher le plot
      si signal = "TS_day" .
      Si frame_day=[None,None], toute la série temporelle est affichée.
    
    Output:
    ------
    - Graphiques 
    """
    
    if signal == "TS_week":
        SB = TS_week
        frame = frame_week
    elif signal == "TS_day":
        SB = TS_day
        frame = frame_day

    Srec,PSNR = DebruitTrans(SB,wave,seuil,translation)

    fig, axs = plt.subplots(2,1, figsize=(10,8))
    plt.subplots_adjust(hspace=0.5)

    # Graphique de la série temporelle d'origine
    #--------------------------------
    axs[0].plot(SB[slice(frame[0],frame[1])])
    axs[0].set_title("Signal d'origine")
    axs[0].set_xlabel("Temps")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel("NB_DOSS_DAY")
    axs[0].grid(linestyle=':')

    # Graphique de la série débruitée
    #--------------------------------        
    axs[1].plot(Srec[slice(frame[0],frame[1])])
    axs[1].set_title("Signal débruité\nPSNR={}".format(round(PSNR,2)))
    axs[1].set_xlabel("Temps")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].set_ylabel("NB_DOSS_DAY")
    axs[1].grid(linestyle=':')
    


# ***********************************************************************
# Graphique non intéractif pour afficher le débruitage d'un signal
# ***********************************************************************
def plot_DebruitTrans(SB,qmf,T,trans,frame=[None,None],fig_size=[10,8]):
    """
    Cette fonction permet d'afficher le débruitage d'un signal 
    à l'aide de la fonction 'DebruitTrans'. 
    Pour plus d'info, voir la doc de 'DebruitTrans'.

    Input:
    -----
    - SB (np.array): signal à débruiter.
    - qmf (string): base d'ondelettes de PyWavelets.
    - T (float): seuil du seuillage doux en ondelettes est calculé comme Seuil=T*sigma
      où sigma a été estimée par 'EstimEcartTypeBruit'.
    - trans (integer): valeur de la translation.
    - frame (liste): bornes des abscisses sur lesquelles afficher le plot.
      Si frame=[None,None], toute la série temporelle est affichée.
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 

    Output:
    ------
    - Srec (np.array): signal débruité.
    - PSNR (float): valeur du psnr
    """
    # Calcul du signal reconstitué
    Srec,PSNR = DebruitTrans(SB,qmf='haar',T=T,trans=trans)
    
    fig, axs = plt.subplots(2,1, figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(hspace=0.5)

    # Graphique de la série temporelle d'origine
    #--------------------------------
    axs[0].plot(SB[slice(frame[0],frame[1])])
    axs[0].set_title("Signal d'origine")
    axs[0].set_xlabel("Temps")
    axs[0].set_ylabel("NB_DOSS_DAY")
    axs[0].grid(linestyle=':')

    # Graphique de la série débruitée
    #--------------------------------
    axs[1].plot(Srec[slice(frame[0],frame[1])])
    axs[1].set_title("Signal débruité\nbase={},PSNR={}".format(qmf,round(PSNR,2)))
    axs[1].set_xlabel("Temps")
    axs[1].set_ylabel("NB_DOSS_DAY")
    axs[1].grid(linestyle=':')