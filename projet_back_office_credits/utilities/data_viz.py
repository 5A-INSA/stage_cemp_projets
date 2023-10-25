"""
Ce fichier contient les fonctions permettant de faire 
l'analyse exploiratoire du jeu de données CREDIT.

Contient les fonctions :
>> fonctions pour les variables quantitatives 
- is_outlier
- plot_hist_box
- plot_1_hist_box
- plot_pairs

>> fonctions pour les variables qualitatives  
- print_levels
- plot_bar

>> fonctions pour les variables qualitatives & quantitatives
- plot_na_null
- plot_corr
- round_up_to_even
- find_multiplied_nb 
- plot_box_cat_num

"""

# ***********************************************************************
# Importation des librairies nécessaires 
# ***********************************************************************
# Calculs 
import numpy as np
import pandas as pd 
from collections import Counter

# Graphes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ***********************************************************************
# Calcul des données manquantes & données nulles
# ***********************************************************************
def plot_na_null(df,var,fig_size,plot_null:bool):
    """
    Cette fonction trace pour chaque variable le pourcentage et le nombre 
    de valeurs manquantes et de valeurs nulles présentes dans le dataframe df.

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables pour lesquelles chercher les valeurs
      manquantes ou nulles. 
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 
    - plot_null (bool): si False, on ne trace que les valeurs manquantes.  
    
    Output:
    ------
    - Graphiques.
    """

    d_na = {}    #dict contenant le pourcentage de données NaN pour chaque variable   
    d_na_2 =  {} #dict contenant le nb de données NaN pour chaque variable   
    d_null = {}   #dict contenant le pourcentage de données nulles pour chaque variable  
    d_null_2 = {} #dict contenant le nb de données nulles pour chaque variable   

    # Calcul du % et du nb de valeurs manquantes et de valeurs nulles
    for v in var:
        nb_na = round(len(np.where(df[v].isna())[0])/len(df)*100,4) #pourcentage
        nb_na_2 = len(np.where(df[v].isna())[0]) #valeur réelle
        d_na[v] = nb_na 
        d_na_2[v] = nb_na_2

        if plot_null: #si plot_null, on calcule le nb et le % de valeurs nulles 
            nb_null = round(len(np.where(df[v]==0.0)[0])/len(df)*100,4) #pourcentage
            nb_null_2 = len(np.where(df[v]==0.0)[0]) #valeur réelle
            d_null[v] = nb_null
            d_null_2[v] = nb_null_2


    # Plot des données manquantes & données nulles
    # -------------------------------------------    
    if plot_null: 
        fig, axs = plt.subplots(2, 1, figsize=(fig_size[0], fig_size[1]))
        fig.subplots_adjust(hspace=.5)

        #---- Valeurs manquantes 
        axs[0].bar(d_na.keys(), d_na.values(), width=.3, color='b',alpha=.3)
        axs[0].set_xticklabels(d_na.keys(), rotation = 45, ha="right")
        axs[0].grid(linestyle=':')
        axs[0].set_title("Pourcentage de valeurs manquantes pour chaque variable")
        axs[0].set_ylabel("pourcentage"); axs[0].set_xlabel("variables")
        axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        # Affiche le nb de valeurs NaN au dessus des plots
        j = 0
        for index,data in enumerate(d_na_2.values()):
            axs[0].text(x=index-0.2,y =list(d_na.values())[j],s=f"{data}",
                        fontdict=dict(fontsize=10),color='b',alpha=.7)
            j+=1

        #---- Valeurs nulles 
        axs[1].bar(d_null.keys(), d_null.values(), width=.3, color='g',alpha=.3)
        axs[1].set_xticklabels(d_null.keys(), rotation = 45, ha="right")
        axs[1].grid(linestyle=':')
        axs[1].set_title("Pourcentage de valeurs nulles pour chaque variable")
        axs[1].set_ylabel("pourcentage"); axs[1].set_xlabel("variables")
        axs[1].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        # Affiche le nb de valeurs NaN au dessus des plots
        j = 0
        for index,data in enumerate(d_null_2.values()):
            axs[1].text(x=index-0.2, y =list(d_null.values())[j],s=f"{data}",
                        fontdict=dict(fontsize=10),color='g')
            j+=1

    # Plot des données manquantes uniquement
    # -------------------------------------------  
    else: 
        fig, axs = plt.subplots(1,1, figsize=(fig_size[0], fig_size[1]))
        axs.bar(d_na.keys(), d_na.values(), width=.3, color='b',alpha=.3)
        axs.set_xticklabels(d_na.keys(), rotation = 45, ha="right")
        axs.grid(linestyle=':')
        axs.set_title("Pourcentage de valeurs manquantes pour chaque variable")
        axs.set_ylabel("pourcentage"); axs.set_xlabel("variables")
        axs.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # Affiche le nb de valeurs NaN au dessus des plots
        j = 0
        for index,data in enumerate(d_na_2.values()):
            axs.text(x=index-0.2,y =list(d_na.values())[j],s=f"{data}",
                        fontdict=dict(fontsize=10),color='b',alpha=.7)
            j+=1


    plt.show()
    

# ***********************************************************************
# Histogrammes ou boxplot de toutes les variables quantitatives
# ***********************************************************************

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
    - https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting
    - Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
      Handle Outliers", The ASQC Basic References in Quality Control:
      Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



def plot_hist_box(df,var,type_plot,layout=[3,4],fig_size=[20,20]):
    """
    Cette fonction trace les histogrammes ou les boxplots des variables 
    var du dataframe df. Requiert la fonction is_outlier qui enlève 
    les outliers pour le tracé des histogrammes afin de ne pas applatir 
    le graphe. 

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables quantitatives pour lesquelles tracer
      les histogrammes ou les boxplots
    - type_plot (str): prend les valeurs 'hist' ou 'box' et indique si 
      on trace les histogrammes ou les boxplots.
    - layout (list): liste de 2 valeurs indiquant la disposition des 
      subplots. Par exemple, pour tracer 10 histogrammes, on peut 
      entrer layout = [5,2] et on aura une matrice 5x2 de plots. 
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 

    Output:
    ------
    - Graphiques.

    Reference:
    ---------
    https://stackoverflow.com/questions/17210646/python-subplot-
    within-a-loop-first-panel-appears-in-wrong-position
    """

    if type_plot == 'hist':
        kind = 'Histogramme'
    elif type_plot == 'box':
        kind = 'Boxplot'
    else:
        print("Pour type_plot entrer 'hist' ou 'box'")

    fig, axs = plt.subplots(layout[0],layout[1], figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(kind + " des variables quantitatives", fontsize=16, y=0.95)
    axs = axs.ravel()

    #---- Parcourt les variables et trace leur histogramme
    i=0
    for v in var:
        if type_plot == 'hist':
            df_v = df[v][~is_outlier(df[v])] #colonne de la variable v sans les outliers
            sns.histplot(ax=axs[i], data=df_v, edgecolor='w',alpha=0.6)
        elif type_plot == 'box':
            df_v = df[v]
            sns.boxplot(ax=axs[i], data=df_v, saturation=0.75,
            flierprops = dict(markerfacecolor = '0.5', markersize = 3))
        axs[i].set_title(kind + " de " + v)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        i+=1
    
    n_var = len(var); n_plots = layout[0]*layout[1]
    if n_plots > n_var:
        for a in range(n_var,n_plots):
                fig.delaxes(axs[a])
    plt.show()


def plot_1_hist_box(df,var):
    """
    Cette fonction trace l'histogramme et le boxplot de la variable 
    var du dataframe df. Requiert la fonction is_outlier qui enlève 
    les outliers pour le tracé des histogrammes afin de ne pas applatir 
    le graphe. 

    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variable quantitative pour laquelle tracer
      l'histogramme et le boxplot.

    Output:
    ------
    - Graphique.

    Reference:
    ---------
    https://stackoverflow.com/questions/17210646/python-subplot-
    within-a-loop-first-panel-appears-in-wrong-position
    """
         
    fig, axs = plt.subplots(1,2,figsize=(10, 4),gridspec_kw={'width_ratios':[2,1]})
    fig.suptitle(" Histogramme & Boxplot de " + var, fontsize=16, y=0.95)    
    # tracé histograme
    df_v = df[var][~is_outlier(df[var])] #colonne de la variable v sans les outliers
    sns.histplot(ax=axs[0], data=df_v, edgecolor='w',alpha=0.6)
    # tracé boxplot
    df_v = df[var]
    sns.boxplot(ax=axs[1], data=df_v, saturation=0.75,
                flierprops = dict(markerfacecolor = '0.5', markersize = 3))
        
    sp = ['right', 'top', 'bottom', 'left']
    axs[0].spines[sp].set_visible(False); axs[1].spines[sp].set_visible(False)
    axs[0].grid(linestyle = ':'); axs[1].grid(linestyle = ':')
    plt.show()
    

# ***********************************************************************
# Tracé des pairplots pour les variables quantitatives
# ***********************************************************************

def plot_pairs(df,var,fig_size,save_path=None,save_name=None):
    """
    Cette fonction trace les pairplots pour les variables*
    quantitatives var de la dataframe df.  
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables quantitatives pour lesquelles tracer
      les graphe
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 
    - save_path (str): chemin du dossier vers lequel sauvegarder 
      les graphes. Si none, les graphes ne sont pas sauvegardés. 
    - save_name (str): nom du graphe sauvegardé.

    Output:
    ------
    - Graphiques.
    """
    # Tracé des graphiques
    sns.set_style('darkgrid')
    g = sns.pairplot(data=df[var], diag_kind='hist', palette='CMRmap_r',diag_kws={'color':'green'},plot_kws={"s": 8})
    g.fig.suptitle("Pairplot des variables quantitatives", y=1.08,fontsize=18) 
    g.fig.set_size_inches(fig_size[0],fig_size[1])
    
    # Sauvegarde
    if save_path:
        g.figure.savefig(save_path + '/' + save_name + ".png")



# ***********************************************************************
# Matrices de corrélations
# ***********************************************************************

def plot_corr(df,var,type_corr,fig_size=[8,6],save_path=None,save_name=None):
    """
    Cette fonction trace la matrice de corrélation pour les variables
    quantitatives var du dataframe df. Le type de corrélation choisi
    est indiqué par type_corr.
    /.\ Attention, la corrélation phik est très longue à run.
        Il faut au préalable avoir transformé les variables 
        qualitatives en categorical. 
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables quantitatives pour lesquelles tracer
      le graphe.
    - type_corr (str): type de corrélation à tracer. Peut prendre 
      les valeurs 'pearson','kendall','spearman','phik'. 
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 
    - save_path (str): chemin du dossier vers lequel sauvegarder 
      les graphes. Si none, les graphes ne sont pas sauvegardés. 
    - save_name (str): nom du graphe sauvegardé. 

    Output:
    ------
    - Graphiques.
    
    # =========================== Notes ============================
    Les différents types de corrélations:
    The correlation is a value between -1 and +1, -1 indicating total negative 
    monotonic correlation, 0 indicating no monotonic correlation and 1 indicating 
    total positive monotonic correlation.

    - Pearson's r : The Pearson's correlation coefficient (r) measures the degree
      of the relationship between linearly related variables. Furthermore, r is 
      invariant under separate changes in location and scale of the two variables. 
      This correlation is based on the values and is sensitive to outliers. 
      To calculate r for two variables X and Y, one divides the covariance of X and
      Y by the product of their standard deviations.

    - Spearman's ρ : The Spearman's rank correlation coefficient (ρ) is a measure of
      monotonic correlation between two variables, and is therefore better in catching 
      nonlinear monotonic correlations than Pearson's r. The Spearman test is based on ranks
      and does not carry any assumptions about the distribution of the data and is the appropriate
      correlation analysis when the variables are measured on a scale that is at least ordinal. 
      To calculate ρ for two variables X and Y, one divides the covariance of the rank variables
      of X and Y by the product of their standard deviations.

    - Kendall's τ : Similarly to Spearman's rank correlation coefficient. Is is an alternative
      to Spearman's, allowing better management of ex-æquos but also being more pessimistic.

    - Phik (φk) : new and practical correlation coefficient that has the same characteristics
      as Pearson's but works consistently between categorical, ordinal and interval variables, 
      captures non-linear dependency. Paper: https://arxiv.org/abs/1811.11440 (2018). 
      It is based on several refinements to Pearson’s χ2 (chi-squared) contingency test 
      (a hypothesis test for independence between two (or more) variables).

    ⇒  On utilisera spearman et phik. 
    # ==============================================================

    Reference:
    ---------
    phik librairie : https://phik.readthedocs.io/en/latest/phik.html
    """ 
    # Calcul de la matrice de covariance
    if type_corr == 'phik':
        #Correlation matrix of bivariate gaussian derived from chi2-value:
        corr = df[var].phik_matrix()
        
    else:   
        corr = df[var].corr(method=type_corr)
        
   # Tracé de la matrice de covariance
    fig, ax = plt.subplots(figsize=(fig_size[0],fig_size[1]))  
    g = sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                annot=True,
                annot_kws={"fontsize":8},ax=ax)
    ax.set_title("Matrice de corrélation {}".format(type_corr))
    # Sauvegarde
    if save_path:
        fig = g.get_figure()
        fig.savefig(save_path + '/' + save_name + ".png")    


# ***********************************************************************
# Affiche les modalités des variables qualitatives 
# ***********************************************************************

def print_levels(df,var,fig_size,verbose:bool=False):
    """
    Cette fonction trace pour chaque variable qualitative var du
    dataframe df, le nombre de modalités.
    
    Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (list): liste des variables qualitatives
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des plots. 
    - verbose (bool): si True, affiche un texte décrivant le 
      nombre de modalités. 
      
    Output:
    ------
    - Graphiques.
    """
    
    #---- Calcul du nombre de modalités
    # Dictionnaire contenant en key la variable et 
    # en value le nb de modalités de la variable
    d_nb_mod = {}  

    for v in var:
        #dict contenant en clé la modalité et en value le nb d'occurences dans le dataset
        d = Counter(df[v]) 
        d_nb_mod[v] = len(d.keys())
    
    #---- Affichage si verbose=True
    if verbose: 
        print("Nombre de modalités pour chaque variable:")
        print("-----------------------------------------")
        for v in var:
            print("{} possède {} modalités".format(v,d_nb_mod[v]))
            
    #---- Tracé des graphes
    fig, axs = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    axs.bar(d_nb_mod.keys(), d_nb_mod.values(), width=.3, color='b',alpha=.3)
    axs.set_xticklabels(d_nb_mod.keys(), rotation = 45, ha="right")
    axs.grid(linestyle=':')
    axs.set_title("Nombre de modalités pour chaque variable")
    axs.set_ylabel("nombre de modalités"); axs.set_xlabel("variables")
    axs.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    # Affiche le nb de valeurs NaN au dessus des plots
    j = 0
    for index,data in enumerate(d_nb_mod.values()):
        axs.text(x=index-0.2,y =list(d_nb_mod.values())[j],s=f"{data}",
                    fontdict=dict(fontsize=10),color='b',alpha=.7)
        j+=1


# ***********************************************************************
# Affiche les barplots des variables qualitatives 
# ***********************************************************************
 
def plot_bar(df,var,layout,fig_size=[8,8]):
    """
    Cette fonction trace les barplots des variables var 
    de la dataframe df.
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var (str): variables qualitatives pour lesquelles 
      tracer les barplots.
    - layout (list): liste de 2 valeurs indiquant la disposition des 
      subplots. Par exemple, pour tracer 10 histogrammes, on peut 
      entrer layout = [5,2] et on aura une matrice 5x2 de plots. 
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 

    Output:
    ------
    - Graphiques.    
    """
    fig, axs = plt.subplots(layout[0],layout[1], figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(hspace=.5,wspace=.3)
    fig.suptitle("Barplot des variables qualitatives", fontsize=16, y=0.95)
    axs = axs.ravel()

    #---- Parcourt les variables et trace leur barplot
    i=0
    for v in var:
        #dict contenant en clé la modalité et en value le nb d'occurences dans me dataset
        d = Counter(df[v]) 
        axs[i].bar(x=list(d.keys()),height=list(d.values()),color='b',alpha=0.5)

        axs[i].set_title("Barplot de " + v)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylabel("nb d'occurrences")
        axs[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        i+=1

    plt.show()


# ***********************************************************************
# Trace les boxplots variables quantitiatives~qualitatives
# ***********************************************************************
import math
def round_up_to_even(f):
    """
    Donne la valeur entière paire supérieure à f la plus proche. 
    """
    return math.ceil(f / 2.) * 2

def find_multiplied_nb(x:int):
    """
    Cette fonction prend en entrée un integer x et renvoie un tuble (a,b)
    tels que a*b=x où a et b sont les plus proches possibles.
    Ex: 16 = 1*16 = 2*8 = 4*4. La fonction renverra (4,4) car ce sont 
    les nombres les plus proches entre eux. 
    
    Ref: https://stackoverflow.com/questions/42351977/how-to-split-an
    -integer-into-two-integers-that-when-multiplied-they-give-the-res
    """
    # trouve toutes les combinaisons possibles de (a,b) telles que x = a*b
    possible_combinations = [(i, x / i) for i in range(1, int(math.ceil(x**0.5)) + 1) if x % i == 0]
    # liste des différentes valeurs de |a-b|
    absolute_diff = [] 
    for item in possible_combinations:
        absolute_diff.append(abs(item[0]-item[1]))
    # On veut a et b les plus proches entre eux tq a*b=x
    # on prend donc la combinaison tel que |a-b| soit minimal
    best_combi = possible_combinations[np.argmin(absolute_diff)]
    
    return (int(best_combi[0]),int(best_combi[1]))



def plot_box_cat_num(df,var_quanti,var_quali,fig_size,layout=None,hide_labels:bool=False):
    """
    Cette fonction trace les boxplots des variables var_quanti~var_quali 
    du dataframe df. Nécessite les fonctions round_up_to_even et 
    find_multiplied_nb pour calculer la matrice des subplots.
    
     Input:
    ------
    - df (dataframe): dataframe contenant les données.
    - var_quanti (str): variables quantitatives pour lesquelles 
      tracer les boxplots.
    - var_quanli (str): variables qualitatives pour lesquelles 
      tracer les boxplots.
    - fig_size (list): liste de 2 valeurs indiquant la taille 
      des subplots. 
    - layout (list): liste de 2 valeurs indiquant la taille de
      la matrice des subplots. Si None, la taille est calculée
      automatiquement.
    - hide_labels (bool): si True, n'affiche pas les x labels.

    Output:
    ------
    - Graphiques.    
    
    Reference:
    ----------
    https://stackoverflow.com/questions/57891028/
    unable-to-recreate-catplot-with-matplotlib
    """

    #----- Calculs préliminaires
    # Calcul du nb total de variables
    n_var = int(len(var_quali) *len(var_quanti))
    print("{} variables".format(n_var))

    # CAS 1: Plusieurs boxplots
    # -------------------------
    if n_var > 1:
        # Trouve l'entier pair le plus proche supérieur à n_var 
        n_var2 = round_up_to_even(n_var)
        # Calcul de la taille de la matrice des subplots 
        if not(layout):
            layout = find_multiplied_nb(n_var2)

        #----- Tracé des graphiques
        fig, axs = plt.subplots(layout[0],layout[1], figsize=(fig_size[0], fig_size[1]))
        plt.subplots_adjust(hspace=0.5,wspace=0.3)
        fig.suptitle("Boxplots des variables quantitatives ~ qualitatives", fontsize=16, y=0.95)
        axs = axs.ravel()

        i=0
        for v_num in var_quanti:
            for v_cat in var_quali:
                # dataframe réduit à 2 variables: une quantitative et une qualitative
                df_v = pd.DataFrame(data = {'V_QUALI': df[v_cat],
                                            'V_QUANTI': df[v_num]})
                
                # labels de var_quali. On trie car plus bas, groupby trie par
                # ordre alphabétique et on veut que chaque boxplot ait le bon label. 
                labels = list(np.sort(df_v.V_QUALI[df_v.V_QUALI.notna()].unique()))
                
                # on retire les NaN sinon le boxplot peut ne pas apparaître
                df_v.dropna(axis=0,inplace = True)

                axs[i].boxplot([col.V_QUANTI.values for n, col in df_v.groupby(by="V_QUALI",sort=True)],patch_artist=False,
                           labels=labels,flierprops = dict(markerfacecolor = '0.5', markersize = 2))
                axs[i].set_xlabel(v_cat)
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].set_ylabel(v_num)
                axs[i].set_title("Boxplot {}~{}".format(v_num,v_cat))
                if hide_labels:
                    axs[i].set_xticks([])
                
                i+=1
        # suppression des subplots en trop
        n_plots = layout[0]*layout[1]
        if n_plots > n_var:
            for a in range(n_var,n_plots):
                fig.delaxes(axs[a])
        plt.show()


    # CAS 2: Un boxplot
    # -------------------------
    else: 
        v_cat=var_quali[0]; v_num=var_quanti[0]
        
        #----- Tracé des graphiques
        fig, axs = plt.subplots(figsize=(fig_size[0], fig_size[1]))
        # dataframe réduit à 2 variables: une quantitative et une qualitative
        df_v = pd.DataFrame(data = {'V_QUALI': df[v_cat],
                                    'V_QUANTI': df[v_num]})
        
        # labels de var_quali. On trie car plus bas, groupby trie par
        # ordre alphabétique et on veut que chaque boxplot ait le bon label.
        labels = list(np.sort(np.unique(df_v.V_QUALI[df_v.V_QUALI.notna()])))
        # on retire les NaN sinon le boxplot peut ne pas apparaître
        df_v.dropna(axis=0,inplace = True)
        axs.boxplot([col.V_QUANTI.values for n, col in df_v.groupby(by="V_QUALI",sort=True)],patch_artist=False,
                   labels=labels,flierprops = dict(markerfacecolor = '0.5', markersize = 2))
        
        axs.set_xlabel(var_quali)
        axs.tick_params(axis='x', rotation=45)
        axs.set_ylabel(var_quanti)
        axs.set_title("Boxplot {}~{}".format(var_quanti,var_quali))
        if hide_labels:
            axs.set_xticks([])

    # Rq: liste de array de taille (nb modalités de V_QUALI)
    # Chaque élément de la liste est un tableau contenant 
    # les valeurs que prend V_QUANTI pour une modalité de var_quali donnée 
    # [col.V_QUANTI.values for n, col in df.groupby(by="V_QUALI")] 