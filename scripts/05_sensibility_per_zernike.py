import sys
from pathlib import Path            
import numpy as np                    

root = Path(__file__).resolve().parent.parent         
sys.path.append(str(root)) 

from src import sensibility as sb
from src import metrics as mt
from src import dataset_tools as dt

# -----------------------------------------
# Variables des conditions
# -----------------------------------------

# Fichiers à considérer 
base_dir = r"M:\everyone\Laetitia\AO\BANDPASS\CARAC\CARAC_AO_THESE"
profondeur = "Coverslip"
quantite = "No_aber"
mode_zernike = "Zer4"
jeux = (1, 2, 3, 4, 5, 6, 7)
metrique = "MIP"

# Activer/désactiver la visualisation intermédiaire des résultats
PLOT_A = True
PLOT_N = True

# Mode de calcul du bruit N
METHODE = "std" #"std" (moyenne ± écart-type) ou "minmax" (vrai min/max observé)
MODE = "fit" #"direct" (aire sur les points bruts) ou "fit" (aire après fit quadratique)
ALPHAMAX = 0.26

# -----------------------------------------
# Main sensibilité sur 1 mode de Zernike 
# -----------------------------------------

# 1. Aller chercher les images + calculer la métrique + construire la matrice
x, matrice, jeux_utils = sb.obtenir_matrice_metrique(base_dir, profondeur, quantite, mode_zernike, metrique, jeux)
x = np.array(x)
x = x * ALPHAMAX / 4
# 2. Normaliser par ligne (chaque jeu par son propre max)
matrice_norm = sb.normaliser_par_ligne(matrice)

# 3. Calculer a pour chaque jeu, puis moyenner
liste_a, liste_x_max = zip(*(sb.coefficient_a(jeu_valeurs, x, PLOT_A) for jeu_valeurs in matrice_norm))

a_moyen = float(np.mean(liste_a))
x_max_moyen = float(np.mean(liste_x_max))

# 4. Calculer N (sur les données normalisées, transposées : une ligne par x, une valeur par jeu)
courbes_norm = list(zip(*matrice_norm))
N = sb.coefficient_N(courbes_norm, x, METHODE, MODE, PLOT_N)

print(f"b/2a moyen : {x_max_moyen}")
print(f"a moyen   : {a_moyen}  (sur {len(jeux_utils)} jeux)")
print(f"N         : {N}")