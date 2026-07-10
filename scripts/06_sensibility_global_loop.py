import sys
from pathlib import Path
import csv
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src import sensibility as sb
from src import metrics as mt
from src import dataset_tools as dt

# -----------------------------------------
# Dossier de sortie
# -----------------------------------------
BASE_DIR = r"M:\everyone\Laetitia\AO\BANDPASS\CARAC\CARAC_AO_THESE"
OUTPUT_DIR = Path(r"C:\Users\slesinq\Documents")  # <-- à adapter
OUTPUT_CSV = OUTPUT_DIR / "resultats_analyse.csv"

# -----------------------------------------
# Conditions à parcourir
# -----------------------------------------
PROFONDEURS = [
    "Coverslip",
    #"Profondeur",
]

QUANTITES_BY_PROFONDEUR = {
    "Coverslip": [
        "No_aber",
        #"Low_aber",
        #"Medium_aber",
        #"High_aber",
    ],
    "Profondeur": [
        "No_aber",
        "Low_aber",
        "Medium_aber",
        "High_aber",
        "High2",
    ],
}

MODES_ZERNIKE = [
    "Zer4",
    "Zer5",
    "Zer6",
    "Zer7",
    "Zer8",
    "Zer9",
    "Zer10",
]

METRIQUES = [
    "mean_ROI",
    "MIP",
    #"STD",
    #"SUM",
    #"Gallery",
    #"Frame",
    "Dapi",
]

JEUX = (1, 2, 3, 4, 5, 6, 7)  # la gestion des dossiers manquants s'occupe du reste

# Réglages fixes
PLOT_A = False   # désactivé pour une boucle longue (sinon ça ouvre une fenêtre par itération !)
PLOT_N = False
METHODE = "std"
MODE = "direct"
ALPHAMAX = 0.26

# -----------------------------------------
# Boucle principale
# -----------------------------------------
sb.create_analysis_csv(OUTPUT_CSV)

combinaisons = [
    (profondeur, quantite, mode_zernike, metrique)
    for profondeur in PROFONDEURS
    for quantite in QUANTITES_BY_PROFONDEUR[profondeur]
    for mode_zernike in MODES_ZERNIKE
    for metrique in METRIQUES
]

for i, (profondeur, quantite, mode_zernike, metrique) in enumerate(combinaisons, start=1):
    print(f"\n[{i}/{len(combinaisons)}] {profondeur} | {quantite} | {mode_zernike} | {metrique}")

    try:
        x, matrice, jeux_utilises = sb.obtenir_matrice_metrique(
            BASE_DIR, profondeur, quantite, mode_zernike, metrique, JEUX
        )
        x = np.array(x)
        x = x * ALPHAMAX / 4

        matrice_norm = sb.normaliser_par_ligne(matrice)

        liste_a, liste_x_max = zip(*(
            sb.coefficient_a(jeu_valeurs, x, PLOT_A) for jeu_valeurs in matrice_norm
        ))
        a_moyen = float(np.mean(liste_a))
        x_max_moyen = float(np.mean(liste_x_max))

        courbes_norm = list(zip(*matrice_norm))
        N = sb.coefficient_N(courbes_norm, x, METHODE, MODE, PLOT_N)

        S = abs(a_moyen) / N

        values = [
            metrique, profondeur, quantite, str(jeux_utilises), mode_zernike,
            abs(a_moyen), N, S, x_max_moyen, METHODE, MODE
        ]
        sb.append_analysis_result(OUTPUT_CSV, values)

    except Exception as e:
        print(f"⚠️ Combinaison ignorée à cause d'une erreur : {e}")

print(f"\nTerminé. Résultats dans : {OUTPUT_CSV}")