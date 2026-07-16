import sys
from pathlib import Path
import csv
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from src import sensibility as sb
from src import metrics as mt
from src import dataset_tools as dt
import config as cf

# -----------------------------------------
# Dossier de sortie
# -----------------------------------------
# BASE_DIR = r"M:\everyone\Laetitia\AO\BANDPASS\CARAC\CARAC_AO_THESE"
BASE_DIR = cf.DATA_ROOT_DIR

# OUTPUT_DIR = Path(r"C:\Users\slesinq\Documents")  # <-- à adapter
OUTPUT_DIR = cf.RESULTS_DIR

OUTPUT_CSV = cf.SENSITIVITY_DIR

# -----------------------------------------
# Conditions à parcourir
# -----------------------------------------
CONDITIONS = [
    "Coverslip",
    #"Profondeur",
]

ABERRATION_LEVEL_BY_PROFONDEUR = {
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
    #"mean_ROI",
    "MIP",
    #"STD",
    #"SUM",
    #"Gallery",
    #"Frame",
    #"Dapi",
]

JEUX = (1, 2, 3, 4, 5, 6, 7)  # la gestion des dossiers manquants s'occupe du reste

# Réglages fixes
PLOT_A = False   # désactivé pour une boucle longue (sinon ça ouvre une fenêtre par itération !)
PLOT_N = True
SAVE_METRIC_POINT_AND_FIT = True    # Sauveagrde des points calculés pour chaque métrique et conditins ainsi que le fit associés
METHODE = "std"     #"std" (moyenne ± écart-type) ou "minmax" (vrai min/max observé)
MODE = "fit"     #"direct" (aire sur les points bruts) ou "fit" (aire après fit quadratique)
ALPHAMAX = 0.26

# -----------------------------------------
# Petites fonctions locales
# -----------------------------------------

def build_sensitivity_base_name(output_csv, method, mode):
    """
    Construit un nom de bse pour la sauvegarde des résultats
    """
    base_name = (
        f"Sensitivity_{method}"
        f"_{mode}"
        ".csv"
    )
    
    return base_name
   

# -----------------------------------------
# Boucle principale
# -----------------------------------------
filename = build_sensitivity_base_name(OUTPUT_CSV, METHODE, MODE)
sb.create_analysis_csv(OUTPUT_CSV / filename)

combinaisons = [
    (condition, aberration_level, mode_zernike, metrique)
    for condition in CONDITIONS
    for aberration_level in ABERRATION_LEVEL_BY_PROFONDEUR[condition]
    for mode_zernike in MODES_ZERNIKE
    for metrique in METRIQUES
]

points_rows = []
fit_rows = []
envelope_rows = []

for i, (condition, aberration_level, mode_zernike, metrique) in enumerate(combinaisons, start=1):
    print(f"\n[{i}/{len(combinaisons)}] {condition} | {aberration_level} | {mode_zernike} | {metrique}")

    try:
        x, matrice, jeux_utilises = sb.obtenir_matrice_metrique(
            BASE_DIR, condition, aberration_level, mode_zernike, metrique, JEUX
        )
        x = np.array(x)
        x = x * ALPHAMAX / 4

        matrice_norm = sb.normaliser_par_ligne(matrice)

        # Sauvergarde optionnelle des valeurs de métriques et des fit associé
        if SAVE_METRIC_POINT_AND_FIT:
            for  jeu, y_raw, y_norm, in zip(jeux_utilises, matrice, matrice_norm):
                for alpha_value, raw_value, norm_value in zip(x, y_raw, y_norm):
                    points_rows.append({
                        "metrique": metrique,
                        "condition": condition,
                        "aberration_level": aberration_level,
                        "zernike": mode_zernike,
                        "jeu": jeu,
                        "alpha": alpha_value,
                        "metric_raw": raw_value,
                        "metric_norm": norm_value,
                    })

        # ===========================================================================
        # Ancien code sans sauvegarde du détail des valeurs des métriques et des fits
        """  
        liste_a, liste_x_max = zip(*(
            sb.coefficient_a(jeu_valeurs, x, PLOT_A) for jeu_valeurs in matrice_norm
        ))
        """
        # Remplace par:
        liste_a = []
        liste_x_max = []

        for jeu, jeu_valeurs in zip(jeux_utilises, matrice_norm):

            fit_details = sb.coefficient_a_details(jeu_valeurs, x, plot=PLOT_A)

            liste_a.append(fit_details["a"])
            liste_x_max.append(fit_details["x_max"])

            if SAVE_METRIC_POINT_AND_FIT:
                for x_fit, y_fit in zip(fit_details["x_fit"], fit_details["y_fit"]):
                    fit_rows.append({
                        "fit_type": "quadratic_per_jeu",
                        "metrique": metrique,
                        "condition": condition,
                        "aberration_level": aberration_level,
                        "zernike": mode_zernike,
                        "jeu": jeu,
                        "x_fit": x_fit,
                        "y_fit": y_fit,
                        "a": fit_details["a"],
                        "b": fit_details["b"],
                        "c": fit_details["c"],
                        "r2": fit_details["r2"],
                        "x_max": fit_details["x_max"],
                        "methode": METHODE,
                        "mode": MODE,
                    })
        # ===========================================================================

        a_moyen = float(np.mean(liste_a))
        x_max_moyen = float(np.mean(liste_x_max))

        # ===========================================================================
        # Ancien code sans sauvegarde du détail des valeurs des métriques et des fits
        """
        courbes_norm = list(zip(*matrice_norm))
        N = sb.coefficient_N(courbes_norm, x, METHODE, MODE, PLOT_N)
        """
        # Remplace par:
        courbes_norm = list(zip(*matrice_norm))

        N_details = sb.coefficient_N_details(
            courbes_norm,
            x,
            methode=METHODE,
            mode=MODE,
            plot=PLOT_N
        )

        N = N_details["N"]
        # ===========================================================================

        # Sauvegarde des points bruts et fits
        if SAVE_METRIC_POINT_AND_FIT:

            # Points moyens et enveloppes aux x expérimentaux
            for alpha_value, y_mean, y_min, y_max, diff_direct in zip(
                N_details["x_points"],
                N_details["y_mean"],
                N_details["y_min"],
                N_details["y_max"],
                N_details["diff_direct"],
            ):
                envelope_rows.append({
                    "curve_type": "points",
                    "metrique": metrique,
                    "condition": condition,
                    "aberration_level": aberration_level,
                    "zernike": mode_zernike,
                    "alpha": alpha_value,
                    "y_mean": y_mean,
                    "y_min": y_min,
                    "y_max": y_max,
                    "diff": diff_direct,
                    "N": N,
                    "N_direct": N_details["N_direct"],
                    "methode": METHODE,
                    "mode": MODE,
                })

            # Courbes fit ymin / ymax si mode='fit'
            for x_fit, y_min_fit, y_max_fit, diff_fit in zip(
                N_details["x_fit"],
                N_details["y_min_fit"],
                N_details["y_max_fit"],
                N_details["diff_fit"],
            ):
                envelope_rows.append({
                    "curve_type": "fit",
                    "metrique": metrique,
                    "condition": condition,
                    "aberration_level": aberration_level,
                    "zernike": mode_zernike,
                    "alpha": x_fit,
                    "y_mean": np.nan,
                    "y_min": y_min_fit,
                    "y_max": y_max_fit,
                    "diff": diff_fit,
                    "N": N,
                    "N_direct": N_details["N_direct"],
                    "methode": METHODE,
                    "mode": MODE,
                })

        # Calcul sensibilité et sauvegarde sensibilité
        S = abs(a_moyen) / N

        values = [
            metrique, condition, aberration_level, str(jeux_utilises), mode_zernike,
            abs(a_moyen), N, S, x_max_moyen, METHODE, MODE
        ]
        sb.append_analysis_result(OUTPUT_CSV / filename, values)

    except Exception as e:
        print(f"⚠️ Combinaison ignorée à cause d'une erreur : {e}")

# Sauvegarde des points bruts et fit
if SAVE_METRIC_POINT_AND_FIT:

    points_csv = OUTPUT_CSV / (
        f"Sensitivity_points_{METHODE}_{MODE}.csv"
    )

    fits_csv = OUTPUT_CSV / (
        f"Sensitivity_fitCurves_{METHODE}_{MODE}.csv"
    )

    envelope_csv = OUTPUT_CSV / (
        f"Sensitivity_envelope_{METHODE}_{MODE}.csv"
    )

    pd.DataFrame(points_rows).to_csv(points_csv, index=False)
    pd.DataFrame(fit_rows).to_csv(fits_csv, index=False)
    pd.DataFrame(envelope_rows).to_csv(envelope_csv, index=False)

    print(f"\nPoints Prism sauvegardés : {points_csv}")
    print(f"Fits Prism sauvegardés : {fits_csv}")
    print(f"Enveloppes Prism sauvegardées : {envelope_csv}")

print(f"\nTerminé. Résultats dans : {OUTPUT_CSV / filename}")