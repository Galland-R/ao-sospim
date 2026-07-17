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
        "Low_aber",
        "Medium_aber",
        "High_aber",
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
    "STD",
    "SUM",
    #"Gallery",
    "Frame",
    #"Dapi",
]

JEUX = (1, 2, 3, 4, 5, 6, 7)  # la gestion des dossiers manquants s'occupe du reste

# Réglages fixes
PLOT_A = False   # désactivé pour une boucle longue (sinon ça ouvre une fenêtre par itération !)
PLOT_N = False
SAVE_METRIC_POINT_AND_FIT = True    # Sauveagrde des points calculés pour chaque métrique et conditins ainsi que le fit associés
METHODE = "std"     #"std" (moyenne ± écart-type) ou "minmax" (vrai min/max observé)
MODE = "fit"     #"direct" (aire sur les points bruts) ou "fit" (aire après fit quadratique)
ALPHAMAX = 0.26
FIT_SKIP_FIRST_N = 1  # 0 = on utilise tous les points; 1 = enléve le premier points ("-4Alpha"); ...
FIT_SKIP_LAST_N = 0

# -----------------------------------------
# Petites fonctions locales
# -----------------------------------------

def build_sensitivity_base_name(output_csv, method, mode, fit_skip_first_n=0, fit_skip_last_n=0):
    """
    Construit un nom de bse pour la sauvegarde des résultats
    """
    base_name = (
        f"Sensitivity_{method}"
        f"_{mode}"
    )

    if fit_skip_first_n > 0:
        base_name += f"_fitSkipFirts-{fit_skip_first_n}"
    if fit_skip_last_n > 0:
        base_name += f"_fitSkipLast-{fit_skip_last_n}"
    
    base_name += ".csv"
    
    return base_name

def make_group_key(condition, aberration_level, metrique, methode, mode):
    """
    Définit un groupe de sortie.

    Un groupe = un fichier Prism.
    """
    return (condition, aberration_level, metrique, methode, mode)


def make_group_filename(prefix, key, fit_skip_first_n=0, fit_skip_last_n=0):
    """
    Construit un nom de fichier à partir d'un groupe.
    """
    condition, aberration_level, metrique, methode, mode = key

    filename = (
        f"{prefix}_"
        f"{condition}_"
        f"{aberration_level}_"
        f"{metrique}_"
        f"{methode}_"
        f"{mode}"
    )

    if fit_skip_first_n > 0:
        filename += f"_fitSkipFirts-{fit_skip_first_n}"
    if fit_skip_last_n > 0:
        filename += f"_fitSkipLast-{fit_skip_last_n}"
    
    filename += ".csv"

    return filename


def get_or_create_alpha_table(tables, key, x_values, x_column="alpha"):
    """
    Récupère un tableau existant ou crée un tableau avec la colonne alpha.
    """
    if key not in tables:
        tables[key] = {x_column: x_values}

    return tables[key]


def get_or_create_fit_table(tables, key, x_fit):
    """
    Récupère un tableau de fit existant ou crée un tableau avec x_fit.
    """
    if key not in tables:
        tables[key] = {"x_fit": x_fit}

    return tables[key]

# -----------------------------------------
# Boucle principale
# -----------------------------------------
filename = build_sensitivity_base_name(OUTPUT_CSV, METHODE, MODE, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N)
sb.create_analysis_csv(OUTPUT_CSV / filename)

combinaisons = [
    (condition, aberration_level, mode_zernike, metrique)
    for condition in CONDITIONS
    for aberration_level in ABERRATION_LEVEL_BY_PROFONDEUR[condition]
    for mode_zernike in MODES_ZERNIKE
    for metrique in METRIQUES
]

points_tables = {}          # points bruts et normalisés, un fichier par condition
fit_tables = {}             # fits quadratiques par expérience
envelope_tables = {}        # moyenne, ymin, ymax
envelope_fit_tables = {}    # fits des enveloppes
coeff_rows = []             # coefficients a, b, c, r², x_max par expérience

for i, (condition, aberration_level, mode_zernike, metrique) in enumerate(combinaisons, start=1):
    print(f"\n[{i}/{len(combinaisons)}] {condition} | {aberration_level} | {mode_zernike} | {metrique}")

    try:
        x, matrice, jeux_utilises = sb.obtenir_matrice_metrique(
            BASE_DIR, condition, aberration_level, mode_zernike, metrique, JEUX
        )
        x = np.array(x)
        x = x * ALPHAMAX / 4

        print("x utilisés :", x)
        print("x pour fit :", x[FIT_SKIP_FIRST_N:len(x)-FIT_SKIP_LAST_N if FIT_SKIP_LAST_N > 0 else len(x)])

        matrice_norm = sb.normaliser_par_ligne(matrice)

        group_key = make_group_key(condition, aberration_level, metrique, METHODE, MODE)

        # -------------------------------------------------
        # 1) Création table points expérimentaux
        # -------------------------------------------------

        if SAVE_METRIC_POINT_AND_FIT:
            points_df = get_or_create_alpha_table(points_tables, group_key, x, x_column="alpha")
        
        # -------------------------------------------------
        # 2) Fit quadratique par expérience
        # -------------------------------------------------
        
        liste_a = []
        liste_x_max = []

        for jeu, y_raw, y_norm in zip(jeux_utilises, matrice, matrice_norm):

            label = f"{mode_zernike}_exp-{jeu}"
            
            # Fit quadratique
            fit_details = sb.coefficient_a_details(y_norm, x, plot=PLOT_A, fit_skip_first_n=FIT_SKIP_FIRST_N, fit_skip_last_n=FIT_SKIP_LAST_N)

            # Points bruts et normalisés
            if SAVE_METRIC_POINT_AND_FIT:
                points_df[f"raw_{label}"] = y_raw
                points_df[f"norm_{label}"] = y_norm
                points_df[f"fitUsed_{label}"] = fit_details["fit_used_mask"].astype(int)

            liste_a.append(fit_details["a"])
            liste_x_max.append(fit_details["x_max"])

            if SAVE_METRIC_POINT_AND_FIT:
                fit_df = get_or_create_fit_table(fit_tables, group_key, fit_details["x_fit"])
            
                fit_df[f"fit_{label}"] = fit_details["y_fit"]

                coeff_rows.append({
                    "metrique": metrique,
                    "conditions": condition,
                    "aberration_level": aberration_level,
                    "zernike": mode_zernike,
                    "jeu": jeu,
                    "a": fit_details["a"],
                    "b": fit_details["b"],
                    "c": fit_details["c"],
                    "r2": fit_details["r2"],
                    "x_max": fit_details["x_max"],
                    "methode": METHODE,
                    "mode": MODE,
                })

        a_moyen = float(np.mean(liste_a))
        x_max_moyen = float(np.mean(liste_x_max))

        # -------------------------------------------------
        # 3) Enveloppes pour le calcul de N
        # -------------------------------------------------

        courbes_norm = list(zip(*matrice_norm))

        N_details = sb.coefficient_N_details(
            courbes_norm, x, plot=PLOT_N,
            methode=METHODE, mode=MODE,
            fit_skip_first_n=FIT_SKIP_FIRST_N, fit_skip_last_n=FIT_SKIP_LAST_N
        )
        N = N_details["N"]

        if SAVE_METRIC_POINT_AND_FIT:
            envelope_df = get_or_create_alpha_table(envelope_tables, group_key, N_details["x_points"], x_column="alpha")

            envelope_df[f"y_mean_{mode_zernike}"] = N_details["y_mean"]
            envelope_df[f"y_min_{mode_zernike}"] = N_details["y_min"]
            envelope_df[f"y_max_{mode_zernike}"] = N_details["y_max"]
            envelope_df[f"diff_{mode_zernike}"] = N_details["diff_direct"]
            envelope_df[f"fitUsed_{mode_zernike}"] = N_details["fit_used_mask"].astype(int)
        
            if SAVE_METRIC_POINT_AND_FIT and MODE == "fit":
                # fit enveloppe dans un tableau séparé car x plus fin
                if group_key not in envelope_fit_tables:
                    envelope_fit_tables[group_key] = {"x_fit": N_details["x_fit"]}
                
                envelope_fit_df = envelope_fit_tables[group_key]

                envelope_fit_df[f"y_min_fit_{mode_zernike}"] = N_details["y_min_fit"]
                envelope_fit_df[f"y_max_fit_{mode_zernike}"] = N_details["y_max_fit"]
                envelope_fit_df[f"diff_fit_{mode_zernike}"] = N_details["diff_fit"]

        # -------------------------------------------------
        # 4) Calcul sensibilité et sauvegarde sensibilité
        # -------------------------------------------------
        S = abs(a_moyen) / N

        values = [
            metrique, condition, aberration_level, str(jeux_utilises), mode_zernike,
            abs(a_moyen), N, S, x_max_moyen, METHODE, MODE, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N,
        ]
        sb.append_analysis_result(OUTPUT_CSV / filename, values)

    except Exception as e:
        print(f"⚠️ Combinaison ignorée à cause d'une erreur : {e}")

# Sauvegarde des points bruts et fit
if SAVE_METRIC_POINT_AND_FIT:
    print("\n")
    # Points expérimentaux : raw et normalisés
    for key, df in points_tables.items():
        name = make_group_filename("Sensitivity_points", key, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N)
        out = OUTPUT_CSV / name
        pd.DataFrame(df).to_csv(out, index=False)
        print(f"Points Prism sauvegardés : {name}")

    # Fits quadratiques par expérience
    for key, df in fit_tables.items():
        name = make_group_filename("Sensitivity_fits_by_exp", key, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N)
        out = OUTPUT_CSV / name
        pd.DataFrame(df).to_csv(out, index=False)
        print(f"Fits Prism sauvegardés : {name}")

    # Enveloppes aux points expérimentaux
    for key, df in envelope_tables.items():
        name = make_group_filename("Sensitivity_envelope_points", key, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N)
        out = OUTPUT_CSV / name
        pd.DataFrame(df).to_csv(out, index=False)
        print(f"Enveloppes Prism sauvegardées : {name}")

    # Fits des enveloppes
    for key, df in envelope_fit_tables.items():
        name = make_group_filename("Sensitivity_envelope_fits", key, FIT_SKIP_FIRST_N, FIT_SKIP_LAST_N)
        out = OUTPUT_CSV / name
        pd.DataFrame(df).to_csv(out, index=False)
        print(f"Fits enveloppes Prism sauvegardés : {name}")

    # Coefficients de fit par expérience
    if coeff_rows:
        coeffs_csv = OUTPUT_CSV / (f"Sensitivity_fit_coefficients_{METHODE}_{MODE}.csv")
        pd.DataFrame(coeff_rows).to_csv(coeffs_csv, index=False)
        print(f"Coefficients de fit sauvegardés : Sensitivity_fit_coefficients_{METHODE}_{MODE}.csv")

print(f"\nTerminé. Résultats dans : {OUTPUT_CSV / filename}")