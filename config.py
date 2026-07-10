from pathlib import Path


# =========================
# Paramètres 
# =========================

NA_BY_CONDITION = {
    "Coverslip": 1.49,
    "Profondeur": 1.27,
}

EXPERIMENTAL_FC_BY_CONDITION = {    # En mm-1
    "Coverslip": 3846,
    "Profondeur": 3125,
}

ANALYSIS = {
    "analysis_version": "band_detection_v0.1",

    "condition": "Coverslip",          # "Profondeur" ou "Coverslip"
    "aberration_level": "No_aber",      # Niveau d'aberration ("High_aber", "High2", Low_aber, "Medium_aber", No_aber)
    "no_exp": 1,                        # Numero réptition expérience (1 --> 5 ou 7)

    "pixel_size_um": 0.108,
    "lambda_um": 0.589,

    "projection_mode": "frame",         # "mip", "sum", "std", "mean", "frame"
    "mode_zernike": "Zer4",             # Mode Zernike a analyser "Zer4", "Zer5", ..., "Zer10"
    "alpha": "+4alpha",
    "plane_index": 0,                   # Frame retenue pour analyse "frame"

    "fft_profile_smoothing": True,      # Applique smoothing au calcul moyenne rotationnelle (FFT)
    "fft_profile_sigma": 2,             # Facteur smoothing Gaussien pour calcul moyenne rotationnelle (FFT)
    "fft_profile_normalize": True,      # Normalisation des FFT (FFT = FFT / max(FFT))

    "band_detection_sigma": 3,          # Facteur Smoothing Gaussien pour différence (FFT_ref - FFT_aberr)
    "k": 1,                           # Seuil pour détection bandes: moyenne(diff_smooth) + k * écart-type(diff_smooth)
    "ignore_first_n": 0,                # Ignore n premier poit pour détermination bande

    "use_fc": True,                     # fc utilisée pour limitée la recherch de bande ?
    "fc_source": "experimental",        # "theoretical" -> 2*NA/lambda; "experimental" -> valeur mesurée, par exemple 3125 (profondeur) / 3846 (coverslip)
    "fc_fraction": 1.0,                 # 1.0: Toute la fréquence; 0.5: fc/2 (métrique REALM)

    "fft_profile_save": True,           # Sauvegarde profile fft ref & Aber + diff en csv
    "fft_profile_plot_save": True,      # Sauvergae les figures de comparaison en png
    "fft_profile_plot_show": True,      # Affiche figure de comparaison
}


BANDPASS_RADII_PX = {
    "Coverslip": {
        "MIP": (22, 67), "SUM": (20, 61), "Frame": (16, 52), "Dapi": (11, 42),
        "Gallery": (16, 52), "STD": (23, 71), "MIP25": (21, 64), "avg_psf": (31, 73),
    },
    "Profondeur": {
        "MIP": (16, 66), "SUM": (13, 57), "Frame": (7, 51), "Dapi": (6, 43),
        "Gallery": (11, 61), "STD": (16, 70),
        "mip_10frames": (16, 66), "mip_30frames": (16, 66),
        "mip_50frames": (16, 66), "mip_100frames": (16, 66),
        "std_10frames": (16, 70), "std_30frames": (16, 70),
        "std_50frames": (16, 70), "std_100frames": (16, 70),
    },
}



# =========================
# Chemins principaux
# =========================

# Chemin du dossier ao-sospim (VS Code)
PROJECT_DIR = Path(__file__).resolve().parent       

# Chemin ou se trouve les images à analyser
DATA_ROOT_DIR = Path(
    "/Volumes/TEAM_M/everyone/Laetitia/AO/BANDPASS/CARAC/CARAC_AO_THESE"
    #"/Users/remi_galland/Library/CloudStorage/Dropbox-Sibarita_QIC/Rémi Galland/Work_RG/Publications/2026 - AO-soSPIM/Data/Figure01/Data"
)

DATA_DIR = DATA_ROOT_DIR / ANALYSIS["condition"] / ANALYSIS["aberration_level"] / str(ANALYSIS["no_exp"])    

# Chemin pour sauvegarde des résultats avec arborescence (dossier "csv", "fft_profile")
RESULTS_DIR = Path(                                 
    "/Users/remi_galland/Library/CloudStorage/Dropbox-Sibarita_QIC/Rémi Galland/Work_RG/Publications/2026 - AO-soSPIM/Data/Figure01/Results/"
)
CSV_DIR = RESULTS_DIR / "csv"
FFT_DIR = RESULTS_DIR / "fft_profiles"
IMAGE_DIR = RESULTS_DIR / "images"
for folder in [RESULTS_DIR, CSV_DIR, FFT_DIR, IMAGE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# Conversion pixels <-> fréquence
# =========================

def pix_to_freq_mm1(radius_px, image_size_px=256, pixel_size_um=ANALYSIS["pixel_size_um"]):
    """
    Convertit un rayon en pixels dans la FFT en fréquence spatiale mm^-1.

    Formule :
        fréquence = radius_px / (image_size_px * pixel_size_mm)
    """

    pixel_size_mm = pixel_size_um * 1e-3

    return radius_px / (image_size_px * pixel_size_mm)


def freq_mm1_to_pix(freq_mm1, image_size_px=256, pixel_size_um=ANALYSIS["pixel_size_um"]):
    """
    Convertit une fréquence spatiale mm^-1 en rayon FFT en pixels.
    """

    pixel_size_mm = pixel_size_um * 1e-3
    return freq_mm1 * image_size_px * pixel_size_mm


# =========================
# Fréquence de coupures
# =========================

def theoretical_fc_mm1(na, lambda_um=0.589):
    """
    Calcule la fréquence de coupure théorique en mm^-1.

    Par défaut :
        fc = 2 * NA / lambda

    avec lambda en µm.
    Le facteur 1000 convertit µm^-1 en mm^-1.
    """

    return 2 * na / lambda_um * 1000

def experimental_fc_mm1 (condition):
    """
    Retourne la fréquence de coupure expérimentale mesurée en mm^-1.
    """

    return EXPERIMENTAL_FC_BY_CONDITION[condition]


def get_base_fc_mm1(condition, lambda_um, fc_source):
    """
    Retourne la fréquence de coupure complète avant application d'une fraction.

    cutoff_source :
    - "theoretical"  -> 2*NA/lambda par défaut
    - "experimental" -> valeur mesurée
    """

    if fc_source == "theoretical":
        na = NA_BY_CONDITION[condition]
        return theoretical_fc_mm1(
            na=na,
            lambda_um=lambda_um
        )

    if fc_source == "experimental":
        return experimental_fc_mm1(condition)

    raise ValueError(f"fc_source inconnu : {fc_source}")


def get_analysis_fc_mm1(fc_fraction=None):
    """
    Retourne la fréquence de coupure effectivement utilisée pour l'analyse.

    Exemple :
    - cutoff_fraction = 1.0 -> coupure complète
    - cutoff_fraction = 0.5 -> moitié de la coupure, cas REALM
    """

    if fc_fraction is None:
        fc_fraction = ANALYSIS["fc_fraction"]

    base_fc = get_base_fc_mm1(
        condition=ANALYSIS["condition"],
        lambda_um=ANALYSIS["lambda_um"],
        fc_source=ANALYSIS["fc_source"],
    )

    return fc_fraction * base_fc

def get_analysis_fc_radius_px(image_size_px=256, fc_fraction=None):
    """
    Convertit la fréquence de coupure utilisée en rayon FFT, en pixels.
    """

    fc_mm1 = get_analysis_fc_mm1(fc_fraction=fc_fraction)

    return freq_mm1_to_pix(fc_mm1, image_size_px=image_size_px, pixel_size_um=ANALYSIS["pixel_size_um"])

def obtenir_parametres_metrique(profondeur, metrique):
    """
    Détermine NA, Lambda, r_min_px, r_max_px selon la profondeur et le type d'image.
    """
    NA = NA_BY_CONDITION[profondeur]
    Lambda = 0.465 if metrique == "Dapi" else 0.589

    r_min_px, r_max_px = BANDPASS_RADII_PX[profondeur][metrique]

    return NA, Lambda, r_min_px, r_max_px