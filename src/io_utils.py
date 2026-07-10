
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def write_analysis_header(file, analysis_params, script_name, metadata=None):
    """
    Écrit un en-tête générique dans un fichier CSV.

    metadata permet d'ajouter des informations spécifiques :
    - image source
    - dossier analysé
    - image référence
    - alpha utilisé
    - condition
    - expérience
    etc.

    outes les lignes commencent par '#'.
    Elles sont donc ignorées par pandas avec :

        pd.read_csv(..., comment="#")
    """

    metadata = metadata or {}

    file.write("# ======================================================\n")
    file.write("# AO-soSPIM analysis\n")
    file.write("# ======================================================\n")
    file.write(f"# Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    file.write(f"# Script : {script_name}\n")
    file.write("#\n")

    file.write("# --- Analysis parameters ---\n")
    for key, value in analysis_params.items():
        file.write(f"# {key} : {value}\n")

    if metadata:
        file.write("#\n")
        file.write("# --- Metadata ---\n")
        for key, value in metadata.items():
            file.write(f"# {key} : {value}\n")

    file.write("# ======================================================\n\n")



def save_rotational_average_csv(bins, radial_mean, pixel_counts, output_path, image_size_px, analysis_params, mode):
    """
    Sauvegarde la moyenne rotationnelle dans un fichier CSV.

    Colonnes :
    - radius_px : rayon en pixels dans l'espace de Fourier
    - radial_mean : amplitude moyenne pour ce rayon
    - pixel_counts : nombre de pixels utilisés dans chaque couronne
    """

    output_path = Path(output_path)                             
    output_path.parent.mkdir(parents=True, exist_ok=True)   #Créer le dossier outpu_path si n'exsite pas

    df = pd.DataFrame({
        "radius_px": bins,
        "frequency_mm1": bins / (image_size_px * analysis_params["pixel_size_um"] * 1e-3),
        f"radial_mean_{mode}": radial_mean,
        f"pixel_counts_{mode}": pixel_counts
    })


    # Sans header mais avec paramètre dans nom
    df.to_csv(output_path, index=False)

    print(f"Moyenne rotationnelle sauvegardée : {output_path}")

from pathlib import Path
import pandas as pd


def save_band_detection_csv(results, output_path):
    """
    Sauvegarde les résultats de détection de bandes dans un CSV.
    Pas de header commenté : uniquement un tableau.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Résultats bandes sauvegardés : {output_path}")


from pathlib import Path
import numpy as np
import pandas as pd


def save_fft_profile_comparison_csv(
    bins,
    ref_profile,
    aberr_profile,
    diff_smooth,
    diff_norm,
    output_path,
    image_size_pix,
    pixel_size_um,
    r_min=None,
    r_max=None,
):
    """
    Sauvegarde les profils FFT comparés dans un CSV compatible Prism.

    Colonnes :
    - radius_px
    - frequency_mm1
    - reference_profile
    - aberrated_profile
    - difference
    - difference_smooth
    - band_mask : 1 dans la bande détectée, 0 ailleurs
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frequency_mm1 = bins / (image_size_pix * pixel_size_um * 1e-3)

    band_mask = np.zeros_like(bins, dtype=int)

    if r_min is not None and r_max is not None:
        band_mask[(bins >= r_min) & (bins <= r_max)] = 1

    df = pd.DataFrame({
        "radius_px": bins,
        "frequency_mm1": frequency_mm1,
        "reference_profile": ref_profile,
        "aberrated_profile": aberr_profile,
        "difference_smooth": diff_smooth,
        "difference_norm": diff_norm,
        "band_mask": band_mask,
    })

    df.to_csv(output_path, index=False)

    # print(f"Profil FFT sauvegardé : {output_path}")