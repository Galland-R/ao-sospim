import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import config as cf
import src.image_io as im
import src.fourier_tools as fft
import src.dataset_tools as ds
import src.band_detection as bd
import src.io_utils as io


# -----------------------------------------
# Paramètres globaux
# -----------------------------------------

condition = cf.ANALYSIS["condition"]

# on boucle sur tous les niveaux d'aberrations induits (fait en sorte d'ignorer dossiers absents)
ABERRATION_LEVEL = ["No_aber", "Low_aber", "Medium_aber", "High_aber", "High2"]

# On test différentes acqusitions de 1 à 7 (on ignore expérience absente)
EXPERIMENTS = range(1,8)

# Boucle pour calcul des bandes
PROJECTION_MODE = ["mip", "sum", "std", "frame"]
K = [1.0, 0.8, 0.5]

# Variables
plane_index = cf.ANALYSIS["plane_index"]
alpha = cf.ANALYSIS["alpha"]

fft_profile_smoothing = cf.ANALYSIS["fft_profile_smoothing"]
fft_profile_sigma = cf.ANALYSIS["fft_profile_sigma"]
fft_profiles_normalize = cf.ANALYSIS["fft_profile_normalize"]

band_detection_sigma = cf.ANALYSIS["band_detection_sigma"]
ignore_first_n = cf.ANALYSIS["ignore_first_n"]

reference_zernike = "Zer10"   # choix reproduisant le comportement du code étudiant


# -----------------------------------------
# Petites fonctions locales
# -----------------------------------------

def folder_has_tif(folder):
    return len(ds.list_tif_images(folder)) > 0

def preprocessing_mode_for_stack(projection_mode):
    """
    Convertit le nom de métrique/dossier en mode utilisable par make_2d_image().

    Par exemple :
    - "mip" -> "max"
    - "frame" -> "plane"
    """

    mode = projection_mode.lower()

    if mode in ["mip", "max"]:
        return "max"

    if mode == "frame":
        return "plane"

    return mode

def compute_fc_values(condition, image_size_pix):
    """
    Calcule la fréquence de coupure et son rayon FFT associé.
    """

    if not cf.ANALYSIS["use_fc"]:
        return None, None, None

    base_fc_mm1 = cf.get_base_fc_mm1(
        condition=condition,
        lambda_um=cf.ANALYSIS["lambda_um"],
        fc_source=cf.ANALYSIS["fc_source"],
    )

    used_fc_mm1 = cf.ANALYSIS["fc_fraction"] * base_fc_mm1

    used_fc_radius_px = cf.freq_mm1_to_pix(
        used_fc_mm1,
        image_size_px=image_size_pix,
        pixel_size_um=cf.ANALYSIS["pixel_size_um"],
    )

    return base_fc_mm1, used_fc_mm1, used_fc_radius_px

def compute_fc_values(condition, image_size_pix):
    """
    Calcule la fréquence de coupure et son rayon FFT associé.
    """

    if not cf.ANALYSIS["use_fc"]:
        return None, None, None

    base_fc_mm1 = cf.get_base_fc_mm1(
        condition=condition,
        lambda_um=cf.ANALYSIS["lambda_um"],
        fc_source=cf.ANALYSIS["fc_source"],
    )

    used_fc_mm1 = cf.ANALYSIS["fc_fraction"] * base_fc_mm1

    used_fc_radius_px = cf.freq_mm1_to_pix(
        used_fc_mm1,
        image_size_px=image_size_pix,
        pixel_size_um=cf.ANALYSIS["pixel_size_um"],
    )

    return base_fc_mm1, used_fc_mm1, used_fc_radius_px

def make_band_row(
    row_type,
    condition,
    aberration_level,
    no_exp,
    image_folder,
    zernike_code,
    zernike_name,
    r_min,
    r_max,
    width,
    image_size_pix,
    area,
    base_fc_mm1,
    used_fc_mm1,
    used_fc_radius_px,
):
    """
    Construit une ligne de résultats homogène pour le CSV.
    """

    f_min_mm1 = cf.pix_to_freq_mm1(r_min, image_size_pix) if r_min is not None else None
    f_max_mm1 = cf.pix_to_freq_mm1(r_max, image_size_pix) if r_max is not None else None
    width_mm1 = f_max_mm1 - f_min_mm1 if r_min is not None and r_max is not None else None

    return {
        "row_type": row_type,  # "zernike", "intersection", "union"
        "condition": condition,
        "aberration_level": aberration_level,
        "no_exp": no_exp,
        "image_folder": str(image_folder),

        "projection_mode": projection_mode,
        "reference_zernike": reference_zernike,
        "alpha": alpha,

        "zernike_name": zernike_name,
        "zernike_code": zernike_code,

        "r_min_pix": r_min,
        "r_max_pix": r_max,
        "width_pix": width,

        "f_min_mm1": f_min_mm1,
        "f_max_mm1": f_max_mm1,
        "width_mm1": width_mm1,

        "area": area,

        "pixel_size_um": cf.ANALYSIS["pixel_size_um"],
        "image_size_pix": image_size_pix,

        "fft_profile_smoothing": fft_profile_smoothing,
        "fft_profile_sigma": fft_profile_sigma,
        "fft_profile_normalize": fft_profiles_normalize,

        "band_detection_sigma": band_detection_sigma,
        "k": k,
        "ignore_first_n": ignore_first_n,

        "use_fc": cf.ANALYSIS["use_fc"],
        "fc_source": cf.ANALYSIS["fc_source"],
        "fc_fraction": cf.ANALYSIS["fc_fraction"],
        "lambda_um": cf.ANALYSIS["lambda_um"],
        "NA": cf.NA_BY_CONDITION[condition],
        "base_fc_mm1": base_fc_mm1,
        "used_fc_mm1": used_fc_mm1,
        "used_fc_radius_px": used_fc_radius_px,
    }


# -----------------------------------------
# Fontion calcul pour une expérience
# -----------------------------------------

def analyze_one_experiment(condition, aberration_level, no_exp):
    """
    Analyse un dossier correspondant à:
    condition/niveau aberration/expérience
    """

    base_exp_folder = (cf.DATA_ROOT_DIR / condition / aberration_level / str(no_exp))
    
    if folder_has_tif(base_exp_folder) == 0:
        print(f"Dossier ignoré, pas de tif: {base_exp_folder}")
        return []
    
    print("\n" + "=" * 70)
    print(f"Analyse: {condition} | projection mode: {projection_mode} | k = {k} | {aberration_level} | experiment no {no_exp}")
    print(f"dossier image: {base_exp_folder}")
    print("=" * 70)

    image_paths = ds.list_tif_images(base_exp_folder)

    if len(image_paths) == 0:
        print("Aucune imlage Tif trouvées")
        return []
    
    # Obtention image de référence (reference_zernike = "Zer10" comme dans code étudiants)
    images_mode_ref = ds.recuperer_images_par_mode(image_paths, reference_zernike)
    try:
        image_ref_path = ds.trouver_reference(images_mode_ref, reference_zernike)
    except FileNotFoundError as e:
        print(f"Exprience ignorée: {e}")
        return []
    
    ref_stack = im.load_tif(image_ref_path)
    ref_image = im.make_2d_image(ref_stack, mode=projection_mode, plane_index=plane_index)

    # Calcul taille image pour calcul fc_pix
    image_size_pix = min(ref_image.shape)
    base_fc_mm1, fc_mm1, fc_radius_px = compute_fc_values(condition, image_size_pix)
    
    print(f"Image référence: {image_ref_path.name}")
    if fc_mm1 is not None:
        print(f"fc utilisée: {fc_mm1:1f} mm-1")
        print(f"rayon fc: {fc_radius_px:1f} pix")
    
    # Calcul moyenne rotationelle image de référence
    bins_ref, profile_ref, counts_ref = fft.calc_rotational_average(
        ref_image,
        image_size_pix // 2,
        apply_smoothing=fft_profile_smoothing,
        sigma=fft_profile_sigma
    )

    results = []
    band_limits = []

    # boucle sur mode de Zernike
    for mode_name, mode_zernike in ds.MODES_ZERNIKE.items():

        print(f"Mode: {mode_name} ({mode_zernike})")

        image_mode = ds.recuperer_images_par_mode(image_paths, mode_zernike)

        try:
            image_aber_path = ds.trouver_image_aberration(image_mode, mode_zernike, alpha=alpha)
        except FileNotFoundError as e:
            print(f"ignoré: {e}")
            continue

        # Obtention image mode
        aber_stack = im.load_tif(image_aber_path)
        aber_image = im.make_2d_image(aber_stack, mode=projection_mode, plane_index=plane_index)

        bins_aber, profile_aber, counts_aber = fft.calc_rotational_average(
            aber_image,
            image_size_pix // 2,
            apply_smoothing=fft_profile_smoothing,
            sigma=fft_profile_sigma
        )

        # Compare profile aberré vs référence
        bins, ref_norm, aber_norm, diff, diff_smooth = bd.compare_rotational_profiles(
            bins_ref, profile_ref,
            bins_aber, profile_aber,
            smoothing_sigma=band_detection_sigma,
            normalize=fft_profiles_normalize
        )

        # Calcul bandes
        r_min, r_max, width = bd.detect_adaptive_band(
            bins,
            diff_smooth,
            k=k,
            ignore_first_n=ignore_first_n,
            max_radius=fc_radius_px
        )

        area = bd.compute_integral_in_band(r_min, r_max, bins, diff_smooth)

        if r_min is not None:
            band_limits.append((r_min, r_max))
        
        results.append(
            make_band_row(
                row_type="zernike",
                condition=condition,
                aberration_level=aberration_level,
                no_exp=no_exp,
                image_folder=base_exp_folder,
                zernike_code=mode_zernike,
                zernike_name=mode_name,
                r_min=r_min,
                r_max=r_max,
                width=width,
                image_size_pix=image_size_pix,
                area=area,
                base_fc_mm1=base_fc_mm1,
                used_fc_mm1=fc_mm1,
                used_fc_radius_px=fc_radius_px,
            )
        )

    # -----------------------------------------
    # Intersection et Union
    # -----------------------------------------
    
    if band_limits:
        common_min = max(r[0] for r in band_limits)
        common_max = min(r[1] for r in band_limits)

        union_r_min = min(r[0] for r in band_limits)
        union_r_max = max(r[1] for r in band_limits)
        union_width = union_r_max - union_r_min

        if common_min < common_max:
            intersection_common_min = common_min
            intersection_common_max = common_max
            intersection_width = common_max - common_min
        else :
            intersection_common_max = None
            intersection_common_min = None
            intersection_width = None
        
        # concaténation résultats 
        results.append(
                make_band_row(
                    row_type="intersection",
                    condition=condition,
                    aberration_level=aberration_level,
                    no_exp=no_exp,
                    image_folder=base_exp_folder,
                    zernike_code="Intersection",
                    zernike_name="Intersection",
                    r_min=intersection_common_min,
                    r_max=intersection_common_max,
                    width=intersection_width,
                    image_size_pix=image_size_pix,
                    area=None,
                    base_fc_mm1=base_fc_mm1,
                    used_fc_mm1=fc_mm1,
                    used_fc_radius_px=fc_radius_px,
                )
            )
        results.append(
                make_band_row(
                    row_type="union",
                    condition=condition,
                    aberration_level=aberration_level,
                    no_exp=no_exp,
                    image_folder=base_exp_folder,
                    zernike_code="Union",
                    zernike_name="Union",
                    r_min=union_r_min,
                    r_max=union_r_max,
                    width=union_width,
                    image_size_pix=image_size_pix,
                    area=None,
                    base_fc_mm1=base_fc_mm1,
                    used_fc_mm1=fc_mm1,
                    used_fc_radius_px=fc_radius_px,
                )
            )

    else:
        print("  Aucune bande détectée pour cette expérience.")

    return results


# -----------------------------------------
# Boucle global condition
# -----------------------------------------

for projection_mode in PROJECTION_MODE:
    for k in K:

        all_results = []

        for aberration_level in ABERRATION_LEVEL:
            for no_exp in EXPERIMENTS:

                results_exp = analyze_one_experiment(
                    condition=condition,
                    aberration_level=aberration_level,
                    no_exp=no_exp
                )

                all_results.extend(results_exp)


        # -----------------------------------------
        # Sauvegarde globale
        # -----------------------------------------

        if not all_results:
            print("Aucun résultats à sauvegarder")
        else:
            filename = (
                f"Band_{condition}_All-Aber&exp"
                f"_{projection_mode}"
                f"_Ref-{reference_zernike}"
                f"_fc-{cf.ANALYSIS['fc_source']}"
                f"_k-{k}"
            )

            if fft_profiles_normalize:
                filename += "_fft-norm"
            
            if fft_profile_smoothing:
                filename += f"_fft-smooth-{fft_profile_sigma}"
            
            filename += f"_band-smooth-{band_detection_sigma}.csv"

            output_csv = cf.CSV_DIR / filename

            io.save_band_detection_csv(all_results, output_csv)


