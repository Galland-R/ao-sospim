import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import pandas  as pd
import matplotlib.pyplot as plt

import src.image_io as im
import src.fourier_tools as fft
import src.dataset_tools as ds
import src.band_detection as bd
import src.io_utils as io
import config as cf

#---------------------------
# Paramètre utilisateur
#---------------------------

projection_mode = cf.ANALYSIS["projection_mode"]
plane_index = cf.ANALYSIS["plane_index"]
alpha = cf.ANALYSIS["alpha"]

fft_profile_smoothing = cf.ANALYSIS["fft_profile_smoothing"]
fft_profile_sigma = cf.ANALYSIS["fft_profile_sigma"]
fft_profiles_normalize = cf.ANALYSIS["fft_profile_normalize"]

band_detection_sigma = cf.ANALYSIS["band_detection_sigma"]
k = cf.ANALYSIS["k"]
ignore_first_n = cf.ANALYSIS["ignore_first_n"]

#---------------------------
# Analyse
#---------------------------

images_path = ds.list_tif_images(cf.DATA_DIR)    # Listing images Tif dans dossier

results = []
band_limits = []

for mode_name, mode_zernike in ds.MODES_ZERNIKE.items():        # Boucle sur ensemble mode Zernike Zer4 --> Zer10 indiqué dans bibliothèque de mode (dataset_tools.py)
    
    print(f"\nAnalyse: {mode_name} ({mode_zernike})")
    images_mode = ds.recuperer_images_par_mode(images_path, mode_zernike)    # Selectionne image correspondant au mode de Zerike a analyser
    images_mode_ref = ds.recuperer_images_par_mode(images_path, "Zer10")

    try:
        # Utilise "refZer10" comme image de référence pour calcul de différence bande
        image_ref_path = ds.trouver_reference(images_mode_ref, "Zer10")
        image_aber_path = ds.trouver_image_aberration(images_mode, mode_zernike, alpha=alpha)
    except FileNotFoundError as e:
        print(f"Mode ignoré: {e}")
        continue

    # Ouverture stacks ref et aberr
    ref_stack = im.load_tif(image_ref_path)
    aberr_stack = im.load_tif(image_aber_path)

    # Pre-processing stack ("sum", "max", ...)
    ref_image = im.make_2d_image(ref_stack, mode=projection_mode, plane_index=plane_index)
    aber_image = im.make_2d_image(aberr_stack, mode=projection_mode, plane_index=plane_index)
    image_size_pix = min(ref_image.shape)

    # Calcul moyennes rotationnelles
    bins_ref, profile_ref, counts_ref = fft.calc_rotational_average(
        ref_image,
        image_size_pix // 2,
        apply_smoothing=fft_profile_smoothing,
        sigma=fft_profile_sigma,
    )
    bins_aberr, profile_aber, counts_aberr = fft.calc_rotational_average(
        aber_image,
        image_size_pix // 2,
        apply_smoothing=fft_profile_smoothing,
        sigma=fft_profile_sigma,
    )

    # Comparaison des moyennes rotationnelles ref vs aberr
    bins, ref_norm, aberr_norm, diff, diff_smooth = bd.compare_rotational_profiles(
        bins_ref,
        profile_ref,
        bins_aberr,
        profile_aber,
        smoothing_sigma=band_detection_sigma,
        normalize=False
    )

    # Calcul frequence coupure (fc) théorique ou experimentale
    if cf.ANALYSIS["use_fc"]:
        fc_mm1 = cf.get_analysis_fc_mm1()
        fc_radius_pix = cf.get_analysis_fc_radius_px(image_size_pix)
    
        base_fc_mm1 = cf.get_base_fc_mm1(
            condition=cf.ANALYSIS["condition"],
            lambda_um=cf.ANALYSIS["lambda_um"],
            fc_source=cf.ANALYSIS["fc_source"],
        )

        print(f"Coupure utilisée : {fc_mm1:.1f} mm^-1")
        print(f"Rayon FFT max : {fc_radius_pix:.1f} px")
    
    else:
        fc_mm1 = None
        fc_radius_pix = None
        base_fc_mm1 = None

    # Détection bande spectrale selon seuil  = moyenne(diff_smooth) + k * écart-type(diff_smooth)
    r_min, r_max, width = bd.detect_adaptive_band(
        bins,
        diff_smooth,
        k=k,
        ignore_first_n=ignore_first_n,
        max_radius=fc_radius_pix
    )

    # Calcul intégrale de diff_smooth dans bande (peut etre comparé à une sensibilité dans la bande)
    area = bd.compute_integral_in_band(r_min, r_max, bins, diff_smooth)

    f_min_mm1 = cf.pix_to_freq_mm1(r_min, image_size_pix) if r_min is not None else None
    f_max_mm1 = cf.pix_to_freq_mm1(r_max, image_size_pix) if r_max is not None else None
    width_mm1 = f_max_mm1 - f_min_mm1 if r_min is not None else None

    # Concaténation des limites de bandes calculées
    if r_min is not None:
        band_limits.append((r_min, r_max))


    # Concaténation des résultats et conditions analyses
    results.append({
        "zernike_code": mode_zernike,
        "r_min_pix": r_min,
        "r_max_pix": r_max,
        "width_pix": width,
        "f_min_mm1": f_min_mm1,
        "f_max_mm1": f_max_mm1,
        "width_mm1": width_mm1,
        "area": area
    })

    # Figure FFT ref vs aberr
    plt.figure()
    plt.plot(bins, ref_norm, label="Référence")
    plt.plot(bins, aberr_norm, label=f"{alpha}{mode_zernike}")
    plt.xlabel("Rayon FFT (px)")
    plt.ylabel("Amplitude normalisée")
    plt.title(f"{mode_zernike} - {projection_mode}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # Figure diff FFT et détection bande
    plt.figure()
    plt.plot(bins, diff_smooth, label="|Δ FFT| lissé")
    if r_min is not None:
        plt.axvspan(r_min, r_max, alpha=0.3, label="Bande détectée")
    plt.xlabel("Rayon FFT (px)")
    plt.ylabel("|Δ FFT|")
    plt.title(f"Bande sensible - {mode_zernike}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

#-----------------------------------
# Calcul bande interscetion / Union
#-----------------------------------

if band_limits:
     # Intersection des bandes
    common_min = max(r[0] for r in band_limits)
    common_max = min(r[1] for r in band_limits)

    common_min_mm1 = cf.pix_to_freq_mm1(common_min, image_size_pix)
    common_max_mm1 = cf.pix_to_freq_mm1(common_max, image_size_pix)

    # Union des bandes
    global_min = min(r[0] for r in band_limits)
    global_max = max(r[1] for r in band_limits)

    global_min_mm1 = cf.pix_to_freq_mm1(global_min, image_size_pix)
    global_max_mm1 = cf.pix_to_freq_mm1(global_max, image_size_pix)

    print("\nRésumé des bandes")
    if common_min < common_max:
        print(f"Intersection des bandes: {common_min} - {common_max} pix ; {common_min_mm1} - {common_max_mm1} mm1")
    else:
        print("Aucune Intersection entre les bandes")

    print(f"Union bandes: {global_min} - {global_max} pix; {global_min_mm1} - {global_max_mm1} mm1")

else:
    common_min = common_max = common_min_mm1 = common_max_mm1 = None
    global_min = global_max = global_min_mm1 = global_max_mm1 = None
    print("Auncune bande détectée")


results.append({
        "zernike_code": "Intersection",
        "r_min_pix": common_min,
        "r_max_pix": common_max,
        "width_pix": (common_max - common_min),
        "f_min_mm1": common_min_mm1,
        "f_max_mm1": common_max_mm1,
        "width_mm1": (common_max_mm1 - common_min_mm1),
        "area": area
    })

results.append({
        "zernike_code": "Union",
        "r_min_pix": global_min,
        "r_max_pix": global_max,
        "width_pix": (global_max - global_min),
        "f_min_mm1": global_min_mm1,
        "f_max_mm1": global_max_mm1,
        "width_mm1": (global_max_mm1 - global_min_mm1),
        "area": area,

        "condition": cf.ANALYSIS["condition"],
        "Projection_mode": cf.ANALYSIS["projection_mode"],
        "NA": cf.NA_BY_CONDITION[cf.ANALYSIS["condition"]],
        "alpha": cf.ANALYSIS["alpha"],
        "fc_source": cf.ANALYSIS["fc_source"],
        "pixel_size_um": cf.ANALYSIS["pixel_size_um"],
        "k": k,
        "ignore_first_n": ignore_first_n,
        "band_detection_sigma": band_detection_sigma,
        "lambda_um": cf.ANALYSIS["lambda_um"],
        "base_fc_mm1": base_fc_mm1,
        "fc_fraction": cf.ANALYSIS["fc_fraction"],
        "used_fc_mm1": fc_mm1,
        "used_fc_radius_px": fc_radius_pix,
    })

#---------------------------
# Sauvegarde résultats bandes
#---------------------------

filename = (
    f"{cf.ANALYSIS['condition']}_{cf.ANALYSIS['aberration_level']}_exp-{cf.ANALYSIS['no_exp']}_"
    f"Band_{projection_mode}"
    f"_Ref-vs-{alpha}"
    f"_fc-{cf.ANALYSIS['fc_source']}"
    f"_k-{k}"
)
if fft_profiles_normalize:
    filename += "_norm"

if fft_profile_smoothing:
    filename += f"_fftSmooth-{fft_profile_sigma}"
        
filename += f"_bandsmooth-{band_detection_sigma}.csv"

output_csv = cf.CSV_DIR / filename

io.save_band_detection_csv(results, output_csv)















