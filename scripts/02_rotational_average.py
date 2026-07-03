import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent           #__file__ = chemin complet du fichier scripts/0A_load_image.py puis remonte de deux niveau (.paretn.parent) --> root = dossier racine
sys.path.append(str(root))                              #Liste des endroits ou Python cherche les modules --> Append = rajoute le dossier racine "(ici dans dossier racine: "ao-sospim" 

import matplotlib.pyplot as plt

import src.fourier_tools as fft
import src.image_io as im
import src.io_utils as io
import config as cf

image_name = "Image0042_refZer8"        # <-- A changer pour image a analysiser

image_path = cf.DATA_DIR / f"{image_name}.tif"          
mode = cf.ANALYSIS["projection_mode"]
plane_index = cf.ANALYSIS["plane_index"]

image = im.make_2d_image(
    im.load_tif(image_path),
    mode=mode,
    plane_index=plane_index
)
print("Image :", image_path)
print("mode:", mode)

"""image = m.generate_isotropic_pattern(
    shape=[256, 256],
    frequencies=[10, 10, 10]
)"""

fft_image = fft.getFFT(image)

bins, radial_mean, pixel_counts = fft.calc_rotational_average(
    image,
    max_radius=None,
    apply_smoothing=cf.ANALYSIS["fft_profile_smoothing"],
    sigma=cf.ANALYSIS["fft_profile_sigma"],
    norm=cf.ANALYSIS["fft_profile_normalize"]
)


print("Nb point moyenne rotationnelle:", len(radial_mean))

plt.figure()
plt.imshow(image, cmap="grey")
plt.pause(2)
plt.close()

plt.figure()
plt.imshow(fft_image, cmap="grey")
plt.pause(2)
plt.close()

plt.figure()
plt.plot(bins, radial_mean)
plt.title(f"Moyenne reotationnelle FFT - mode {mode}")
plt.xlabel("Rayon (Pixels)")
plt.ylabel("log(|FFT|)")
plt.grid(True)
plt.show()
plt.pause(2)
plt.close()

# Definition du nom du fichier
filename = (
    f"{cf.ANALYSIS['condition']}_{cf.ANALYSIS['aberration_level']}_exp-{cf.ANALYSIS['no_exp']}_"
    f"{image_name}_rotAvg_"
    f"{cf.ANALYSIS['projection_mode']}"
)
if cf.ANALYSIS["fft_profile_normalize"]:
    filename += (
        f"_norm"
    )
if cf.ANALYSIS["fft_profile_smoothing"]:
    filename += (
        f"_smoothing"
        f"-{cf.ANALYSIS['fft_profile_sigma']}"
    )
filename += ".csv"
output_csv = cf.FFT_DIR / filename

io.save_rotational_average_csv(
    bins,
    radial_mean,
    pixel_counts,
    output_csv,
    image_size_px=image.shape[0],
    analysis_params=cf.ANALYSIS,
    mode=mode,
)
