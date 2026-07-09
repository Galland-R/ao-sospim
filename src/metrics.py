import os
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

import matplotlib.pyplot as plt

import config 
from src import fourier_tools as ft

# Attention pour mean_ROI il faut avoir Napalm_tracer
from palm_tracer import PALMTracer
from palm_tracer.Tools import FileIO
from palm_tracer.Processing import Palm

# -----------------------------------------
# Paramètres pour mean_ROI
# ----------------------------------------- 

NSIGMA = 0.5
ROI_SIZE = 15
SIGMA = 1
THETA = 0
WATERSHED = True
FIT = 3 # Gaussian Fit with X,Y,SigmaX,SigmaY,Theta

# -----------------------------------------
# Fonctions pour mean_ROI
# ----------------------------------------- 


def extract_rois(stack, roi_size: int) -> list[np.ndarray]:
    """
    Extrait les ROI entièrement contenues dans l'image.
    """
    rois = []
    depth, max_height, max_width = stack.shape
    palm = Palm()
    auto = palm.auto_threshold(stack[depth//2,:,:], np.array([roi_size], None))
    #print(f"Seuil trouvé : {auto}")
    localizations = palm.localization(stack, auto, WATERSHED, FIT, np.array([roi_size, 2 * SIGMA, SIGMA, THETA], None))


    half_size = roi_size // 2

    for plane, y, x in zip(
        localizations["Plane"],
        localizations["Y"],
        localizations["X"]
    ):
        x_center = round(x)
        y_center = round(y)

        if (
            x_center - half_size < 0 or
            x_center + half_size >= max_width or
            y_center - half_size < 0 or
            y_center + half_size >= max_height
        ):
            continue

        roi = stack[
            plane - 1,
            y_center - half_size:y_center + half_size + 1,
            x_center - half_size:x_center + half_size + 1
        ]

        rois.append(roi)

    return rois

def filter_rois_by_max_sigma(
    rois: list[np.ndarray],
    n_sigma: float = 1.0
) -> list[np.ndarray]:
    """
    Conserve les ROI dont le maximum est dans
    moyenne ± n_sigma * écart-type.
    """
    if not rois:
        return []

    max_values = np.array([roi.max() for roi in rois])

    mean_max = max_values.mean()
    std_max = max_values.std()

    lower = mean_max - n_sigma * std_max
    upper = mean_max + n_sigma * std_max

    return [
        roi
        for roi, roi_max in zip(rois, max_values)
        if lower <= roi_max <= upper
    ]

def compute_mean_roi(rois: list[np.ndarray]) -> np.ndarray:
    """
    Calcule l'image moyenne pixel par pixel.
    """
    if not rois:
        raise ValueError("La liste des ROI est vide.")

    return np.mean(rois, axis=0)

def average_roi(stack , roi_size=ROI_SIZE, n_sigma = NSIGMA)-> np.ndarray :
    rois = extract_rois(stack,roi_size=ROI_SIZE)
    rois = filter_rois_by_max_sigma(rois, n_sigma=NSIGMA)
    mean_roi = compute_mean_roi(rois)
    return mean_roi

def imax(image: np.ndarray) -> float:
    return float(np.max(image))

# -----------------------------------------
# Fonctions pour fft_band
# ----------------------------------------- 

bp = {"Coverslip":{"MIP" :[22,67],"SUM" :[20,61],"Frame":[16,52],"Dapi" :[11,42],"Gallery" :[16,52],"STD":[23,71],"MIP25":[21,64],"avg_psf" :[31,73]},"Profondeur" :{"MIP" :[16,66],"SUM" :[13,57],"Frame":[7,51],"Dapi" :[6,43],"Gallery" :[11,61],"STD":[16,70],"mip_10frames":[16,66],"mip_30frames":[16,66],"mip_50frames":[16,66],"mip_100frames":[16,66],"std_10frames":[16,70],"std_30frames":[16,70],"std_50frames":[16,70],"std_100frames":[16,70]}} #px


def closest2power(x): 
    x= int(x) 
    return 1 if x == 0 else 2**(x - 1).bit_length()


def get_freq_bandpass(image, radius_interne, radius_externe):
    """
    Créer un masque de fréquence en pixel en anneau.
    """
    rows, cols = image.shape
    # Centre de la FFT
    crow, ccol = rows // 2, cols // 2

    # Distance de chaque pixel au centre
    x, y = np.ogrid[:rows, :cols]
    center_distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Anneau compris entre les deux rayons
    bandpass_mask = np.logical_and(
        center_distance >= radius_interne,
        center_distance <= radius_externe
    )

    return bandpass_mask

def get_metric3(abs_fft, bandpass_mask, n, thisNA, thislambda):

    # Conversion du rayon de coupure de l'OTF en pixels
    freq_cutoff_mm = ((2 * thisNA) / thislambda) * 1000
    radius_cutoff = config.freq_mm1_to_pix(freq_cutoff_mm, image_size_px=n)

    rows = cols = n

    # Centre de la FFT
    crow, ccol = rows // 2, cols // 2

    # Distance au centre
    x, y = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)

    # Masque circulaire correspondant à la fréquence maximale de l'OTF
    circmask1NA = distance <= radius_cutoff

    # Énergie dans le bandpass
    Mnum = np.sum(abs_fft * bandpass_mask)

    # Énergie totale dans l'OTF
    Mden = np.sum(abs_fft * circmask1NA)

    # Métrique normalisée
    return Mnum / Mden

def fft_band(img, jeu, metrique):

    # Récupération des bornes du bandpass (mm^-1)
    freq_min, freq_max = bp[jeu][metrique]

    # Conversion en pixels
    radius_interne = freq_mm1_to_pix(freq_min)
    radius_externe = freq_mm1_to_pix(freq_max)

    # Lecture de l'image et calcul de la FFT

    height, width = img.shape[:2]

    # Compute the closest power of two for the maximum dimension
    n = closest2power(max(width, height))
    abs_fft = ft.getFFT(img)

    # Création du masque bandpass
    bandpass_mask = get_freq_bandpass(
        img,
        radius_interne=radius_interne,
        radius_externe=radius_externe
    )

    # Calcul de la métrique
    M = get_metric3(
        abs_fft,
        bandpass_mask,
        n,
        thisNA=1.49,
        thislambda=0.46
    )

    return M
