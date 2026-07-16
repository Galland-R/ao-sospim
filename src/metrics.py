import os
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

import matplotlib.pyplot as plt

import config 
from config import ANALYSIS
from src import fourier_tools as ft

# Attention pour mean_ROI il faut avoir Napalm_tracer
"""
from palm_tracer import PALMTracer
from palm_tracer.Tools import FileIO
from palm_tracer.Processing import Palm
"""
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


def closest2power(x): 
    x= int(x) 
    return 1 if x == 0 else 2**(x - 1).bit_length()


def get_bandpass_mask(n, r_min_px, r_max_px):
    """Construit le masque passe-bande (anneau) en pixels."""
    crow, ccol = n // 2, n // 2
    x, y = np.ogrid[:n, :n]
    center_distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    bandpass_mask = np.logical_and(center_distance > r_min_px, center_distance < r_max_px)
    return bandpass_mask, center_distance


def fft_band(image, NA, Lambda, r_min_px, r_max_px, pixel_size_um=ANALYSIS["pixel_size_um"]):
    """
    Calcule la métrique bandpass sur une image 2D déjà chargée.

    Paramètres :
        image : np.ndarray 2D (déjà chargée via load_tif + make_2d_image)
        NA : ouverture numérique (ex: 1.49 pour Coverslip, 1.27 pour Profondeur)
        Lambda : longueur d'onde en µm (ex: 0.46 pour Dapi, 0.589 sinon)
        r_min_px, r_max_px : bornes de la bande passante en pixels
    """
    n = image.shape[0]

    abs_fft = ft.getFFT(image)

    bandpass_mask, center_distance = get_bandpass_mask(n, r_min_px, r_max_px)

    # fréquence de coupure NA convertie en mm^-1, puis en pixels via freq_mm1_to_pix
    freq_NA_mm1 = (2 * NA) / (Lambda * 1e-3)
    radius_NA_px = config.freq_mm1_to_pix(freq_NA_mm1, image_size_px=n, pixel_size_um=pixel_size_um)

    circmask1NA = (center_distance <= radius_NA_px).astype(float)

    Mnum = np.sum(abs_fft * bandpass_mask)
    Mden = np.sum(abs_fft * circmask1NA)

    return Mnum / Mden
