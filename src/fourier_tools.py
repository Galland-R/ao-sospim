import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from pathlib import Path

def getFFT(img):

    f = np.fft.fft2(img)                        #do the 2D fourier transform
    fshift = np.fft.fftshift(f)                 #shift FFT to the center
    epsilon = 1e-8                              # Small constant to prevent log(0)

    abs_fft= np.log(np.abs(fshift) + epsilon)   # logarithme du module de la fft: |F| --> log(|FFT(image)|)
    return abs_fft

def calc_rotational_average(image_np, max_radius=None, apply_smoothing=False, sigma=2, norm=True):  
    magnitude = getFFT(image_np)                # Calcul FFT (fonction getFFT)

    rows, cols = image_np.shape                 #Taille image (ex: 256) et définition centre (ex: 128)
    cy, cx = rows // 2, cols // 2

    x = np.arange(cols) - cx                    #Ligne (resp. colonne) arrange de [-cx, .., cx-1] (ex: [-128, -127, ...., 126, 127])
    y = np.arange(rows) - cy
    X, Y = np.meshgrid(x, y)                    #Transforme vecteur x et y en matrice 2D
    R = np.sqrt(X**2 + Y**2)                    #Calcul distance au centre pour obetnir une carte 2D radiale

    if max_radius is None:
        max_radius = min(rows, cols) // 2
    
    radial_mean = np.zeros(max_radius)          #Def tableaux mean et nombre de pixels utilisées (pixel_counts)
    pixel_counts = np.zeros(max_radius)

    for r in range(max_radius):                 #Parcours chaque rayon
        mask = (R >= r) & (R < r + 1)           #Selectionne tous les pixels entre rayons r et r+1
        if np.any(mask):
            radial_mean[r] = magnitude[mask].mean()     #Moyenne des pixels de cette couronne r - r+1  
            pixel_counts[r] = np.count_nonzero(mask)    #Nombre de pixel dans cette couronne

    if apply_smoothing:
        radial_mean = gaussian_filter1d(radial_mean, sigma=sigma)

    if norm:
        radial_mean = radial_mean / np.max(radial_mean) if np.max(radial_mean) !=0 else radial_mean

    return np.arange(max_radius), radial_mean, pixel_counts #Renvoie: les rayons (bins), la moyenne rotationnelle, le nombre de pixels pour chaque rayns


