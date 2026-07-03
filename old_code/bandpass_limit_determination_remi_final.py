#%% Bibliothèque

import numpy as np
import re
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d
import time
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import tifffile
import scipy.stats
import os
import glob
from PyQt6.QtWidgets import QFileDialog, QApplication, QInputDialog
from sklearn.metrics import r2_score
from PIL import Image
# Trouver les plus longues bandes continues
from itertools import groupby
from operator import itemgetter
import pandas as pd

#%% Variables

modes = [
    "Astigmatisme 0°",
    "Astigmatisme 45°",
    "Coma 0°",
    "Coma 90°",
    "Aberration Sphérique",
    "Trefoil 0°",
    "Trefoil 90°"
]

modes_zernike = {
    "Astigmatisme 0°": "Zer4",
    "Astigmatisme 45°": "Zer5",
    "Coma 0°": "Zer6",
    "Coma 90°": "Zer7",
    "Aberration Sphérique": "Zer8",
    "Trefoil 0°": "Zer9",
    "Trefoil 90°": "Zer10"
}
modes_cibles = {
    "+4alphaZer4": "astig4",
    "+4alphaZer5": "astig5",
    "+4alphaZer6": "coma6",
    "+4alphaZer7": "coma7",
    "+4alphaZer8": "spherique8",
    "+4alphaZer9": "trefoil9",
    "+4alphaZer10": "trefoil10",
    "refZer10": "image_ref"
}

IMAGE_METRIQUES = ["MIP","SUM","Frame","Dapi","Gallery","STD","MIP25","avg_psf","avg_psf_zp","mip_10frames","mip_30frames","mip_50frames","mip_100frames","std_10frames","std_30frames","std_50frames","std_100frames"]

METRIQUE_VALUE = ["bandpass","imax", "realm"]

QUANTITES = ["No_aber","Low_aber","Medium_aber","High_aber","High2"]

JEUX = ["1","2","3","4","5","6","7","Tous"]

PROFONDEURS = ["Coverslip","Profondeur"]

bp = {"Coverslip":{"MIP" :[22,67],"SUM" :[20,61],"Frame":[16,52],"Dapi" :[11,42],"Gallery" :[16,52],"STD":[23,71],"MIP25":[21,64],"avg_psf" :[31,73]},"Profondeur" :{"MIP" :[16,66],"SUM" :[13,57],"Frame":[7,51],"Dapi" :[6,43],"Gallery" :[11,61],"STD":[16,70],"mip_10frames":[16,66],"mip_30frames":[16,66],"mip_50frames":[16,66],"mip_100frames":[16,66],"std_10frames":[16,70],"std_30frames":[16,70],"std_50frames":[16,70],"std_100frames":[16,70]}} #px

lowpass = {"Coverslip":{"MIP" :[0,22],"SUM" :[0,20],"Frame":[0,16],"Dapi" :[0,11],"Gallery" :[0,16],"STD":[0,23],"MIP25":[0,21],"avg_psf" :[0,31]},"Profondeur" :{"MIP" :[0,16],"SUM" :[0,13],"Frame":[0,7],"Dapi" :[0,6],"Gallery" :[0,11],"STD":[0,16],"mip_10frames":[0,16],"mip_30frames":[0,16],"mip_50frames":[0,16],"mip_100frames":[0,16]}} #px

highpass = {"Coverslip":{"MIP" :[67,88],"SUM" :[61,88],"Frame":[52,88],"Dapi" :[11,88],"Gallery" :[52,88],"STD":[71,88],"MIP25":[64,88],"avg_psf" :[73,88]},"Profondeur" :{"MIP" :[66,87],"SUM" :[57,87],"Frame":[51,87],"Dapi" :[43,87],"Gallery" :[61,87],"STD":[70,87],"mip_10frames":[66,87],"mip_30frames":[66,86],"mip_50frames":[66,86],"mip_100frames":[66,87]}} #px


fc = {
    "Coverslip": {
        "MIP": 2614,
        "SUM": 2614,
        "Frame": 2614,
        "Dapi": 3232,
        "Gallery": 2614,
        "STD": 2614,
    },
    "Profondeur": {
        "MIP": 2228,
        "SUM": 2228,
        "Frame": 2228,
        "Dapi": 2755,
        "Gallery": 2228,
        "STD": 2228,
    }
}#pas à jour
#mm-1
conditions = ["Coverslip","Profondeur"]

#%% Fonctions

def get_image_key(image_obj):
    return image_obj if isinstance(image_obj, str) else id(image_obj)

#Calcul la moyenne rotationnelle
def calc_rotational_average(image_np, max_radius, apply_smoothing=False, sigma=2):  
    magnitude = getFFT(image_np)
    rows, cols = image_np.shape
    cy, cx = rows // 2, cols // 2
    x = np.arange(cols) - cx
    y = np.arange(rows) - cy
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    radial_mean = np.zeros(max_radius)
    pixel_counts = np.zeros(max_radius)

    for r in range(max_radius):
        mask = (R >= r) & (R < r + 1)
        if np.any(mask):
            radial_mean[r] = magnitude[mask].mean()
            pixel_counts[r] = np.count_nonzero(mask)
        else:
            radial_mean[r] = 0
            pixel_counts[r] = 0

    if apply_smoothing:
        radial_mean = gaussian_filter1d(radial_mean, sigma=sigma)

    return np.arange(max_radius), radial_mean, pixel_counts

#  Génération d'une image sinusoïdale de référence pr tests
def generate_sinusoidal_image(shape=(256, 256), frequency=30, amplitude=1.0):
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    return amplitude * np.sin(2 * np.pi * frequency * X / cols)

# Génération d'une image isotropique pr tests
def generate_isotropic_pattern(shape=(256, 256), frequencies=[10, 20, 30]):
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    pattern = np.zeros_like(X, dtype=float)
    for freq in frequencies:
        angle = np.random.uniform(0, 2*np.pi)
        pattern += np.sin(2 * np.pi * freq * (np.cos(angle) * X + np.sin(angle) * Y) / cols)
    return pattern / len(frequencies)

#Détection de la bande
def detect_band(strategy="fixed_threshold", bins=None, profile=None, **kwargs):
    if strategy not in {"fixed_threshold", "adaptive_threshold", "second_derivative", "integral_based"}:
        raise ValueError("Unknown strategy")

    if strategy == "fixed_threshold":
        ratio = kwargs.get("ratio", 0.5)
        threshold = ratio * np.max(profile)
        mask = profile >= threshold

    elif strategy == "adaptive_threshold":
        k = kwargs.get("k", 1.0)
        threshold = np.mean(profile) + k * np.std(profile)
        mask = profile >= threshold

    elif strategy == "second_derivative":
        d2 = np.gradient(np.gradient(profile))
        threshold = kwargs.get("d2_threshold", np.std(d2))
        mask = np.abs(d2) > threshold

    elif strategy == "integral_based":
        min_w = kwargs.get("min_width", 5)
        best_band = (0, 0)
        best_score = -np.inf
        for i in range(len(bins)):
            for j in range(i + min_w, len(bins)):
                score = np.sum(profile[i:j])
                if score > best_score:
                    best_score = score
                    best_band = (i, j)
        r_min = bins[best_band[0]]
        r_max = bins[best_band[1]]
        return r_min, r_max, best_score

    indices = np.where(mask)[0]
    groups = [list(g) for k, g in groupby(indices, key=lambda i, c=iter(range(len(indices))): i - next(c))]
    longest_band = max(groups, key=len) if groups else []

    if longest_band:
        r_min = bins[longest_band[0]]
        r_max = bins[longest_band[-1]]
        return r_min, r_max, r_max - r_min
    else:
        return None, None, 0

def detect_adaptive_band(bins, profile, k=1.0, ignore_first_n=10):
    profile_crop = profile[ignore_first_n:]
    bins_crop = bins[ignore_first_n:]

    threshold = np.mean(profile_crop) + k * np.std(profile_crop)
    mask = profile_crop >= threshold
    indices = np.where(mask)[0]

    from itertools import groupby
    groups = [list(g) for k_, g in groupby(indices, key=lambda i, c=iter(range(len(indices))): i - next(c))]
    longest_band = max(groups, key=len) if groups else []

    if longest_band:
        r_min = bins_crop[longest_band[0]]
        r_max = bins_crop[longest_band[-1]]
        return r_min, r_max, r_max - r_min
    else:
        return None, None, 0

#Calcul de l'aire sous la courbe dans la bande
def compute_integral_in_band(r_min, r_max, bins, profile):
    if r_min is None or r_max is None:
        return 0
    mask = (bins >= r_min) & (bins <= r_max)
    return np.trapz(profile[mask], bins[mask])

# Récupération des images dans un dossier
def get_images_from_folder_qt(extensions=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    folder_path = QFileDialog.getExistingDirectory(
        None,
        "Sélectionnez le dossier contenant les images",
        os.getcwd()
    )

    if not folder_path:
        print("Aucun dossier sélectionné.")
        return []

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    print(f"{len(image_paths)} image(s) trouvée(s) dans : {folder_path}")
    return image_paths

#Extrait le suffixe de l'image contenant l'aberration induite et sa quantité
def extraire_suffixe(chemin):
    filename = os.path.basename(chemin)  # "Image0003_-3alphaZer4.tif"
    name, _ = os.path.splitext(filename)  # "Image0003_-3alphaZer4"
    
    if "_" in name:
        return name.split("_")[-1]  # "-3alphaZer4"
    else:
        return None  # ou name si tu veux tout renvoyer

def choisir_element_liste(titre, texte, liste_elements):
    """
    Boîte de dialogue pour sélectionner un élément dans une liste.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    element, ok = QInputDialog.getItem(
        None, titre, texte, liste_elements, editable=False
    )
    return element if ok else None

#Récupération de toutes les images où un mode particulier a été induit
def recuperer_images_par_mode(image_paths, mode):
    chemins_mode = []
    for path in image_paths:
        suffixe = extraire_suffixe(path)
        if suffixe and suffixe.endswith(mode):
            chemins_mode.append(path)
    return chemins_mode

#Calcul de la FFT
def getFFT(img):
    f = np.fft.fft2(img) #do the 2D fourier transform
    fshift = np.fft.fftshift(f) #shift FFT to the center
    magnitude_spectrum = fshift
    epsilon = 1e-8  # Small constant to prevent log(0)

    abs_fft= np.log(np.abs(magnitude_spectrum)+epsilon)
    # plt.imshow(abs_fft)
    # plt.show()
    # abs_fft= swapquadrants(abs_fftpre)

    return abs_fft

#Création de paires d'images à partir d'un jeu d'images et d'une référence
def creer_images_paires(image_a_pairer,image_ref):
    image_pairs = []
    for img in image_a_pairer:
        image_pairs.append((img,image_ref))
    return image_pairs

def projection_folder(input_folder, output_folder, mode="max", frame_idx=None, max_z=None):
    """
    Applique une opération de projection sur un dossier d’images empilées 3D.
    mode : "max", "std", "sum", "frame"
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        filepath = os.path.join(input_folder, filename)
        img = imageio.volread(filepath)

        if img.ndim != 3:
            result = img
        else:
            volume = img if max_z is None else img[:max_z]

            if mode == "max":
                result = np.max(volume, axis=0)
            elif mode == "std":
                result = np.std(volume, axis=0)
            elif mode == "sum":
                result = np.sum(volume, axis=0)
            elif mode == "frame":
                if frame_idx is None or frame_idx >= volume.shape[0]:
                    raise ValueError(f"Frame index {frame_idx} invalide pour {filename}")
                result = volume[frame_idx]
            else:
                raise ValueError(f"Mode de projection inconnu : {mode}")

        save_path = os.path.join(output_folder, filename)
        imageio.imwrite(save_path, result.astype(np.uint16))
        print(f"{mode.upper()} sauvegardé : {save_path}")

    print(f"✅ Projection '{mode}' terminée.")

def importation_chemin(choix_image_metrique,choix_quantite,choix_jeu,choix_profondeur):
    if choix_jeu == "Tous":
        dic_images_paths={}
        if choix_profondeur == "Profondeur" and choix_quantite == "No_aber" and choix_image_metrique!="Dapi":
            n = 7
        else:
            n=5
        for i in range (n):
            paths = []
            dossier = f"C:/Users/rlusson/Desktop/Data/Carac_AO/{choix_profondeur}/{choix_quantite}/{i+1}/{choix_image_metrique}"
            for root, _, files in os.walk(dossier):
                for file in files:
                    if file.lower().endswith(".tif"):
                        paths.append(os.path.join(root, file))
            dic_images_paths[f"{i+1}"] = paths
        return dic_images_paths
    else :
        images_paths = []
        dossier = f"C:/Users/rlusson/Desktop/Data/Carac_AO/{choix_profondeur}/{choix_quantite}/{choix_jeu}/{choix_image_metrique}"
        for root, _, files in os.walk(dossier):
            for file in files:
                if file.lower().endswith(".tif"):
                    images_paths.append(os.path.join(root, file))
        return images_paths
    
def extraire_quantite_alpha(nom_fichier):
    """
    Extrait la quantité d’alpha d’un nom de fichier, par exemple :
    - Image0064_+4alphaZer10.tif → 4
    - Image0006_refZer4.tif      → 0 (cas "ref")
    """
    # Cas du fichier "ref" (référence)
    if "ref" in nom_fichier.lower():
        return 0

    # Regex pour trouver la quantité d’alpha (ex: +4alpha ou -3alpha)
    match = re.search(r'([+-]?\d+)alpha', nom_fichier)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Impossible d’extraire la quantité d’alpha de : {nom_fichier}")

############Métrique

def closest2power(x): 
    x= int(x) 
    return 1 if x == 0 else 2**(x - 1).bit_length()

def getImage(img_input):
    
    
    if isinstance(img_input, str):
        # It's a file path; read the image from disk
        img = imageio.imread(img_input)
    else:
        # It's already an image array
        img = img_input

    # Ensure the image is a NumPy array
    img = np.array(img)

    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Compute the closest power of two for the maximum dimension
    n = closest2power(max(width, height))
    
    return n, img

def get_radius(px_size_um, freq_voulue_mm, taille_image_px ):
    px_size= px_size_um * 0.001

    freq_echantillonage= 1/(px_size)

    radius= (freq_voulue_mm * taille_image_px)/freq_echantillonage
    radius = int(radius)
    
    return radius

def pix_to_freq(pixels_value, px_size_um = 0.108, taille_image_px=256):
    
    
    px_size= px_size_um * 0.001

    freq_echantillonage= 1/(px_size)

    freq= (freq_echantillonage*pixels_value) / ( taille_image_px)
    freq = int(freq)
    
    return freq

def afficher_masque_bandpass(taille_image, px_size_um, freq_interne, freq_externe):
    """
    Affiche le masque passe-bande fréquentiel basé sur la taille de l'image et les fréquences spatiales internes et externes.

    Paramètres :
    - taille_image : tuple (rows, cols) en pixels
    - px_size_um : taille d'un pixel en µm
    - freq_interne : fréquence spatiale interne (mm⁻¹)
    - freq_externe : fréquence spatiale externe (mm⁻¹)
    """
    rows, cols = taille_image
    crow, ccol = rows // 2, cols // 2

    # Convertir µm en mm
    px_size_mm = px_size_um * 0.001

    # Fréquence d’échantillonnage (en mm⁻¹)
    fe_x = 1 / px_size_mm
    fe_y = 1 / px_size_mm

    # Rayon en pixels dans chaque direction
    radius_int_x = (freq_interne * cols) / fe_x
    radius_int_y = (freq_interne * rows) / fe_y
    radius_ext_x = (freq_externe * cols) / fe_x
    radius_ext_y = (freq_externe * rows) / fe_y

    # Grilles normalisées
    y, x = np.ogrid[:rows, :cols]
    norm_x = (x - ccol) / radius_ext_x
    norm_y = (y - crow) / radius_ext_y

    dist = np.sqrt(norm_x**2 + norm_y**2)

    # Masque passe-bande
    bandpass_mask = np.logical_and(dist < 1, dist > radius_int_x / radius_ext_x)

    # Affichage
    plt.figure(figsize=(6, 6))
    plt.imshow(bandpass_mask, cmap='gray')
    plt.title(f"Masque passe-bande ({freq_interne}-{freq_externe} mm⁻¹)")
    plt.axis('off')
    plt.show()

    #return bandpass_mask

def get_freq_bandpass(image,px_size_um, taille_image_px, freq_interne, freq_externe):
    #print(image.shape)
    rows, cols = image.shape

    

    crow, ccol = rows // 2, cols // 2 

    radius_interne= get_radius(px_size_um, freq_voulue_mm=freq_interne, taille_image_px=taille_image_px)
    radius_externe=get_radius(px_size_um, freq_voulue_mm=freq_externe, taille_image_px=taille_image_px)

    x,y= np.ogrid[:rows, :cols]
    center_distance= np.sqrt((x-ccol)**2+(y-crow)**2)
    bandpass_mask= np.logical_and(center_distance>radius_interne, center_distance<radius_externe)

    return bandpass_mask

def get_lowpass_from_bandpass(weighted_mask: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """
    Génère un masque lowpass [0, fmax] (valeurs 1.0 dans un disque centré)
    à partir d’un masque pondéré comme `metric_weight`.

    Paramètres :
        weighted_mask : np.ndarray (float)
            Masque avec pondération fréquentielle (ex : metric_weight)
        threshold : float
            Valeur en dessous de laquelle le masque est considéré comme nul

    Retourne :
        lowpass_mask : np.ndarray bool
            Masque lowpass [0, fmax] en pixels
    """
    rows, cols = weighted_mask.shape
    crow, ccol = rows // 2, cols // 2

    # Coordonnées et distances au centre
    y, x = np.ogrid[:rows, :cols]
    center_distance = np.sqrt((x - ccol)**2 + (y - crow)**2)

    # Trouver le rayon max en pixels : là où le masque est encore significatif
    nonzero_mask = weighted_mask > threshold
    radius_fmax_px = center_distance[nonzero_mask].max()

    # Construire le masque lowpass
    lowpass_mask = (center_distance <= radius_fmax_px).astype(float)

    return lowpass_mask

def getfreqsup4_ref(n, param_alpha, px_size, thisNA, thislambda):
    # Create 2D arrays for fx and fy using NumPy broadcasting 
    fx = np.outer((np.arange(n) - n // 2) / (2 * n) / px_size, np.ones(n)) 
    fy = np.outer(np.ones(n), (np.arange(n) - n // 2) / (n) / px_size) 
    fr = np.abs(np.sqrt(fy**2 + fx**2)) 

    # Create circular masks    
    circmask1NA = (fr <= ( thisNA) / thislambda) / 1.0 
    circmask2NA = (fr <= (2 * thisNA) / thislambda) / 1.0 

    # Compute optical transfer function (otf) 
    otf = np.where(fr <= (thisNA) / thislambda, (2 / np.pi) * np.arccos(fr / ((thisNA) / thislambda)) - ((fr / ((thisNA) / thislambda)) * (np.sqrt(1 - (fr**2 / ((thisNA) / thislambda)**2)))), 0) 
    #otf = np.where(fr <= (2 * thisNA) / thislambda, (2 / np.pi) * np.arccos(fr / ((2 * thisNA) / thislambda)) - ((fr / ((2 * thisNA) / thislambda)) * (np.sqrt(1 - (fr**2 / ((2 * thisNA) / thislambda)**2)))), 0) 

    # Compute metric weight
    metric_weight = np.power(1 - otf, param_alpha) * otf * circmask1NA 

    # Normalize metric_weight to have values between 0 and 1 
    metric_weight /= np.max(metric_weight) 

    
    
    return metric_weight, circmask1NA

def getfreqsup4(n, param_alpha, beta, px_size, thisNA, thislambda):
    """
    Calcule le masque de pondération fréquentielle (metric_weight) basé sur l'OTF
    pour une image de taille (n, n), avec gestion robuste des instabilités numériques.
    """

    # Fréquence spatiale normalisée
    fx = np.outer((np.arange(n) - n // 2) / n / px_size, np.ones(n))
    fy = np.outer(np.ones(n), (np.arange(n) - n // 2) / n / px_size)
    fr = np.sqrt(fx**2 + fy**2)

    # Fréquence de coupure : NA / lambda
    #fc = 2*thisNA / thislambda
    #fc=0.003846
    fc=0.003125

    # Masques circulaires
    circmask1NA = (fr <= fc/2).astype(float)
    circmask2NA = (fr <=  fc).astype(float)  # non utilisé ici mais potentiellement utile

    # Normalisation sécurisée de fr (évite racines négatives et arccos > 1)
    fr_norm = np.clip(fr / fc, 0, 1)

    # OTF avec domaine restreint
    otf = np.zeros_like(fr)
    inside_mask = fr <= fc
    otf[inside_mask] = (2 / np.pi) * (
        np.arccos(fr_norm[inside_mask]) -
        fr_norm[inside_mask] * np.sqrt(1 - fr_norm[inside_mask]**2)
    )

    # Calcul du poids métrique
    metric_weight = np.power(1 - otf**beta, param_alpha) * otf * circmask1NA

    # Normalisation
    metric_weight /= np.max(metric_weight)
    #print("fr[n//2, n//2] =", fr[n//2, n//2])
    #print("OTF[n//2, n//2] =", otf[n//2, n//2])
    #print("metric_weight[n//2, n//2] =", metric_weight[n//2, n//2])
    return metric_weight, circmask1NA

def getMetric(abs_fft, metric_weight, circmask1NA,n):
    Mnum=0
    Mden=0
    M=0
    
    i=0
    while i < n:
        j=0
        while j<n:
            
            Mnum+= abs_fft[i][j]* metric_weight[i][j]
            Mden+= abs_fft[i][j]*circmask1NA[i][j]
            j=j+1

        i=i+1
        
    M= Mnum / Mden

    return M

def get_metric3(abs_fft, bandpass_mask, n, thisNA, thislambda,px_size):
    
    fx = np.outer((np.arange(n) - n // 2) / ( 2*n) / px_size, np.ones(n)) 
    fy = np.outer(np.ones(n), (np.arange(n) - n // 2) / ( 2*n) / px_size) 
    fr = np.abs(np.sqrt(fy**2 + fx**2)) 

    #fc=2*thisNA/thislambda
    if thisNA == 1.27:
        fc=3.175 #m-1
    else:
        fc=3.215 #m-1
    
    # Create circular masks    
    circmask1NA = (fr <= fc) / 1.0 
    mask_norm = get_lowpass_from_bandpass(bandpass_mask) 
    #plt.imshow(mask_norm)
    #plt.show()
    #plt.imshow(bandpass_mask)
    #plt.show()
    

    Mnum=0
    Mden=0
    M=0
    
    i=0
    while i < n:
        j=0
        while j<n:
            
            Mnum+= abs_fft[i][j]* bandpass_mask[i][j]
            #Mden+= abs_fft[i][j]*mask_norm[i][j]
            Mden+= abs_fft[i][j]*circmask1NA[i][j]
            j=j+1

        i=i+1
        
    M= Mnum / Mden
    #mettre M ou 1/M Suivant ce qUE L4ON VEUT
    return 1/M
    
def this_realm(img_path,NA,Lambda):
    n, img= getImage(img_path)
    a = 1.3
    b = 1
    metric_weight, circmask1NA= getfreqsup4(n, beta =b ,param_alpha=a, px_size=108, thisNA=NA, thislambda=Lambda) #px size en nm
    #plt.imshow(circmask1NA*metric_weight)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(circmask1NA)
    #plt.show()
    #print(metric_weight[127])

    h, w = metric_weight.shape
    center_y = h // 2
    
    # Profil horizontal au centre vertical
    profile = metric_weight[center_y, :]

    #plt.plot(profile)
    #plt.title(f"Profil horizontal au centre – alpha={a}, beta={b}")
    #plt.xlabel("X")
    #plt.ylabel("Intensité")
    #plt.grid(True)
    #plt.show()
    

    #x= np.linspace(2.5,0,256)
    #plt.plot(x,(metric_weight[255])) ##SHow plot metric weight
    #plt.show()

    

    abs_fft= getFFT(img)
    fft_img= fftshift(fft2(img))

    #plt.imshow(img)
    #plt.show()

    #plt.imshow(np.real(fft_img), cmap='gray')
    #plt.show()

    #plt.imshow(np.real(fft_img*metric_weight))
    #plt.show()

    #plt.imshow( np.real(ifft2(ifftshift(abs_fft*metric_weight))), cmap='gray')
    #plt.show()
    #plt.imshow( np.real(ifft2(ifftshift(abs_fft))), cmap='gray')
    #plt.show()

   #plt.figure(num=None, figsize=(5, 5), dpi=100)
    #plt.imshow(np.log(abs_fft), cmap = 'gray')
    #plt.show()

    #plt.imshow(metric_weight)
    #plt.show()


    thisM=get_metric3(abs_fft, metric_weight, n, thisNA=NA, thislambda=Lambda,px_size=0.108)

    return thisM    

def this_bandpass2(img_path,NA,Lambda,freq_min,freq_max):
    
    n, img= getImage(img_path)
    
    abs_fft= getFFT(img)

    bandpass_mask= get_freq_bandpass(image=img,px_size_um=0.108, taille_image_px=256,freq_interne = freq_min, freq_externe = freq_max) ###BP on actinGFP/DAPI : 450-2100mm-1 // on matrices of SM : 940-1870mm-1 //MIP:669-2242 mm-1

    M=get_metric3(abs_fft, bandpass_mask, n, thisNA=NA, thislambda=Lambda,px_size=0.108)
    # M=get_metric2(abs_fft, bandpass_mask, n)

    # plt.imshow(img)
    # plt.show()
    # plt.imshow(abs_fft)
    # plt.show()
    #plt.imshow(bandpass_mask)
    #plt.show()
    
    # visu= bandpass_mask*abs_fft
    # plt.imshow(visu)
    # plt.show()
    

    return M

def imax(image):
    img = imageio.imread(image) if isinstance(image, str) else image
    return float(np.max(img))

def generic_metric_9N(images, metric_func,thisNA, thislambda, *metric_args, **metric_kwargs):
    """
    Applique une fonction métrique (ex: bandpass, imax, realm) sur 8 images pour le schéma 9N.
    
    Paramètres :
        images : list de 8 images [m3, m2, m1, ref, p1, p2, p3, p4]
        metric_func : fonction de la métrique à appliquer à chaque image
        *metric_args, **metric_kwargs : paramètres optionnels à passer à la métrique
    
    Retour :
        Tuple de 8 valeurs (métriques)
    """
    if len(images) != 8:
        raise ValueError("La fonction attend exactement 8 images.")
    
    return tuple(metric_func(img, thisNA, thislambda,*metric_args, **metric_kwargs) for img in images)


#################### MIP########################

def mip_metric_9N(imagem4, imagem3, imagem2, imagem1, imageref, imagep1, imagep2, imagep3, imagep4):
    def mip_bandpass(img_path):
        mip_img= get_max_intensity_proj(img_path, start_frame=0,end_frame='all')
        M= this_bandpass2(mip_img)
        return M
    
    ym4= mip_bandpass(imagem4)
    ym3=mip_bandpass(imagem3)
    ym2=mip_bandpass(imagem2)
    ym1=mip_bandpass(imagem1)
    yref=mip_bandpass(imageref)
    yp1=mip_bandpass(imagep1)
    yp2=mip_bandpass(imagep2)
    yp3=mip_bandpass(imagep3)
    yp4=mip_bandpass(imagep4)

    return ym4, ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4

############### fonctions fits


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A,B,C

def get_fit_array_9N(ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4):
    alpha_array=np.linspace(-float(0.195), +float(0.26), 8)
    x_min=-float(0.195)
    xplus=float(0.26)

    x = np.linspace(x_min, xplus, 8)
    x_np= np.array(x, dtype=np.float64)

    y=[ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4]
    y_np=np.array(y, dtype=np.float64)

    fit_array=[x_np,y_np]

    return fit_array

def fit(array, show_plot=False):
    xl = array[0]  # x data
    yl = array[1]  # y data

    fit = np.polyfit(xl, yl, 2)
    poly = np.poly1d(fit)
    y_pred = poly(xl)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xl, yl)
    R2 = 1 - (r_value ** 2)
    r2 = r2_score(yl, y_pred)

    if len(xl) == 3:
        a, b, c = calc_parabola_vertex(xl[0], yl[0], xl[1], yl[1], xl[2], yl[2])
    elif len(xl) > 3:
        c = poly[0]
        b = poly[1]
        a = poly[2]

    if show_plot:
        xn = np.linspace(min(xl), max(xl), 200)
        yn_quad = a * xn**2 + b * xn + c

        plt.figure()
        plt.plot(xl, yl, 'o', label='Data')
        plt.plot(xn, yn_quad, '--', label='Quadratic approximation')
        plt.text(0.05, 0.88, f"$r^2$  = {r2:.4f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')
        plt.legend()
        plt.title("Quadratic fit")
        plt.xlabel("Quantity of aberration in µRMS")
        plt.ylabel("Weighted difference")
        plt.grid()
        plt.show()

    return a, b, c, xl, yl, R2, r2

def gaussienne(x, A, mu, sigma,offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def estimer_p0_gaussienne(x, y):
    """
    Estime les paramètres initiaux (p0) pour un fit gaussien.
    
    Paramètres :
    - x : array-like, les abscisses
    - y : array-like, les ordonnées
    
    Retour :
    - p0 : tuple (A, mu, sigma, offset)
    """
    A = np.max(y) - np.min(y)              # Amplitude approximative
    mu = x[np.argmax(y)]                   # Position du maximum
    offset = np.min(y)                     # Niveau de base
    sigma = (np.max(x) - np.min(x)) / 4    # Largeur estimée (1/4 plage x)
    L = [A, mu, sigma, offset]

    return L

def fit_gaussien(x, y, p0, show_plot=True):
    # Fit gaussien
    popt, pcov = curve_fit(gaussienne, x, y, p0)
    A, mu, sigma, offset = popt

    # Fit quadratique approché autour du sommet de la gaussienne
    a = -1 / (2 * sigma**2)
    b = mu / (sigma**2)
    c = A + offset  # Valeur au sommet

    # Prédictions pour R²
    y_fit = gaussienne(x, *popt)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - ss_res / ss_tot

    # Affichage
    if show_plot:
        xn = np.linspace(min(x), max(x), 200)
        yn_gauss = gaussienne(xn, *popt)
        yn_quad = a * xn**2 + b * xn + c

        plt.figure()
        plt.plot(x, y, 'o', label='Données')
        plt.plot(xn, yn_gauss, '-', label='Fit gaussien')
        plt.plot(xn, yn_quad, '--', label='Approx. quadratique')
        plt.legend()
        plt.title("Fit gaussien et approximation quadratique")
        plt.grid()
        plt.show()

    return a, b, c, x, y, R2

#################### Fréquence########################

def apply_bandpass_filter(input_path, output_path, bandpass, save=False, show=True):
    """
    Applique une bande passante fréquentielle à une image en FFT.
    Conserve uniquement les fréquences dans la bande [lo, hi], puis effectue l'IFFT.
    
    Paramètres :
    - input_path : chemin de l'image d'entrée
    - output_path : chemin pour sauvegarder l'image
    - bandpass : tuple (lo, hi) en mm⁻¹
    - save : bool, si True enregistre l’image filtrée
    - show : bool, si True affiche l’image filtrée
    """
    # Charger l’image
    image = imageio.imread(input_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {input_path}")

    if image.ndim == 3:
        image = image[..., 0]  # RGB → grayscale

    rows, cols = image.shape
    lo, hi = bandpass

    # FFT centrée
    fft_image = fftshift(fft2(image))

    # Masque de bande passante
    mask = get_freq_bandpass(image, px_size_um=0.108,taille_image_px=256, freq_interne=lo, freq_externe=hi)
    if not np.any(mask):
        print(f"⚠️ Aucun pixel dans la bande [{lo} - {hi}] mm⁻¹")

    # Application du masque
    fft_filtered = fft_image * mask
    image_filtered = np.real(ifft2(ifftshift(fft_filtered)))

    # Normalisation
    image_filtered -= image_filtered.min()
    image_filtered /= image_filtered.max()
    image_filtered *= 255
    image_uint8 = image_filtered.astype(np.uint8)

    # Affichage si demandé
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_uint8, cmap="gray")
        plt.title(f"Image filtrée : bande [{lo} - {hi}] mm⁻¹")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Sauvegarde si demandé
    if save:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        imageio.imwrite(output_path, image_uint8)
        print(f"✅ Image enregistrée dans : {output_path}")
    else:
        print("ℹ️ Image non enregistrée (save=False)")

def compute_fft_frequencies(image_path, pixel_size_mm):
    """
    Calcule et affiche le spectre de fréquences spatiales d'une image en mm⁻¹.

    Paramètres :
        image_path (str) : chemin vers l'image (grayscale de préférence).
        pixel_size_mm (float) : taille d'un pixel en millimètres.
    
    Retourne :
        fx (np.ndarray) : fréquences spatiales selon x (en mm⁻¹).
        fy (np.ndarray) : fréquences spatiales selon y (en mm⁻¹).
        magnitude (np.ndarray) : spectre FFT centré.
    """
    
    # Lecture image
    image = imageio.imread(image_path)
    if image.ndim == 3:
        image = image.mean(axis=2)  # conversion en niveau de gris si RGB

    rows, cols = image.shape

    # FFT 2D centrée
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude = np.abs(fft_shifted)

    # Fréquences spatiales (en mm⁻¹)
    fx = np.fft.fftshift(np.fft.fftfreq(cols, d=pixel_size_mm))
    fy = np.fft.fftshift(np.fft.fftfreq(rows, d=pixel_size_mm))

    # Affichage
    plt.figure(figsize=(8, 6))
    extent = [fx[0], fx[-1], fy[0], fy[-1]]
    plt.imshow(np.log1p(magnitude), extent=extent, cmap='gray', origin='lower', aspect='auto')
    plt.xlabel('Fréquence spatiale X (mm$^{-1}$)')
    plt.ylabel('Fréquence spatiale Y (mm$^{-1}$)')
    plt.title('Spectre en fréquence (FFT)')
    plt.colorbar(label='log(Amplitude)')
    plt.tight_layout()
    plt.show()

    return fx, fy, magnitude

def load_image_grayscale(path):
    image = Image.open(path).convert("L")  # convertit en niveaux de gris
    image = np.array(image)
    #print("✅ Image chargée avec shape :", image.shape)
    return image

#################### Padding########################

def feathered_padding(image_path, target_size=256 ,show=False,save=False):
    """
    Applique un zero-padding avec une fenêtre de Hanning pour réduire les artefacts dans les basses fréquences.
    Affiche en option l'image paddée et sa transformée de Fourier (magnitude log).

    Args:
        image: ndarray (image 2D)
        target_size: taille finale (ex: 256)
        show: booléen pour activer les affichages

    Returns:
        image paddée (ndarray)
    """
    image = imageio.imread(image_path)
    h, w = image.shape
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2

    # Fenêtre Hanning 2D
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)
    image_windowed = image * window

    # Zero-padding
    padded = np.zeros((target_size, target_size), dtype=np.float32)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image_windowed

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Image paddée
        axes[0].imshow(padded, cmap='gray')
        axes[0].set_title("Image paddée avec bords fondus")
        axes[0].axis("off")

        # FFT magnitude
        fft = np.fft.fftshift(np.fft.fft2(padded))
        fft_magnitude = np.log1p(np.abs(fft))
        axes[1].imshow(fft_magnitude, cmap='magma')
        axes[1].set_title("Magnitude FFT (log)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    #if save:
        #if output_path is None:
            #raise ValueError("⚠️ Veuillez spécifier 'output_path' si save=True.")
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #result_uint8 = (padded - padded.min()) / (padded.max() - padded.min()) * 255
        #imageio.imwrite(output_path, result_uint8.astype(np.uint8))

    return padded

def feathered_padding_folder(input_folder, output_folder, target_shape=256):
    """
    Applique feathered_padding à toutes les images d’un dossier et sauvegarde les résultats.
    
    Args:
        input_folder (str): dossier d'entrée contenant les images 2D.
        output_folder (str): dossier où enregistrer les images paddées.
        target_shape (tuple): dimensions finales (par défaut 256x256).
        sigma (float): écart-type du flou de transition.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            image = imageio.imread(input_path)
            if image.ndim != 2:
                print(f"⛔ {filename} ignorée (image non 2D).")
                continue

            padded_image = feathered_padding(input_path, target_size=target_shape)

            # Normalisation pour l'enregistrement
            padded_image -= padded_image.min()
            padded_image /= padded_image.max()
            padded_image *= 65535  # pour du uint16
            padded_image_uint16 = padded_image.astype(np.uint16)

            output_path = os.path.join(output_folder, filename)
            imageio.imwrite(output_path, padded_image_uint16)
            print(f"✅ Sauvegardée : {output_path}")

    print("🎉 Traitement terminé pour tous les fichiers.")

def zero_padding(image, target_shape):
    pad_y = target_shape[0] - image.shape[0]
    pad_x = target_shape[1] - image.shape[1]
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

def random_noise_padding(image, target_shape):
    pad_y = target_shape[0] - image.shape[0]
    pad_x = target_shape[1] - image.shape[1]
    if pad_y < 0 or pad_x < 0:
        raise ValueError("Target shape must be larger que l'image initiale.")

    borders = np.concatenate([
        image[0, :], image[-1, :], image[:, 0], image[:, -1]
    ])
    mean, std = np.mean(borders), np.std(borders)
    padded = np.random.normal(mean, std, size=target_shape)
    y0, x0 = pad_y // 2, pad_x // 2
    padded[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image
    return padded

def spline_interpolation(image, target_shape, order=3):
    y = np.arange(image.shape[0])
    x = np.arange(image.shape[1])
    spline = RectBivariateSpline(y, x, image, kx=order, ky=order)
    y_new = np.linspace(0, image.shape[0] - 1, target_shape[0])
    x_new = np.linspace(0, image.shape[1] - 1, target_shape[1])
    return spline(y_new, x_new)

def show_comparison(original, padded_dict, output_path):
    fig, axs = plt.subplots(2, len(padded_dict) + 1, figsize=(12, 6))
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Original")
    axs[0, 0].axis('off')
    axs[1, 0].imshow(np.log1p(np.abs(fftshift(fft2(original)))), cmap='gray')
    axs[1, 0].set_title("FFT")
    axs[1, 0].axis('off')

    for i, (label, img) in enumerate(padded_dict.items(), start=1):
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(label)
        axs[0, i].axis('off')
        axs[1, i].imshow(np.log1p(np.abs(fftshift(fft2(img)))), cmap='gray')
        axs[1, i].set_title(f"FFT {label}")
        axs[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def construire_image_paires(image_paths, modes, modes_zernike):
    image_paires = {}
    for m in modes:
        images_mode = recuperer_images_par_mode(image_paths, modes_zernike[m])
        image_ref = next((p for p in images_mode if extraire_suffixe(p) == "ref" + modes_zernike[m]), "")
        image_paires[m] = creer_images_paires(images_mode, image_ref)[1:]  # skip ref pair
    return image_paires

def construire_image_paires_par_jeu(image_paths_dict, modes, modes_zernike, n):
    dic_image_paires = {}
    for i in range(n):
        dic_image_paires[f"{i+1}"] = {}
        for m in modes:
            images_mode = recuperer_images_par_mode(image_paths_dict[f"{i+1}"], modes_zernike[m])
            image_ref = next((p for p in images_mode if extraire_suffixe(p) == "ref" + modes_zernike[m]), "")
            dic_image_paires[f"{i+1}"][m] = creer_images_paires(images_mode, image_ref)[1:]
    return dic_image_paires

def precompute_profiles(paires_mode, max_radius):
    precomputed_profiles = {}
    all_images = {get_image_key(img): img for pair in paires_mode for img in pair}
    for key, image_path in all_images.items():
        image = imageio.imread(image_path) if isinstance(image_path, str) else image_path
        image_np = np.array(image, dtype=float)
        radius_bins, radial_mean, pixel_counts = calc_rotational_average(image_np, max_radius)
        max_val = np.max(radial_mean)
        if max_val != 0:
            radial_mean /= max_val
        precomputed_profiles[key] = (radius_bins, radial_mean, pixel_counts)
    return precomputed_profiles

def calcul_aires_diff_general(paires_mode, metric_func, *, profiles=None, r_min=None, r_max=None, **kwargs):
    """
    Calcule la différence de métrique entre chaque paire (avec, sans), selon une fonction métrique.

    Args:
        paires_mode : liste de paires d’images [(with, without)]
        metric_func : fonction prenant une image et retournant un scalaire ou un profil
        profiles : (optionnel) dictionnaire de profils précalculés (utile pour bandpass)
        r_min, r_max : (optionnels) limites radiales pour la métrique bandpass
        kwargs : autres arguments passés à metric_func

    Returns:
        np.ndarray : tableau des différences de métrique
    """
    aires = []
    for img_w, img_wo in paires_mode:
        if profiles is not None:
            key_w, key_wo = get_image_key(img_w), get_image_key(img_wo)
            bins, radial_w, pixels_w = profiles[key_w]
            _, radial_wo, _ = profiles[key_wo]
            mask = (bins >= r_min) & (bins <= r_max)
            if np.sum(mask) == 0 or np.sum(pixels_w[mask]) == 0:
                aire = np.nan
            else:
                diff = np.abs(radial_w[mask] - radial_wo[mask])
                aire = np.sum(diff * pixels_w[mask]) / np.sum(pixels_w[mask])
        else:
            # Chargement si besoin
            img_w = imageio.imread(img_w) if isinstance(img_w, str) else img_w
            img_wo = imageio.imread(img_wo) if isinstance(img_wo, str) else img_wo

            val_w = metric_func(img_w, **kwargs)
            val_wo = metric_func(img_wo, **kwargs)
            aire = abs(val_w - val_wo)
        aires.append(aire)
    return np.array(aires)

def get_input_a(profondeurs, quantites, jeux, image_metriques):
    choix_profondeur = choisir_element_liste("Profondeur", "Sélectionnez une profondeur :", profondeurs)
    choix_image_metrique = choisir_element_liste("Métrique", "Sélectionnez une métrique :", image_metriques)
    choix_quantite = choisir_element_liste("Quantité", "Sélectionnez une quantité :", quantites)
    choix_jeu = choisir_element_liste("Jeu", "Sélectionnez un jeu :", jeux)
    choix_metrique_value = choisir_element_liste("Métrique", "Sélectionnez une métrique :", image_metriques)
    return choix_profondeur, choix_image_metrique, choix_quantite, choix_jeu, choix_metrique_value

def get_input_N(profondeurs, quantites, jeux, image_metriques):
    choix_profondeur = choisir_element_liste("Profondeur", "Sélectionnez une profondeur :", profondeurs)
    choix_image_metrique = choisir_element_liste("Métrique", "Sélectionnez une métrique :", image_metriques)
    choix_quantite = choisir_element_liste("Quantité", "Sélectionnez une quantité :", quantites)
    choix_metrique_value = choisir_element_liste("Métrique", "Sélectionnez une métrique :", image_metriques)
    return choix_profondeur, choix_image_metrique, choix_quantite, choix_metrique_value

def fit_a_from_differences(aires_diff, show=False, titre=None):
    if np.isnan(aires_diff).any() or len(aires_diff) < 3:
        return np.nan
    x = np.arange(-3, len(aires_diff) - 3)
    coeffs = np.polyfit(x, aires_diff, deg=2)
    if show:
        plt.figure(figsize=(6, 4))
        plt.plot(x, aires_diff, 'o-', label='aires_diff')
        plt.plot(x, np.polyval(coeffs, x), '--', label=f'fit quad. (a={coeffs[0]:.2e})')
        plt.title(titre or "Fit")
        plt.xlabel("Quantity of aberration")
        plt.ylabel("Weighted difference")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return coeffs[0]

def calculer_aires_diff_par_jeu_general(dic_image_paires, mode, metrique_func, profiles_required=False, **kwargs):
    """
    Calcule les aires de différence pour un mode donné, pour tous les jeux disponibles,
    en utilisant une fonction métrique donnée.

    Args:
        dic_image_paires : dict[str, dict[str, list]] — jeux → mode → paires
        mode : nom du mode à traiter
        metrique_func : fonction métrique à appliquer (ex: imax, this_realm, etc.)
        profiles_required : bool — True si la métrique nécessite un profil radial (bandpass)
        kwargs : paramètres additionnels (r_min, r_max, etc.)

    Returns:
        np.ndarray : tableau (nb_modes × nb_jeux)
    """
    list_aires_diff = []

    for i in range(len(dic_image_paires)):
        paires_mode = dic_image_paires[f"{i+1}"][mode]

        if profiles_required:
            max_radius = min(min(imageio.imread(p[0]).shape) // 2 for p in paires_mode)
            profiles = precompute_profiles(paires_mode, max_radius)
            aires = calcul_aires_diff_general(paires_mode, metric_func=None, profiles=profiles, **kwargs)
        else:
            aires = calcul_aires_diff_general(paires_mode, metric_func=metrique_func, **kwargs)

        list_aires_diff.append(aires)

    return np.array(list_aires_diff).T

def fit_courbes_et_N(x, y_moy, y_min, y_max, show=False, titre=None):
    coeffs_moy = np.polyfit(x, y_moy, 2)
    coeffs_min = np.polyfit(x, y_min, 2)
    coeffs_max = np.polyfit(x, y_max, 2)
    y_fit_moy = np.polyval(coeffs_moy, x)
    y_fit_min = np.polyval(coeffs_min, x)
    y_fit_max = np.polyval(coeffs_max, x)
    r2 = r2_score(y_fit_moy, y_moy)
    if show:
        plt.figure(figsize=(6, 4))
        plt.plot(x, y_moy, 'o-', label='moy')
        plt.plot(x, y_fit_moy, '--', label='fit moy')
        plt.plot(x, y_fit_min, '--', label='fit min')
        plt.plot(x, y_fit_max, '--', label='fit max')
        plt.text(0.05, 0.88, f"$r^2$  = {r2:.4f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')
        plt.title(titre or "Fit")
        plt.xlabel("Quantity of aberrations")
        plt.ylabel("Weighted difference")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    a = coeffs_moy[0]
    r2 = r2_score(y_moy, y_fit_moy)
    N1 = np.trapezoid(np.abs(np.array(y_max) - np.array(y_min)), x)
    N2 = np.trapezoid(np.abs(y_fit_max - y_fit_min), x)
    return a, N1, N2, r2

def enregistrer_excel(df, chemin_fichier):
    if os.path.exists(chemin_fichier):
        df_exist = pd.read_excel(chemin_fichier)
        df = pd.concat([df_exist, df], ignore_index=True)
    df.to_excel(chemin_fichier, index=False)

def extraire_courbes_moy_min_max(*courbes):
    moy = [np.mean(c) for c in courbes]
    std = [np.std(c) for c in courbes]
    return (
        [m for m in moy],
        [m - s for m, s in zip(moy, std)],
        [m + s for m, s in zip(moy, std)],
    )

#################### Calcul a et N ########################

def calcul_du_a_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, choix_jeu, choix_profondeur)
    image_paires = construire_image_paires(image_paths, modes, modes_zernike)
    dic_a = {}
    print(image_paths)
    # Choix des paramètres selon la métrique
    if metrique_value == 'bandpass':
        r_min, r_max = bp[choix_profondeur][choix_image_metrique]
        largeur_cible = r_max - r_min
        centre_cible = r_min + largeur_cible / 2
        metric_func = None  # géré via profiles
        profiles_required = True
        metric_kwargs = {'r_min': r_min, 'r_max': r_max}

    elif metrique_value == 'imax':
        metric_func = imax
        profiles_required = False
        metric_kwargs = {}

    elif metrique_value == 'realm':
        metric_func = this_realm
        profiles_required = False
        metric_kwargs = {}

    else:
        raise ValueError(f"Métrique inconnue : {metrique_value}")

    for m in modes:
        print(m)
        paires_mode = image_paires[m]

        if profiles_required:
            max_radius = min(min(imageio.imread(p[0]).shape) // 2 for p in paires_mode)
            profiles = precompute_profiles(paires_mode, max_radius)
            aires = calcul_aires_diff_general(
                paires_mode,
                metric_func=None,
                profiles=profiles,
                **metric_kwargs
            )
        else:
            aires = calcul_aires_diff_general(
                paires_mode,
                metric_func=metric_func,
                **metric_kwargs
            )

        # Titre affiché
        if metrique_value == 'bandpass':
            titre = f"{m} | center={centre_cible}, width={largeur_cible}"
        else:
            titre = f"{m}"

        dic_a[m] = fit_a_from_differences(aires, Show, titre=titre)

    ligne_resultat = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
        "Jeu": choix_jeu
    }
    ligne_resultat.update(dic_a)
    df = pd.DataFrame([ligne_resultat])

    chemin = fr"C:\Users\rlusson\Desktop\resultats\a_ligne.xlsx"
    enregistrer_excel(df, chemin)

    print(choix_profondeur, choix_image_metrique, choix_quantite, choix_jeu)

def calcul_N_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    # Nombre de points (cas particulier)
    if choix_profondeur == "Profondeur" and choix_quantite == "No_aber" and choix_image_metrique != "Dapi" and choix_image_metrique not in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]:
        n = 7
    elif (
        choix_profondeur == "Profondeur"
        and choix_quantite == "No_aber"
        and choix_image_metrique in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]
    ):
        n = 3
    else:
        n = 5   

    # Importation des images
    dic_image_paths = importation_chemin(choix_image_metrique, choix_quantite, "Tous", choix_profondeur)
    dic_image_paires = construire_image_paires_par_jeu(dic_image_paths, modes, modes_zernike, n)
    print(dic_image_paths)
    # Configuration selon la métrique
    if metrique_value == "bandpass":
        r_min, r_max = bp[choix_profondeur][choix_image_metrique]
        profiles_required = True
        metric_func = None
        metric_kwargs = {"r_min": r_min, "r_max": r_max}
        titre_suffix = f"| center={r_min + (r_max - r_min)/2}, width={r_max - r_min}"

    elif metrique_value == "imax":
        profiles_required = False
        metric_func = imax
        metric_kwargs = {}
        titre_suffix = ""

    elif metrique_value == "realm":
        profiles_required = False
        metric_func = this_realm
        metric_kwargs = {}
        titre_suffix = ""

    else:
        raise ValueError(f"Métrique inconnue : {metrique_value}")

    # Initialisation des dictionnaires résultats
    dic_a_moy, dic_N1, dic_N2 = {}, {}, {}

    # Boucle sur les modes
    for m in modes:
        print(m)
        aires_diff_sets = calculer_aires_diff_par_jeu_general(
            dic_image_paires,
            m,
            metrique_func=metric_func,
            profiles_required=profiles_required,
            **metric_kwargs
        )
        aires_diff_moy = np.mean(aires_diff_sets, axis=1)
        aires_diff_min = aires_diff_moy - np.std(aires_diff_sets, axis=1)
        aires_diff_max = aires_diff_moy + np.std(aires_diff_sets, axis=1)
        x = np.arange(-3, len(aires_diff_moy) - 3)
        titre = f"{m} {titre_suffix}"
        a, N1, N2 = fit_courbes_et_N(x, aires_diff_moy, aires_diff_min, aires_diff_max, Show, titre=titre)
        dic_a_moy[m], dic_N1[m], dic_N2[m] = a, N1, N2

    # Construction et export des DataFrames
    print(choix_profondeur, choix_image_metrique, choix_quantite)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
    }
    df1 = pd.DataFrame([{**base_infos, **dic_N1}])
    df2 = pd.DataFrame([{**base_infos, **dic_N2}])

    enregistrer_excel(df1, fr"C:\Users\rlusson\Desktop\resultats\N_points_ligne.xlsx")
    enregistrer_excel(df2, fr"C:\Users\rlusson\Desktop\resultats\N_fit_ligne.xlsx")

def calcul_du_a_9N_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, choix_jeu, choix_profondeur)
    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    dic_a, dic_r2,dic_beta = {}, {}, {}
    print(image_paths)
    # Configuration des fonctions de métrique
    if metrique_value == "bandpass":
        r_min, r_max = bp[choix_profondeur][choix_image_metrique]
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2, thisNA, thislambda,f_min, f_max)
        

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        

    else:
        raise ValueError(f"Métrique inconnue : {metrique_value}")

    # Traitement par mode
    for m in modes:
        print(m)
        images = sorted(
            recuperer_images_par_mode(image_paths, modes_zernike[m]),
            key=lambda path: extraire_quantite_alpha(os.path.basename(path))
        )
        metrics = metrique_func(*images[1:9])
        x, y = get_fit_array_9N(*metrics)
        a,b , _, _, _, _, r2 = fit((x, y), Show)
        beta_m = b / (2 * a)
        #print(beta_m)
        dic_beta[m] = beta_m
        dic_a[m], dic_r2[m] = -a, r2

    print(choix_profondeur, choix_image_metrique, choix_quantite, choix_jeu)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
        "Jeu": choix_jeu
    }
    df1 = pd.DataFrame([{**base_infos, **dic_a}])
    df2 = pd.DataFrame([{**base_infos, **dic_r2}])
    #df3 = pd.DataFrame([{**base_infos, **dic_beta}])
    enregistrer_excel(df1, fr"C:\Users\rlusson\Desktop\resultats\a9N_ligne.xlsx")
    enregistrer_excel(df2, fr"C:\Users\rlusson\Desktop\resultats\r2_ligne.xlsx")
    #enregistrer_excel(df3, fr"C:\Users\rlusson\Desktop\resultats\beta_ligne.xlsx")

def calcul_du_a_9N_norm_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, choix_jeu, choix_profondeur)
    
    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    dic_a, dic_r2,dic_beta = {}, {}, {}
    print(image_paths)
    # Configuration des fonctions de métrique
    if metrique_value == "bandpass":
        r_min, r_max = bp[choix_profondeur][choix_image_metrique]
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2, thisNA, thislambda,f_min, f_max)
        

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        

    else:
        raise ValueError(f"Métrique inconnue : {metrique_value}")

    # Traitement par mode
    for m in modes:
        print(m)
        images = sorted(
            recuperer_images_par_mode(image_paths, modes_zernike[m]),
            key=lambda path: extraire_quantite_alpha(os.path.basename(path))
        )
        metrics = metrique_func(*images[1:9])
        x, y = get_fit_array_9N(*metrics)
        y_norm = y / y.max()
        a,b , _, _, _, _, r2 = fit((x, y_norm), Show)
        beta_m = b / (2 * a)
        print(beta_m)
        dic_beta[m] = beta_m
        dic_a[m], dic_r2[m] = -a, r2

    print(choix_profondeur, choix_image_metrique, choix_quantite, choix_jeu)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
        "Jeu": choix_jeu
    }
    df1 = pd.DataFrame([{**base_infos, **dic_a}])
    df2 = pd.DataFrame([{**base_infos, **dic_r2}])
    #df3 = pd.DataFrame([{**base_infos, **dic_beta}])
    enregistrer_excel(df1, fr"C:\Users\rlusson\Desktop\resultats\a9N_norm_ligne.xlsx")
    enregistrer_excel(df2, fr"C:\Users\rlusson\Desktop\resultats\r2_norm_ligne.xlsx")
    #enregistrer_excel(df3, fr"C:\Users\rlusson\Desktop\resultats\beta_norm_ligne.xlsx")

def calcul_N_9N_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    if choix_profondeur == "Profondeur" and choix_quantite == "No_aber" and choix_image_metrique != "Dapi" and choix_image_metrique not in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]:
        n = 7
    elif (
        choix_profondeur == "Profondeur"
        and choix_quantite == "No_aber"
        and choix_image_metrique in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]
    ):
        n = 3
    elif choix_profondeur == "Profondeur" and choix_quantite == "High_aber" and choix_image_metrique == "Dapi":
        n=4
    else:
        n = 5    
        
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, "Tous", choix_profondeur)

    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    # Configuration selon la métrique
    if metrique_value == "bandpass":
        r_min, r_max =bp[choix_profondeur][choix_image_metrique]
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2,thisNA, thislambda, f_min, f_max)
        titre_suffix = f"| center={r_min + (r_max - r_min)/2}, width={r_max - r_min}"
        suffix1 = "N_points9N_ligne.xlsx"
        suffix2 = "N_fit9N_ligne.xlsx"

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        titre_suffix = ""

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        titre_suffix = ""

    else:
        raise ValueError(f"Métrique inconnue : {metrique}")

    dic_a_moy, dic_N1, dic_N2 = {}, {}, {}

    # Boucle sur les modes
    for m in modes:
        metrics_all = [[] for _ in range(8)]

        for i in range(n):
            print(m)
            images = sorted(
                recuperer_images_par_mode(image_paths[f"{i+1}"], modes_zernike[m]),
                key=lambda path: extraire_quantite_alpha(os.path.basename(path))
            )
            metriques = metrique_func(*images[1:9])
            for j in range(8):
                metrics_all[j].append(metriques[j])

        courbes = [np.array(c) for c in metrics_all]
        y_moy, y_min, y_max = extraire_courbes_moy_min_max(*courbes)
        x = get_fit_array_9N(*y_moy)[0]
        titre = f"{m} {titre_suffix}"
        a, N1, N2 = fit_courbes_et_N(x, y_moy, y_min, y_max, Show, titre=titre)
        dic_a_moy[m], dic_N1[m], dic_N2[m] = a, N1, N2

    print(choix_profondeur, choix_image_metrique, choix_quantite)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
    }
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_N1}]), fr"C:\Users\rlusson\Desktop\resultats\N_points9N_ligne.xlsx")
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_N2}]), fr"C:\Users\rlusson\Desktop\resultats\N_fit9N_ligne.xlsx")

def calcul_N_9N_norm1_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, "Tous", choix_profondeur)
    
    if choix_profondeur == "Profondeur" and choix_quantite == "No_aber" and choix_image_metrique != "Dapi" and choix_image_metrique not in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames","std_10frames", "std_30frames", "std_50frames", "std_100frames"]:
        n = 7
    elif (
        choix_profondeur == "Profondeur"
        and choix_quantite == "No_aber"
        and choix_image_metrique in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames","std_10frames", "std_30frames", "std_50frames", "std_100frames"]
    ):
        n = 3
    elif choix_profondeur == "Profondeur" and choix_quantite == "High_aber" and choix_image_metrique == "Dapi":
        n=4
    else:
        n = 5    

    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    # Configuration selon la métrique
    if metrique_value == "bandpass":
        r_min, r_max = bp[choix_profondeur][choix_image_metrique] #changer bp en fonction du filtre que l'on veut
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2,thisNA, thislambda, f_min, f_max)
        titre_suffix = f"| center={r_min + (r_max - r_min)/2}, width={r_max - r_min}"
        suffix1 = "N_points9N_ligne.xlsx"
        suffix2 = "N_fit9N_ligne.xlsx"

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        titre_suffix = ""

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        titre_suffix = ""

    else:
        raise ValueError(f"Métrique inconnue : {metrique}")

    dic_a_moy, dic_N1, dic_N2,dic_r2 = {}, {}, {}, {}

    # Boucle sur les modes
    for m in modes:
        metrics_all = [[] for _ in range(8)]

        for i in range(n):
            #print(m)
            images = sorted(
                recuperer_images_par_mode(image_paths[f"{i+1}"], modes_zernike[m]),
                key=lambda path: extraire_quantite_alpha(os.path.basename(path))
            )
            metriques = metrique_func(*images[1:9])
            for j in range(8):
                metrics_all[j].append(metriques[j])
        metrics_all = np.array(metrics_all)
        metrics_all_norm = metrics_all / metrics_all.max()
        courbes = [np.array(c) for c in metrics_all_norm]
        y_moy, y_min, y_max = extraire_courbes_moy_min_max(*courbes)
        x = get_fit_array_9N(*y_moy)[0]
        titre = f"{m} {titre_suffix}"
        a, N1, N2, r2 = fit_courbes_et_N(x, y_moy, y_min, y_max, Show, titre=titre)
        dic_a_moy[m], dic_N1[m], dic_N2[m], dic_r2[m] = a, N1, N2, r2

    print(choix_profondeur, choix_image_metrique, choix_quantite)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
    }
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_N1}]), fr"C:\Users\rlusson\Desktop\resultats\N_points9N_norm1_ligne.xlsx")
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_a_moy}]), fr"C:\Users\rlusson\Desktop\resultats\a_9N_norm_ligne.xlsx")
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_r2}]), fr"C:\Users\rlusson\Desktop\resultats\r2_9N_norm_ligne.xlsx")
    

def calcul_N_9N_norm2_general(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    if choix_profondeur == "Profondeur" and choix_quantite == "No_aber" and choix_image_metrique != "Dapi" and choix_image_metrique not in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]:
        n = 7
    elif (
        choix_profondeur == "Profondeur"
        and choix_quantite == "No_aber"
        and choix_image_metrique in ["mip_10frames", "mip_30frames", "mip_50frames", "mip_100frames"]
    ):
        n = 3
    else:
        n = 5    
        
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, "Tous", choix_profondeur)

    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    # Configuration selon la métrique
    if metrique_value == "bandpass":
        r_min, r_max =bp[choix_profondeur][choix_image_metrique]
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2,thisNA, thislambda, f_min, f_max)
        titre_suffix = f"| center={r_min + (r_max - r_min)/2}, width={r_max - r_min}"
        suffix1 = "N_points9N_ligne.xlsx"
        suffix2 = "N_fit9N_ligne.xlsx"

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        titre_suffix = ""

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        titre_suffix = ""

    else:
        raise ValueError(f"Métrique inconnue : {metrique}")

    dic_a_moy, dic_N1, dic_N2 = {}, {}, {}

    # Boucle sur les modes
    for m in modes:
        metrics_all = [[] for _ in range(8)]

        for i in range(n):
            print(m)
            images = sorted(
                recuperer_images_par_mode(image_paths[f"{i+1}"], modes_zernike[m]),
                key=lambda path: extraire_quantite_alpha(os.path.basename(path))
            )
            metriques = metrique_func(*images[1:9])
            for j in range(8):
                metrics_all[j].append(metriques[j])

        metrics_all = np.array(metrics_all)
        metrics_all_norm = metrics_all / metrics_all.max(axis=1, keepdims=True)
        courbes = [np.array(c) for c in metrics_all_norm]
        y_moy, y_min, y_max = extraire_courbes_moy_min_max(*courbes)
        x = get_fit_array_9N(*y_moy)[0]
        titre = f"{m} {titre_suffix}"
        a, N1, N2 = fit_courbes_et_N(x, y_moy, y_min, y_max, Show, titre=titre)
        dic_a_moy[m], dic_N1[m], dic_N2[m] = a, N1, N2

    print(choix_profondeur, choix_image_metrique, choix_quantite)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
    }
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_N1}]), fr"C:\Users\rlusson\Desktop\resultats\N_points9N_norm2_ligne.xlsx")
    enregistrer_excel(pd.DataFrame([{**base_infos, **dic_N2}]), fr"C:\Users\rlusson\Desktop\resultats\2_ligne.xlsx")


def traiter_lot_images(
    fonctions_a_appliquer,
    profondeurs=None,
    quantites=None,
    jeux=None,
    image_metriques=None,
    metriques_value=None
):
    profondeurs = profondeurs or PROFONDEURS
    quantites = quantites or QUANTITES
    jeux = jeux or JEUX
    image_metriques = image_metriques or IMAGE_METRIQUES
    metriques_value = metriques_value or METRIQUE_VALUE  # valeur par défaut

    if not isinstance(fonctions_a_appliquer, list):
        fonctions_a_appliquer = [fonctions_a_appliquer]

    total = (
        len(fonctions_a_appliquer)
        * len(metriques_value)
        * len(profondeurs)
        * len(quantites)
        * len(image_metriques)
        * len(jeux)
    )
    compteur = 1

    for fonction in fonctions_a_appliquer:
        jeux_a_utiliser = ["Tous"] if getattr(fonction, "requires_all_jeux", False) else jeux
        for image_metrique in image_metriques:
            for metrique_value in metriques_value:
                for profondeur in profondeurs:
                    for quantite in quantites:
                        for jeu in jeux_a_utiliser:
                            try:
                                print(
                                    f"⏳ [{compteur}/{total}] {fonction.__name__} : "
                                    f"OP={metrique_value}, P={profondeur}, Q={quantite}, J={jeu}, M={image_metrique}"
                                )
                                fonction(
                                    image_metrique,
                                    profondeur,
                                    quantite,
                                    jeu,
                                    metrique_value=metrique_value,
                                    Show=False
                                )
                            except Exception as e:
                                print(
                                    f"⚠️ Erreur pour {fonction.__name__} avec "
                                    f"OP={metrique_value}, P={profondeur}, Q={quantite}, J={jeu}, M={image_metrique} : {e}"
                                )
                            compteur += 1

def traiter_lot_projection(
    fonction_projection,
    racine=r"C:/Users/rlusson/Desktop/Data/Carac_AO",
    profondeurs=None,
    quantites=None,
    jeux=None,
    **kwargs
):
    """
    Applique une fonction de projection à tous les sous-dossiers d’un dataset.

    Args:
        fonction_projection (callable): Fonction prenant (input_folder, output_folder, **kwargs)
        **kwargs: Options spécifiques à la projection (ex: mode="max", frame_idx=5...)
    """
    profondeurs = profondeurs or PROFONDEURS
    quantites = quantites or QUANTITES
    jeux = jeux or JEUX

    racine = Path(racine)
    total = len(profondeurs) * len(quantites) * len(jeux)
    compteur = 1

    for profondeur in profondeurs:
        for quantite in quantites:
            for jeu in jeux:
                input_folder = racine / profondeur / quantite / str(jeu)
                output_folder = input_folder / "Frame"

                print(f"📁 [{compteur}/{total}] {fonction_projection.__name__} | {input_folder}")
                try:
                    fonction_projection(
                        input_folder=input_folder,
                        output_folder=output_folder,
                        **kwargs
                    )
                except Exception as e:
                    print(f"⚠️ Erreur sur {input_folder} : {e}")
                compteur += 1

def compter_molecules_dans_bandes(df, filtres):
    """
    Applique des filtres successifs sur les colonnes d’un DataFrame
    et compte le nombre de molécules dans les bandes spécifiées.

    Args:
        df : pd.DataFrame — table contenant les colonnes à filtrer
        filtres : dict — nom_colonne : (borne_min, borne_max)

    Returns:
        total_final : int — nombre de lignes restant après tous les filtres
        resume : dict — résumé du nombre de lignes à chaque étape
    """
    df_filtre = df.copy()
    resume = {}

    for col, (vmin, vmax) in filtres.items():
        if col not in df_filtre.columns:
            print(f"⚠️ Colonne '{col}' non trouvée dans le DataFrame.")
            continue

        avant = len(df_filtre)
        df_filtre = df_filtre[(df_filtre[col] >= vmin) & (df_filtre[col] <= vmax)]
        apres = len(df_filtre)
        resume[col] = {
            "borne_min": vmin,
            "borne_max": vmax,
            "avant_filtrage": avant,
            "apres_filtrage": apres,
            "filtrées": avant - apres
        }

    return len(df_filtre), resume

def calculer_pourcentage_filtrage(df, filtres):
    """
    Calcule le pourcentage de molécules conservées après filtrage.

    Args:
        df : pd.DataFrame — DataFrame complet des localisations
        filtres : dict — nom_colonne : (borne_min, borne_max)

    Returns:
        pourcentage : float — pourcentage de molécules restantes (entre 0 et 100)
        total_avant : int — nombre total de molécules avant filtrage
        total_apres : int — nombre de molécules restantes après filtrage
    """
    total_avant = len(df)
    total_apres, _ = compter_molecules_dans_bandes(df, filtres)

    pourcentage = (total_apres / total_avant) * 100 if total_avant > 0 else 0.0
    return pourcentage, total_avant, total_apres

def calcul_ral(choix_image_metrique,choix_profondeur,  choix_quantite, choix_jeu, metrique_value='bandpass',Show=True):
    image_paths = importation_chemin(choix_image_metrique, choix_quantite, choix_jeu, choix_profondeur)
    sum=0
    dic_beta = {}

    if choix_profondeur == "Coverslip":
        thisNA = 1.49
    else:
        thisNA = 1.27

    if choix_image_metrique == "Dapi":
        thislambda = 0.465
    else:
        thislambda = 0.589

    if choix_quantite == "Low_aber":
        aber_res = {"Astigmatisme 0°": 0.007,"Coma 0°":0.006,"Aberration Sphérique":0.025,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}
    if choix_quantite == "Medium_aber":
        aber_res = {"Astigmatisme 0°":0.017,"Coma 0°":0.016,"Aberration Sphérique":0.045,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}
    if choix_quantite == "High_aber":
        aber_res = {"Astigmatisme 0°":0.055,"Coma 0°":0.024,"Aberration Sphérique":0.079,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}
    if choix_quantite == "High2":
        aber_res = {"Astigmatisme 0°":0.080,"Coma 0°":0.050,"Aberration Sphérique":0.117,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}

    print(image_paths)
    # Configuration des fonctions de métrique
    if metrique_value == "bandpass":
        r_min, r_max = bp[choix_profondeur][choix_image_metrique]
        f_min, f_max = pix_to_freq(r_min), pix_to_freq(r_max)
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_bandpass2,thisNA, thislambda, f_min, f_max)
        

    elif metrique_value == "imax":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, imax)
        

    elif metrique_value == "realm":
        metrique_func = lambda *imgs: generic_metric_9N(imgs, this_realm,thisNA, thislambda*1000)
        

    else:
        raise ValueError(f"Métrique inconnue : {metrique_value}")

    # Traitement par mode
    for m in modes:
        print(m)
        images = sorted(
            recuperer_images_par_mode(image_paths, modes_zernike[m]),
            key=lambda path: extraire_quantite_alpha(os.path.basename(path))
        )
        metrics = metrique_func(*images[1:9])
        x, y = get_fit_array_9N(*metrics)
        a, b, c, _, _, _, r2 = fit((x, y), Show)
        beta_m = b / (2 * a)
        dic_beta[m] = beta_m
        #print(beta_m)
        sum = sum + (dic_beta[m]- aber_res[m])**2
        ral = np.sqrt(sum)
        Rs = np.exp(-4*np.pi**2*(ral**2)/(thislambda)**2)

    print(choix_profondeur, choix_image_metrique, choix_quantite, choix_jeu)
    base_infos = {
        "Metric value" : metrique_value,    
        "Image Métrique": choix_image_metrique,
        "Profondeur" : choix_profondeur,
        "Quantité": choix_quantite,
        "Jeu": choix_jeu
    }
    df1 = pd.DataFrame([{**base_infos, **{"ral":ral}}])
    df2 = pd.DataFrame([{**base_infos, **{"Rs":Rs}}])
    enregistrer_excel(df1, fr"C:\Users\rlusson\Desktop\resultats\ral.xlsx")
    enregistrer_excel(df2, fr"C:\Users\rlusson\Desktop\resultats\Rs.xlsx")

#%% zero padding

# Paramètres
root_input = r"C:\Users\rlusson\Desktop\Data\Carac_AO"
root_output = r"C:\Users\rlusson\Desktop\Test_padding"
target_shape = (256, 256)

# Lancement du traitement
for profondeur in ["Coverslip", "Profondeur"]:
    for quantite in ["No_aber", "Low_aber", "Medium_aber", "High_aber", "High2"]:
        for jeu in range(1, 8):  # jeu de 1 à 7
            for metrique in ["avg_psf"]:
                input_dir = os.path.join(root_input, profondeur, quantite, str(jeu), metrique)
                output_dir = os.path.join(root_output, profondeur, quantite, str(jeu), metrique)

                if not os.path.exists(input_dir):
                    continue

                for f in os.listdir(input_dir):
                    if not f.lower().endswith((".tif", ".tiff")):
                        continue
                    input_path = os.path.join(input_dir, f)
                    try:
                        img = imageio.imread(input_path)
                        padded = {
                            "zero_pad": zero_padding(img, target_shape),
                            "random_pad": random_noise_padding(img, target_shape),
                            "spline": spline_interpolation(img, target_shape)
                        }
                        output_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}_compare.png")
                        show_comparison(img, padded, output_path)
                        print(f"✅ Sauvé : {output_path}")
                    except Exception as e:
                        print(f"⚠️ Erreur avec {input_path}: {e}")

#%% MIP plusieurs frames 

input_folder = r"C:\Users\rlusson\Desktop\Data\Carac_AO\MIP_100f\3"
base_output_folder = r"C:\Users\rlusson\Desktop\Data\Carac_AO\MIP_100f\3"
n_frames_list = [10, 30, 50, 100]  # différentes profondeurs de MIP

for n in n_frames_list:
    output_folder = os.path.join(base_output_folder, f"std_{n}frames")
    projection_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        mode="std",        # MIP
        frame_idx=None,
        max_z=n            # nombre de frames utilisées pour la projection
    )



#%%Visualisation des fréquences

input_root = r"C:\Users\rlusson\Desktop\Data\Carac_AO\Profondeur\No_aber\1\std_10frames\Image0006_refZer4.tif"
output_root = r"C:\Users\rlusson\Desktop\testfc.tif"

apply_bandpass_filter(input_root,output_root,(4500,pix_to_freq(255,0.108,256)),True,True)

#%% Création de toutes les images pour visualiser les freq

input_root = r"C:\Users\rlusson\Desktop\Data\Carac_AO"
output_root = r"C:\Users\rlusson\Desktop\Visualisation_frequence"

types = ["Coverslip", "Profondeur"]
niveaux = ["No_aber", "Low_aber", "Medium_aber", "High_aber","High2"]
jeux = ["1", "2", "3","4","5","6","7"]  # À adapter selon ton contenu réel
metriques = ["MIP", "SUM", "Frame","Dapi","Gallery","STD"]  # Ou autre liste de sous-dossiers
filtres = ["bandpass", "lowpass", "highpass"]

for e in types:
    for i in niveaux:
        for j in jeux:
            for k in metriques:
                folder = os.path.join(input_root, e, i, j, k)
                print(f"🟡 Checking folder: {folder}") 
                if not os.path.isdir(folder):
                    print(f"🔴 Folder does not exist: {folder}")
                    continue

                for image_name in os.listdir(folder):
                    if not image_name.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                        continue

                    input_path = os.path.join(folder, image_name)

                    # Chemin de sortie
                    base_name = os.path.splitext(image_name)[0]
                    output_folder = os.path.join(output_root, e, i, j, k, base_name)
                    os.makedirs(output_folder, exist_ok=True)

                    for filtre in filtres:
                        output_name = f"image_{filtre}.tif"
                        output_path = os.path.join(output_folder, output_name)
                        lo = pix_to_freq(bp[e][k][0])
                        hi = pix_to_freq(bp[e][k][1])

                        # ➤ À TOI DE REMPLIR la bande selon (e, k)
                        #if filtre == "bandpass":
                            #band =  (lo,hi) # une fonction que tu définis
                        #elif filtre == "lowpass":
                            #band = (0.0, lo)
                        if filtre == "highpass":
                            if hi < fc[e][k]:
                                band = (hi, fc[e][k])
                            else:
                                band = (fc[e][k], hi)
                        else:
                            continue

                        apply_bandpass_filter(input_path, output_path, band,True,False)




#%%  1 mode de Zernike 

choix_image_metrique = choisir_metrique_liste(metriques)
choix_quantite = choisir_quantite_liste(quantites)
choix_jeu = choisir_jeu_liste(jeux)
choix_profondeur = choisir_profondeur_liste(profondeurs)

image_paths = importation_chemin(choix_image_metrique,choix_quantite,choix_jeu,choix_profondeur)



#Récupération des images 

choix_nom = choisir_mode_zernike_liste(modes)  # ex: "Astigmatisme 0°"
choix = modes_zernike[choix_nom]               # ex: "Zer4"

images_zernike = recuperer_images_par_mode(image_paths, choix) #récupère toutes les images concernées par le mode choisit
print(images_zernike)
quantites_cibles = {"-3alpha"+choix : "ym3", "-2alpha"+choix : "ym2", "-1alpha"+choix : "ym1", "ref"+choix : "image_ref1", "+1alpha"+choix : "yp1", "+2alpha"+choix : "yp2", "+3alpha"+choix : "yp3", "+4alpha"+choix : "yp4"} #crée un dictionnaire avec les noms de fichiers que l'on cible et lui associe un nom
images_selectionnees1 = {}

for path in image_paths:                                        #récupère les images que l'on souhaite et leur associe un nom 
    name = extraire_suffixe(path)
    if name in quantites_cibles:
        images_selectionnees1[quantites_cibles[name]] = path

# # Lecture et FFT rotationnelle

img1 = np.array(imageio.imread(images_selectionnees1["yp4"]), dtype=float)
img2 = np.array(imageio.imread(images_selectionnees1["image_ref1"]), dtype=float)
print(img1.shape)
max_radius = min(img1.shape) // 2
bins, fft1, _ = calc_rotational_average(img1, max_radius, apply_smoothing=True)
_, fft2, _ = calc_rotational_average(img2, max_radius, apply_smoothing=True)
fft1 = fft1 / np.max(fft1) if np.max(fft1) != 0 else fft1
fft2 = fft2 / np.max(fft2) if np.max(fft2) != 0 else fft2

diff_curve = np.abs(fft1 - fft2)
smoothed_diff = gaussian_filter1d(diff_curve, sigma=3)

# === Optimisation automatique de k ===


k = 1  # ← tu peux changer cette valeur

r_min, r_max, width = detect_adaptive_band(bins, smoothed_diff, k=k)
area = compute_integral_in_band(r_min, r_max, bins, smoothed_diff)

print(f"\nDétection adaptative avec k = {k}")
if r_min is not None:
    print(f"✅ Bande détectée de {r_min:.1f} à {r_max:.1f} px (largeur {width:.1f} px)")
    print(f"➡ Aire sous la courbe dans cette bande : {area:.2f}")
else:
    print("❌ Aucune bande détectée avec ce k")

# === Affichage graphique ===

plt.figure(figsize=(10, 6))
plt.plot(bins, fft1, label="Image avec aberrations")
plt.plot(bins, fft2, label="Image sans aberrations")
plt.title("Comparaison des moyennes rotationnelles normalisées")
plt.xlabel("Rayon (pixels)")
plt.ylabel("Amplitude normalisée")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(bins, np.abs(fft1 - fft2), label='|Δ FFT|')
plt.title("Différence absolue entre FFT rotationnelles")
plt.xlabel("Rayon (pixels)")
plt.ylabel("|Δ FFT|")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Lissage et tracé de la différence ===
difference = np.abs(fft1 - fft2)
smoothed_diff = gaussian_filter1d(difference, sigma=3)

plt.figure(figsize=(10, 4))
plt.plot(bins, difference, alpha=0.3, label='|Δ FFT| (non lissé)', color='gray')
plt.plot(bins, smoothed_diff, label='|Δ FFT| (lissé)', color='blue')
plt.title("Différence absolue entre FFT rotationnelles (lissée)")
plt.xlabel("Rayon (pixels)")
plt.ylabel("|Δ FFT|")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(bins, smoothed_diff, label="|Δ FFT| (lissée)", color='blue')
if r_min is not None:
    plt.axvspan(r_min, r_max, alpha=0.3, color='green', label=f'Bande détectée (k={k})')
plt.title("Détection adaptative manuelle")
plt.xlabel("Rayon (pixels)")
plt.ylabel("|Δ FFT|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% plusieurs modes de Zernike 


choix_profondeur = choisir_profondeur_liste(profondeurs)
#choix_profondeur = "Coverslip"
#choix_image_metrique = choisir_metrique_liste(metriques)
choix_image_metrique = "avg_psf_zp"
choix_quantite = choisir_quantite_liste(quantites)
choix_jeu = choisir_jeu_liste(jeux)


image_paths = importation_chemin(choix_image_metrique,choix_quantite,choix_jeu,choix_profondeur)



# Récupération des images 
images_selectionnees = {}


for path in image_paths:                                        #Récupère les images de chaque mode avec la quantité maximale (en positif) d'aberrations
    name = extraire_suffixe(path)
    if name in modes_cibles: 
        images_selectionnees[modes_cibles[name]] = path

print(images_selectionnees)

k = 1

image_pairs = [
    ("astig4", images_selectionnees["astig4"], images_selectionnees["image_ref"]),
    ("astig5", images_selectionnees["astig5"], images_selectionnees["image_ref"]),
    ("coma6", images_selectionnees["coma6"], images_selectionnees["image_ref"]),
    ("coma7", images_selectionnees["coma7"], images_selectionnees["image_ref"]),
    ("spherique8", images_selectionnees["spherique8"], images_selectionnees["image_ref"]),
    ("trefoil9", images_selectionnees["trefoil9"], images_selectionnees["image_ref"]),
    ("trefoil10", images_selectionnees["trefoil10"], images_selectionnees["image_ref"]),
]

#Calcul des bandes

band_limits = []
csv_data = []

for i, (name, img_path, ref_path) in enumerate(image_pairs):
    img1 = np.array(imageio.imread(img_path), dtype=float)
    img2 = np.array(imageio.imread(ref_path), dtype=float)
    print(img1.shape)
    max_radius = min(img1.shape) // 2
    bins, fft1, _ = calc_rotational_average(img1, max_radius, apply_smoothing=True)
    _, fft2, _ = calc_rotational_average(img2, max_radius, apply_smoothing=True)

    diff_curve = np.abs(fft1 - fft2)
    smoothed_diff = gaussian_filter1d(diff_curve, sigma=3)

    r_min, r_max, width = detect_adaptive_band(bins, smoothed_diff, k=k,ignore_first_n=0)
    plt.figure(figsize=(10, 6))
    plt.plot(bins, fft1, label="Image avec aberrations")
    plt.plot(bins, fft2, label="Image sans aberrations")
    plt.title("Comparaison des moyennes rotationnelles normalisées")
    plt.xlabel("Rayon (pixels)")
    plt.ylabel("Amplitude normalisée")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(bins, np.abs(fft1 - fft2), label='|Δ FFT|')
    plt.title("Différence absolue entre FFT rotationnelles")
    plt.xlabel("Rayon (pixels)")
    plt.ylabel("|Δ FFT|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"🧪 Paire {i+1} ➤ largeur = {width:.1f} px", end='')
    if r_min is not None:
        print(f" (de {r_min:.1f} à {r_max:.1f})")
        band_limits.append((r_min, r_max))
        csv_data.append({
        "Nom de la variable": name,
        "Bande min (px)": round(r_min, 1),
        "Bande max (px)": round(r_max, 1),
        "Largeur (px)": round(width, 1)
        })
    else:
        print(" ❌ aucune bande détectée")
print(choix_profondeur,choix_image_metrique,choix_quantite,choix_jeu)
# Calcul de la bande commune (intersection)
if band_limits:
    common_min = max([r[0] for r in band_limits])
    common_max = min([r[1] for r in band_limits])
    if common_min < common_max:
        print(f"\n✅ Bande commune détectée : {common_min:.1f} à {common_max:.1f} px (largeur {(common_max - common_min):.1f} px)")
    else:
        print("\n❌ Aucune intersection entre toutes les bandes")
else:
    print("❌ Aucune bande détectée sur l'ensemble des paires")


##calcul bande englobante union

if choix_image_metrique == "avg_psf":
    n=64
else:
    n=256


if band_limits:
    global_rmin = min(r[0] for r in band_limits)
    global_rmax = max(r[1] for r in band_limits)
    global_fmin =pix_to_freq(global_rmin,0.108,n)
    global_fmax =pix_to_freq(global_rmax,0.108,n)
    #bandpass[choix_profondeur][choix_image_metrique].append([global_fmin,global_fmax])
    print(f"\n📊 Bande englobante : {global_fmin:.1f} à {global_fmax:.1f} mm-1 "
          f"(largeur {(global_fmax - global_fmin):.1f} mm-1)")
    print(f"\n📊 Bande englobante : {global_rmin:.1f} à {global_rmax:.1f} px "
          f"(largeur {(global_rmax - global_rmin):.1f} px)")

    # Affichage superposé des courbes FFT diff
    plt.figure(figsize=(12, 6))
    for name, aberrated_path, _ in image_pairs:
        img1 = np.array(imageio.imread(aberrated_path), dtype=float)
        img2 = np.array(imageio.imread(images_selectionnees["image_ref"]), dtype=float)

        max_radius = min(img1.shape) // 2
        bins, fft1,_ = calc_rotational_average(img1, max_radius, apply_smoothing=True)
        _, fft2,_ = calc_rotational_average(img2, max_radius, apply_smoothing=True)

        diff_curve = np.abs(fft1 - fft2)
        smoothed_diff = gaussian_filter1d(diff_curve, sigma=3)
        #smoothed_diff /= np.max(smoothed_diff) #normalisation
        plt.plot(bins, smoothed_diff, label=name)

    plt.axvspan(global_rmin, global_rmax, color='black', alpha=0.2, label="Bande englobante")
    plt.xlabel("Rayon (pixels)")
    plt.ylabel("|Δ FFT|")
    plt.title("Superposition des profils ΔFFT et bande commune")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("❌ Aucune bande détectée pour affichage global.")

#df = pd.DataFrame(csv_data)
#csv_path ="C:/Users/rlusson\Desktop/Carac_AO/Coverslip/No_aber/1/MIP"
#df.to_csv(csv_path, index=False)

#print(f"\n✅ Résultats sauvegardés dans : {csv_path}")


#%% Traitement de plusieurs jeux

calcul_N_9N_norm1_general.requires_all_jeux = True
traiter_lot_images([calcul_N_9N_norm1_general],["Profondeur"],["No_aber"],["1","2","3"],["mip_10frames","mip_30frames","mip_50frames","mip_100frames"],["bandpass"])


#%% Calcul du a pour un jeu
#profondeur, image_metrique, quantite, jeu, metrique_value=get_input_a()

calcul_du_a_general("MIP","Coverslip", "No_aber", "1", "bandpass")



#%% Calcul sensibilité 

calcul_N_9N_norm1_general.requires_all_jeux = True


calcul_N_9N_norm1_general("Gallery","Profondeur", "High_aber", "1", "bandpass")

#%% Calcul de a 9N
#profondeur, image_metrique, quantite, jeu, metrique_value=get_input_a()

calcul_du_a_9N_general("MIP","Profondeur",  "No_aber", "1", "bandpass")


#%% Calcul de la sensibilité 9N

calcul_N_9N_general.requires_all_jeux = True

#profondeur, image_metrique, quantite, metrique_value=get_input_N()
calcul_N_9N_general("mip_100frames","Profondeur",  "No_aber", "bandpass")

#%% Residual aberrations level

choix_image_metrique = choisir_metrique_liste(metriques)
choix_quantite = choisir_quantite_liste(quantites)
choix_jeu = choisir_jeu_liste(jeux)


sum = 0
ral = 0
Rs = 0
if choix_quantite == "Low_aber":
    aber_res = {"Astigmatisme 0°": 0.007,"Coma 0°":0.006,"Aberration Sphérique":0.025,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}
if choix_quantite == "Medium_aber":
    aber_res = {"Astigmatisme 0°":0.017,"Coma 0°":0.016,"Aberration Sphérique":0.045,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}
if choix_quantite == "High_aber":
    aber_res = {"Astigmatisme 0°":0.055,"Coma 0°":0.24,"Aberration Sphérique":0.79,"Astigmatisme 45°":0,"Coma 90°":0,"Trefoil 90°":0,"Trefoil 0°":0}


image_paths = importation_chemin(choix_image_metrique,choix_quantite,choix_jeu)

r_min = bp[choix_image_metrique][0]
r_max = bp[choix_image_metrique][1]

f_min = pix_to_freq(r_min)
f_max = pix_to_freq(r_max)

image_paths_par_mode = {}
dic_beta={}

for m in modes:
    print(m)
    # Récupération des chemins pour le mode m
    image_paths_par_mode[m] = recuperer_images_par_mode(image_paths, modes_zernike[m])

    # Tri des images par quantité d’aberration alpha
    image_paths_par_mode[m] = sorted(
        image_paths_par_mode[m],
        key=lambda path: extraire_quantite_alpha(os.path.basename(path))
    )


    # Application des métriques sur les 8 images attendues
    ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4 = bandpass_metric_9N(
         image_paths_par_mode[m][1],
        image_paths_par_mode[m][2], image_paths_par_mode[m][3],
        image_paths_par_mode[m][4], image_paths_par_mode[m][5],
        image_paths_par_mode[m][6], image_paths_par_mode[m][7],
         image_paths_par_mode[m][8],
        f_min, f_max
    )


    # Extraction des points de fit
    x, y = get_fit_array_9N(ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4)


    # Fit quadratique
    #a, b, c, xl, yl, R2 = fit_gaussien(x, y,estimer_p0_gaussienne(x, y))
    a, b, c, xl, yl, R2 = fit((x, y),True)

    dic_beta[m] = b/(2*a)
    sum = sum + (dic_beta[m]- aber_res[m])**2
    print(dic_beta[m])
 
ral = np.sqrt(sum)
Rs = np.exp(-4*np.pi**2*(ral**2)/(0.568)**2) #attention dapi 461nm

print(ral,Rs)

#%%Histogrammes

fichier = r"C:\Users\rlusson\Desktop\Data\250728_Correction_high_aber_lamin\25080_low\low\Image001_PALM_Tracer\localizations-20250108_165530.csv"
fichier_excel = r"C:\Users\rlusson\Desktop\Data\250728_Correction_high_aber_lamin\25080_low\low\histogrammes.xlsx"  # ← sortie Excel

colonnes_a_plot = {
    "Circularity": (0, 1, 100),
    "Sigma X": (0, 10, 100),
    "Integrated Intensity": (0, 15000, 100),
    "MSE XY": (0, 1, 100),
}

# === 📥 Chargement du CSV ===
df = pd.read_csv(fichier)
df_filtré = df[df["Integrated Intensity"] != 0]

# === 📊 Histogrammes & Excel ===
writer = pd.ExcelWriter(fichier_excel, engine='xlsxwriter')

# === 📊 Histogrammes ===
for nom_colonne, (xmin, xmax, nb_bins) in colonnes_a_plot.items():
    if nom_colonne not in df_filtré.columns:
        print(f"⚠️ Colonne '{nom_colonne}' non trouvée dans le fichier.")
        continue

    donnees = df_filtré[nom_colonne].dropna().tolist()  # enlève les NaN

    # Histogramme (matplotlib)
    counts, bins = np.histogram(donnees, bins=nb_bins, range=(xmin, xmax))
    bin_edges = bins[:-1]  # début de chaque bin

    plt.figure(figsize=(8, 5))
    plt.hist(donnees, bins=nb_bins, range=(xmin, xmax), color="#4B8B8D", edgecolor="black")
    plt.title(f"Histogramme de {nom_colonne}")
    plt.xlabel(nom_colonne)
    plt.ylabel("Nombre d'occurrences")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Sauvegarde dans Excel
    df_hist = pd.DataFrame({
    "Bins": bin_edges,
    "Counts": counts
        })
    df_hist.to_excel(writer, sheet_name=nom_colonne[:31], index=False)  # Sheet name max = 31 caractères

# Enregistrement final
writer.close()
print(f"✅ Excel enregistré : {fichier_excel}")

#%%Tests


img_path = r"C:\Users\rlusson\Desktop\Data\Carac_AO\Coverslip\No_aber\1\MIP\Image0006_refZer4.tif"

# 1) Chargement
n, img = getImage(img_path)

# 2) Calcul du poids et du masque
a = 1.3
b = 1
metric_weight, circmask1NA = getfreqsup4(
    n,
    beta=b,
    param_alpha=a,
    px_size=108,
    thisNA=1.27,
    thislambda=0.589
)

# 3) Sauvegarde du masque pondéré (float entre 0 et 1) en TIFF 16-bits
mask_weight = circmask1NA * metric_weight
# normalisation [0–65535]
mask_uint16 = ( (mask_weight - mask_weight.min()) 
                / (mask_weight.max() - mask_weight.min()) 
                * 65535 ).astype(np.uint16)

out_mask = r"C:\Users\rlusson\Desktop\mask_weight.tif"
imageio.imwrite(out_mask, mask_uint16)
print(f"✅ Masque pondéré sauvegardé sous : {out_mask}")

# 4) FFT complexe et filtrage
F  = fftshift(fft2(img))
Ff = F * metric_weight
img_filtered = np.real(ifft2(ifftshift(Ff)))

# 5) Sauvegarde de l’image filtrée en TIFF 16-bits
filtered_uint16 = ( (img_filtered - img_filtered.min()) 
                    / (img_filtered.max() - img_filtered.min()) 
                    * 65535 ).astype(np.uint16)

out_filt = r"C:\Users\rlusson\Desktop\img_filtered.tif"
imageio.imwrite(out_filt, filtered_uint16)
print(f"✅ Image filtrée sauvegardée sous : {out_filt}")

# 6) (Optionnel) Affichage  
plt.figure(figsize=(6,6))
plt.imshow(mask_weight, cmap='gray')
plt.title("Masque pondéré")
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(img_filtered, cmap='gray')
plt.title("Image filtrée")
plt.axis('off')
plt.show()

#%%
def choisir_metrique_liste(metriques_disponibles):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    metrique, ok = QInputDialog.getItem(
        None,
        "Choix de la métrique",
        "Sélectionnez un mode :",
        metriques_disponibles,
        editable=False
    )
    if ok:
        return metrique
    return None


#Choix d'une quantité
def choisir_quantite_liste(quantites_disponibles):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    quantite, ok = QInputDialog.getItem(
        None,
        "Choix de la quantité",
        "Sélectionnez une quantité :",
        quantites_disponibles,
        editable=False
    )
    if ok:
        return quantite
    return None

#Choix d'une profondeur
def choisir_profondeur_liste(profondeurs_disponibles):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    profondeur, ok = QInputDialog.getItem(
        None,
        "Choix de la quantité",
        "Sélectionnez une quantité :",
        profondeurs_disponibles,
        editable=False
    )
    if ok:
        return profondeur
    return None


#Choix d'un jeu
def choisir_jeu_liste(jeux_disponibles):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    jeu, ok = QInputDialog.getItem(
        None,
        "Choix du jeu de données",
        "Sélectionnez un jeu :",
        jeux_disponibles,
        editable=False
    )
    if ok:
        return jeu
    return None

def get_freq_bandpass(image,px_size_um, taille_image_px, freq_interne, freq_externe):
    print(image.shape)
    rows, cols = image.shape

    

    crow, ccol = rows // 2, cols // 2 

    radius_interne= get_radius(px_size_um, freq_voulue_mm=freq_interne, taille_image_px=taille_image_px)
    radius_externe=get_radius(px_size_um, freq_voulue_mm=freq_externe, taille_image_px=taille_image_px)

    x,y= np.ogrid[:rows, :cols]
    center_distance= np.sqrt((x-ccol)**2+(y-crow)**2)
    bandpass_mask= np.logical_and(center_distance>radius_interne, center_distance<radius_externe)

    return bandpass_mask


def get_metric3(abs_fft, bandpass_mask, n, thisNA, thislambda, px_size):
    fx = np.outer((np.arange(n) - n // 2) / (2 * n) / px_size, np.ones(n)) 
    fy = np.outer(np.ones(n), (np.arange(n) - n // 2) / (2 * n) / px_size) 
    fr = np.abs(np.sqrt(fy**2 + fx**2)) 

    # Create circular masks    
    circmask1NA = (fr <= (2 * thisNA) / thislambda) / 1.0  

    Mnum = 0
    Mden = 0
    M=0
    i = 0
    while i < n:
        j = 0
        while j < n:
            Mnum += abs_fft[i][j] * bandpass_mask[i][j]
            Mden += abs_fft[i][j] * circmask1NA[i][j]
            j += 1
        i += 1

    M = Mnum / Mden
    return M

def this_bandpass2(img_path, freq_min, freq_max):
    n, img = getImage(img_path)
    abs_fft = getFFT(img)
    bandpass_mask = get_freq_bandpass(
        image=img,
        px_size_um=0.108,
        taille_image_px=256,
        freq_interne=freq_min,
        freq_externe=freq_max
    )
    M = get_metric3(
        abs_fft,
        bandpass_mask,
        n,
        thisNA=1.49,
        thislambda=0.46,
        px_size=0.108
    )
    return M


def bandpass_metric_9N(imagem3, imagem2, imagem1, imageref, imagep1, imagep2, imagep3, imagep4,fmin,fmax):
    ym3=this_bandpass2(imagem3,fmin,fmax)
    ym2=this_bandpass2(imagem2,fmin,fmax)
    ym1=this_bandpass2(imagem1,fmin,fmax)
    yref=this_bandpass2(imageref,fmin,fmax)
    yp1=this_bandpass2(imagep1,fmin,fmax)
    yp2=this_bandpass2(imagep2,fmin,fmax)
    yp3=this_bandpass2(imagep3,fmin,fmax)
    yp4=this_bandpass2(imagep4,fmin,fmax)

    return  ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4

choix_metrique = "MIP"

choix_quantite = "No_aber"

choix_jeu = "1"

choix_profondeur = "Coverslip"

image_paths = importation_chemin(choix_metrique,choix_quantite,choix_jeu,choix_profondeur)

#r_min = 0
r_min = 22
r_max = 67
#r_max =80

f_min = pix_to_freq(r_min)
f_max = pix_to_freq(r_max)

image_paths_par_mode = {}
dic_a={}
dic_r2={}

for m in modes:
    print(m)

    # Récupération des chemins pour le mode m
    image_paths_par_mode[m] = recuperer_images_par_mode(image_paths, modes_zernike[m])

    # Tri des images par quantité d’aberration alpha
    image_paths_par_mode[m] = sorted(
        image_paths_par_mode[m],
        key=lambda path: extraire_quantite_alpha(os.path.basename(path))
    )

    print(image_paths_par_mode)
    # Application des métriques sur les 8 images attendues
    ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4 = bandpass_metric_9N(
         image_paths_par_mode[m][1],
        image_paths_par_mode[m][2], image_paths_par_mode[m][3],
        image_paths_par_mode[m][4], image_paths_par_mode[m][5],
        image_paths_par_mode[m][6], image_paths_par_mode[m][7],
         image_paths_par_mode[m][8],
        f_min, f_max
    )


    # Extraction des points de fit
    x, y = get_fit_array_9N(ym3, ym2, ym1, yref, yp1, yp2, yp3, yp4)


    # Fit quadratique
    #a, b, c, xl, yl, R2 = fit_gaussien(x, y,estimer_p0_gaussienne(x, y))
    a, b, c, xl, yl, R2 ,r2 = fit((x, y),True)

    dic_a[m] = -a
    dic_r2[m] = r2
    

print("a:")
for key, value in dic_a.items():
    print(f"{key} : {value}")

print("r^2:")
for key, value in dic_r2.items():
    print(f"{key} : {value}")
