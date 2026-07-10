from pathlib import Path
import numpy as np
import tifffile

from src import metrics as met

def load_tif(path):
    """
    Charge une image TIFF et la convertit en tableau NumPy
    """
    image = tifffile.imread(path)
    image = np.asarray(image)

    """
    print(f"Image chargée: {path}")
    print("Type :", type(image))
    print("Shape :", image.shape)
    print("Dtype :", image.dtype)
    """

    return image


def make_2d_image(image, mode="max", plane_index=0):
    """
    Convertit une image ou stack en image 2D.

    Modele possible:
    - "none"    : l'image est déjà 2D
    - "mip"     : maximum intensity projection
    - "sum"     : somme des plans
    - "mean"    : moyenne des plans
    - "frame"   : un seul plan choist par plane_index
    - "std"     : Déviation standard  
    """

    if image.ndim == 2:
        return image
    
    if image.ndim !=3:
        raise ValueError(f"Image avec dimension non gérée: {image.shape}")
    
    if mode == "MIP":
        return np.max(image, axis=0)
    
    if mode == "SUM":
        return np.sum(image, axis=0)
    
    if mode == "mean":
        return np.mean(image, axis=0)
    
    if mode == "Frame":
        return image[plane_index]
    
    if mode == "STD":
        return np.std(image, axis=0)
    
    if mode == "mean_ROI":
        return met.average_roi(image)

    
    raise ValueError(f"Mode inconnu : {mode}")


def save_tif_image(image, output_path, dtype="uint16"):
    """
    Enregistre une image 2D ou 3D au format TIFF compatible ImageJ/Fiji.

    Paramètres
    ----------
    image : np.ndarray
        Image à sauvegarder.
    output_path : str ou Path
        Chemin du fichier de sortie.
    dtype : str
        Type d'image sauvegardé : "uint16", "uint8" ou "float32".
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(image)

    if dtype == "uint16":
        image_to_save = image.astype(np.uint16)

    elif dtype == "uint8":
        image_to_save = image.astype(np.uint8)

    elif dtype == "float32":
        image_to_save = image.astype(np.float32)

    else:
        raise ValueError(f"dtype non reconnu : {dtype}")

    tifffile.imwrite(output_path, image_to_save)

    print(f"Image sauvegardée : {output_path}")


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