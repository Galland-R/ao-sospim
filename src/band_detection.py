import numpy as np
from itertools import groupby
from scipy.ndimage import gaussian_filter1d


def compare_rotational_profiles(
    bins_ref,
    profile_ref,
    bins_aberrated,
    profile_aberrated,
    smoothing_sigma=3,
    normalize=True,
):
    """
    Compare deux moyennes rotationnelles de FFT :
    une image de référence et une image aberrée.

    Cette fonction reprend la logique utilisée dans le code étudiant :
        fft1 = fft1 / np.max(fft1)
        fft2 = fft2 / np.max(fft2)
        diff_curve = np.abs(fft1 - fft2)
        smoothed_diff = gaussian_filter1d(diff_curve, sigma=3)

    1. met les deux profils à la même longueur
    2. normalise éventuellement les profils entre 0 et 1
    3. calcule |profil_aberré - profil_ref|
    4. lisse cette différence

    Retourne:
    bins : array
        Rayons communs.
    ref : array
        Profil de référence, éventuellement normalisé.
    aberrated : array
        Profil aberré, éventuellement normalisé.
    difference : array
        Différence absolue |aberrated - ref|.
    smoothed_difference : array
        Différence lissée.   
    """

    # Sécurité : si les deux profils n'ont pas exactement la même longueur, on tronque à la longueur commune.
    n = min(len(profile_ref), len(profile_aberrated))

    bins = bins_ref[:n]
    ref = profile_ref[:n]
    aberrated = profile_aberrated[:n]

    # Normalisation optionnelle
    if normalize:
        if np.max(ref) != 0:
            ref = ref / np.max(ref)

        if np.max(aberrated) != 0:
            aberrated = aberrated / np.max(aberrated)

    difference = np.abs(ref - aberrated)

    # Lissage de la différence pour éviter que la détection de bande soit dominée par du bruit ponctuel.
    smoothed_difference = gaussian_filter1d(difference,sigma=smoothing_sigma)

    return bins, ref, aberrated, difference, smoothed_difference


# Cherche une zone our la difféence spectral est supéreiur à moyenne + k * std (fonction LB)
def detect_adaptive_band(bins, profile, k=1.0, ignore_first_n=10, max_radius=None):
    """
    Détecte automatiquement une bande spectrale sensible.

    Cette fonction est directement reprise du code étudiant.

    Principe :
    - on calcule un seuil adaptatif :
          seuil = moyenne(profile) + k * écart-type(profile)
    - on garde les points du profil au-dessus du seuil
    - on cherche la plus longue bande continue
    - on retourne ses bornes r_min/r_max en pixels
    - Option: on filtre rayon supéreiur à fréquence de coupure (expérimental ou théorique)

    """

    # On ignore éventuellement les premiers rayons.
    profile_crop = profile[ignore_first_n:]
    bins_crop = bins[ignore_first_n:]

    # Filtre pour fréquence > fréquence coupure
    if max_radius is not None:
        valid = bins_crop <= max_radius
        profile_crop = profile_crop[valid]
        bins_crop = bins_crop[valid]

    if len(profile_crop) == 0:
        return None, None, 0

    # Seuil pour détection bandes
    threshold = np.mean(profile_crop) + k * np.std(profile_crop)
    mask = profile_crop >= threshold        # Masque booléen: True la ou le profil dépasse le seuil
    indices = np.where(mask)[0]             # Indice des points au-dessus du seuil

    # Regroupement des indices consécutifs.
    groups = [
        list(g)
        for _, g in groupby(
            indices,
            key=lambda i, c=iter(range(len(indices))): i - next(c)
        )
    ]

    longest_band = max(groups, key=len) if groups else []   # On garde le plus long groupe continu.

    if longest_band:
        r_min = bins_crop[longest_band[0]]
        r_max = bins_crop[longest_band[-1]]
        width = r_max - r_min
        return r_min, r_max, width
    else:
        return None, None, 0


#Calcul de l'aire de la bande (fonction LB)
def compute_integral_in_band(r_min, r_max, bins, profile):
    """
    Calcule l'aire sous le profil dans la bande détectée.

    Cette fonction reprend la logique du code étudiant :

        mask = (bins >= r_min) & (bins <= r_max)
        np.trapz(profile[mask], bins[mask])
    """
    if r_min is None or r_max is None:
        return 0

    mask = (bins >= r_min) & (bins <= r_max)
    return np.trapezoid(profile[mask], bins[mask])
