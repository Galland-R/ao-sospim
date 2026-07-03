"""
Ce module sert à gérer les noms de fichiers et à retrouver automatiquement:

    toutes les images d'un dossier
    → les images d'un mode donné, par exemple "Zer4"
    → l'image référence "refZer4"
    → l'image aberrée "+4alphaZer4"

"""

import os                   #Manipulation des noms de fichiers
import re                   #Sert aux expressions régulières utiles pour extraire "+4alpha", "-3alpha" ..
from pathlib import Path

# Dictionnaire: permet de passer d'un nom lisible (ex: "Astigmatisme 0°") au code correspondant utilisé dans les fichiers ("Zer4")
MODES_ZERNIKE = {
    "Astigmatisme 0°": "Zer4",
    "Astigmatisme 45°": "Zer5",
    "Coma 0°": "Zer6",
    "Coma 90°": "Zer7",
    "Aberration Sphérique": "Zer8",
    "Trefoil 0°": "Zer9",
    "Trefoil 90°": "Zer10",
}

# Listing des image tif dans un dossier (fonction RG)
def list_tif_images(folder):
    print(f"\ndata folder: {folder}")
    folder = Path(folder)
    return sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))


# Extrait la partie finale du nom de fichier (ex: "image001_+4alphaZer4.tif" --> "+4alphaZer4") (fonction LB)
def extraire_suffixe(path):
    filename = os.path.basename(path)       # Garde le nom du fichier sans le dossier
    name, _ = os.path.splitext(filename)    # Enlève l'extension (".tif")

    if "_" in name:
        return name.split("_")[-1]          # Coupe au niveau de "_" et garde le dernier morceau
    else:
        return None


# Extrait la quantité d'aberration (fonction LB)
def extraire_quantite_alpha(path):
    """
    Extrait la quantité d’alpha d’un nom de fichier, par exemple :
    - Image0064_+4alphaZer10.tif → 4
    - Image0006_refZer4.tif      → 0 (cas "ref")
    """
    filename = os.path.basename(path)

    # Cas du fichier "ref" (Référence)
    if "ref" in filename.lower():
        return 0

    # Regex pour trouver la quantité d’alpha (ex: +4alpha ou -3alpha)
    match = re.search(r"([+-]?\d+)alpha", filename)
    if match:
        return int(match.group(1))

    raise ValueError(f"Impossible d'extraire alpha depuis : {filename}")


# Ne garde uniquement les images correspondante à un mode donné (fonction LB)
def recuperer_images_par_mode(image_paths, mode_zernike):
    chemins_mode = []

    for path in image_paths:
        suffixe = extraire_suffixe(path)

        if suffixe and suffixe.endswith(mode_zernike):
            chemins_mode.append(path)

    return sorted(chemins_mode, key=extraire_quantite_alpha)        # Trié par quantité d'aberration: (key=extraire_quantité_alpha)



# Cherche image de référence (fonction RG)
def trouver_reference(images_mode, mode_zernike):
    """
    pour toutes les images d'un mode (ex "images_zer4") et ce mode (ex "Zer4")
    renvoie le chemine du fichier corespondant
    """
    for path in images_mode:
        suffixe = extraire_suffixe(path)

        if suffixe == "ref" + mode_zernike:
            return path

    raise FileNotFoundError(f"Aucune référence trouvée pour {mode_zernike}")


# Cherche un image aberrée précise (fonction RG)
def trouver_image_aberration(images_mode, mode_zernike, alpha="+4alpha"):
    """"
    pour mode_zernike="Zer4" et alpha="+4alpha" --> cherche "+4alphaZer4" et renvoie le chemin de l'image corespondante
    """
    cible = alpha + mode_zernike

    for path in images_mode:
        suffixe = extraire_suffixe(path)

        if suffixe == cible:
            return path

    raise FileNotFoundError(f"Aucune image {cible} trouvée")