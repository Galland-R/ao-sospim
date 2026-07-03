
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent           #__file__ = chemin complet du fichier scripts/01_load_image.py puis remonte de deux niveau (.paretn.parent) --> root = dossier racine
sys.path.append(str(root))                              #Liste des endroits ou Python cherche les modules --> Append = rajoute le dossier racine "(ici dans dossier racine: "ao-sospim" 


#from old_code.bandpass_limit_determination_remi_final import afficher_masque_bandpass

taille_image=[256, 256]
px_size_um=108
freq_interne=1
freq_externe=2


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

"""afficher_masque_bandpass(
    taille_image=[256, 256],
    px_size_um=108,
    freq_interne=500,
    freq_externe=1000
)"""