
import sys
from pathlib import Path                                #Sert à manipuler les chemins fichiers de façon portable

root = Path(__file__).resolve().parent.parent           #__file__ = chemin complet du fichier scripts/0A_load_image.py puis remonte de deux niveau (.paretn.parent) --> root = dossier racine
sys.path.append(str(root))                              #Liste des endroits ou Python cherche les modules --> Append = rajoute le dossier racine "(ici dans dossier racine: "ao-sospim" 

import matplotlib.pyplot as plt

from src.image_io import load_tif, make_2d_image, save_tif_image

image_path = Path("data/Profondeur_NoAber_1_RefZer4.tif")

print("Chemin image :", image_path)
print("Image existe? :", image_path.exists())

stack_or_image = load_tif(image_path)

image_std = make_2d_image(stack_or_image, mode="std")

"""print("Image 2D obtenue")
print("shape :", image_std.shape)
print("Dtype :",image_std.dtype)
print("min/max :", image_std.min(), image_std.max())"""

"""plt.imshow(image_std, cmap="gray")
plt.title("Image 2D utilisée")
plt.colorbar()
plt.show(block=False)
plt.pause(2)
plt.close()"""

#Sauvergarde image .png
#plt.savefig("results/image_2d_preview.png", dpi=150)
#plt.close()

save_tif_image(image_std, "data/results/Profondeur_NoAber_1_RefZer4_STD.tif", "float32", norm=True)

