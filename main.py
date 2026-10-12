"""Détection webcam avec YOLO et estimation de la couleur dominante."""

import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
from typing import cast
from ultralytics import YOLO
from ultralytics.engine.results import Results


def get_dominant_color(
    hsv_frame: MatLike, object_mask: NDArray[np.bool_] | None
) -> tuple[str, tuple[int, int, int]]:
    """Renvoie le nom et la couleur d'affichage dominants dans la zone masquée."""
    # Sans pixels à analyser, la couleur ne peut pas être déterminée.
    if object_mask is None or not np.any(object_mask):
        return "inconnue", (255, 255, 255)

    # La médiane distingue les objets gris/noirs/blancs des couleurs vives.
    pixels = hsv_frame[object_mask]
    median_saturation = float(np.median(pixels[:, 1]))
    median_value = float(np.median(pixels[:, 2]))

    if median_saturation < 35:
        if median_value < 55:
            return "noire", (255, 255, 255)
        if median_value > 200:
            return "blanche", (0, 0, 0)
        return "grise", (80, 80, 80)

    # Écarte les pixels trop peu saturés ou trop sombres avant d'étudier leur teinte.
    colored_pixels = pixels[(pixels[:, 1] >= 50) & (pixels[:, 2] >= 35)]
    if len(colored_pixels) == 0:
        return "inconnue", (255, 255, 255)

    # L'histogramme choisit la teinte la plus fréquente; OpenCV code H de 0 à 179.
    hue_values = colored_pixels[:, 0].astype(np.intp)
    hue = int(np.bincount(hue_values, minlength=180).argmax())

    # Ces seuils associent la teinte dominante à un nom de couleur lisible.
    if hue < 8 or hue >= 172:
        return "rouge", (0, 0, 255)
    if hue < 20:
        if median_value < 115:
            return "marron", (19, 69, 139)
        return "orange", (0, 140, 255)
    if hue < 32:
        return "jaune", (0, 220, 255)
    if hue < 80:
        return "verte", (0, 190, 0)
    if hue < 100:
        return "turquoise", (180, 220, 0)
    if hue < 130:
        return "bleue", (255, 0, 0)
    if hue < 150:
        return "violette", (200, 0, 200)
    return "rose", (180, 105, 255)


# Le modèle de segmentation fournit les classes détectées et leurs masques.
print("Chargement du modèle YOLO...", flush=True)
model = YOLO("yolo26n-seg.pt")
print("Modèle YOLO chargé.", flush=True)

# La caméra est ouverte au chargement du module et réutilisée par la boucle.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")

# Seules ces classes sont conservées et affichées en français.
object_labels = {
    "apple": "Pomme",
    "cup": "Tasse",
    "keyboard": "Clavier",
    "mouse": "Souris",
    "remote": "Télécommande",
    "cell phone": "Téléphone",
    "book": "Livre",
    "bottle": "Bouteille",
    "banana": "Banane",
    "orange": "Orange",
    "broccoli": "Brocoli",
    "carrot": "Carotte",
    "hot dog": "Hot-dog",
}


def detect_objects(*args, **kwargs):
    """Point d'extension YOLO (installation optionnelle sur PC)."""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Impossible de lire une image depuis la webcam.")

            # HSV sert à classifier les couleurs; output reçoit les annotations.
            height, width = frame.shape[:2]
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            output = frame.copy()

            # conf écarte les détections faibles; stream=False traite cette image.
            predictions = cast(
                list[Results],
                model.predict(frame, conf=0.35, verbose=False, stream=False),
            )

            result: Results = predictions[0]

            if result.boxes is not None:
                detection_index = 0
                for box in result.boxes:
                    current_detection_index = detection_index
                    detection_index += 1

                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name not in object_labels:
                        continue

                    # Les coordonnées de boîte servent à l'annotation et au masque de secours.
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    object_mask = None

                    # Le masque suit la forme de l'objet; à défaut, la boîte entière
                    # fournit une zone approximative pour estimer sa couleur.
                    if result.masks is not None and current_detection_index < len(
                        result.masks.data
                    ):
                        mask = result.masks.data[current_detection_index].cpu().numpy()
                        object_mask = (
                            cv2.resize(
                                mask,
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 0.5
                        )
                    else:
                        object_mask = np.zeros((height, width), dtype=np.bool_)
                        object_mask[
                            max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)
                        ] = True

                    # La couleur est estimée uniquement dans la zone de l'objet.
                    color_name, box_color = get_dominant_color(hsv_frame, object_mask)
                    label = f"{object_labels[class_name]} {color_name}"

                    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(
                        output,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2,
                    )

            # La touche q ferme la boucle; finally libère aussi la caméra en cas d'erreur.
            cv2.imshow("Pommes et tasses", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
