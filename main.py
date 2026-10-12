"""Détection webcam des objets et des visages."""

import cv2
from cv2.data import haarcascades
from cv2.typing import MatLike
import numpy as np
from numpy.typing import NDArray
from typing import cast
from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL_PATH = "yolo26n-seg.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
BLUE = (255, 0, 0)

OBJECT_LABELS = {
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


def get_dominant_color(
    hsv_frame: MatLike, object_mask: NDArray[np.bool_] | None
) -> tuple[str, tuple[int, int, int]]:
    """Renvoie la couleur dominante de la zone masquée et sa couleur RGB."""
    if object_mask is None or not np.any(object_mask):
        return "inconnue", (255, 255, 255)

    # On sélectionne les pixels de la zone masquée et on calcule la médiane de la saturation et de la valeur.
    pixels = hsv_frame[object_mask]
    median_saturation = float(np.median(pixels[:, 1]))
    median_value = float(np.median(pixels[:, 2]))

    # On considère que les pixels peu saturés sont gris, noir ou blanc.
    if median_saturation < 35:
        if median_value < 55:
            return "noire", (255, 255, 255)
        if median_value > 200:
            return "blanche", (0, 0, 0)
        return "grise", (80, 80, 80)

    # On ne considère que les pixels suffisamment saturés pour déterminer la couleur.
    colored_pixels = pixels[(pixels[:, 1] >= 50) & (pixels[:, 2] >= 35)]
    if len(colored_pixels) == 0:
        return "inconnue", (255, 255, 255)

    # L'histogramme trouve la teinte HSV la plus fréquente.
    hue_values = colored_pixels[:, 0].astype(np.intp)
    # hue est un entier entre 0 et 179 représentant la teinte dominante.
    hue = int(np.bincount(hue_values, minlength=180).argmax())

    # On détermine la couleur en fonction de la teinte dominante et de la valeur médiane.
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


def load_face_detector() -> cv2.CascadeClassifier:
    """Charge le classifieur Haar fourni avec OpenCV."""
    detector = cv2.CascadeClassifier(
        haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Impossible de charger le détecteur de visages.")
    return detector


def get_object_mask(
    result: Results,
    detection_index: int,
    box_coordinates: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Retourne le masque de segmentation, ou la boîte si aucun masque n'existe."""
    height, width = frame_shape
    if result.masks is not None and detection_index < len(result.masks.data):
        mask = result.masks.data[detection_index].cpu().numpy()
        return (
            cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0.5
        )

    # La boîte est un repli approximatif pour les modèles sans masque.
    x1, y1, x2, y2 = box_coordinates
    object_mask = np.zeros((height, width), dtype=np.bool_)
    object_mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return object_mask


def draw_object_detection(
    output: MatLike,
    hsv_frame: MatLike,
    result: Results,
    class_name: str,
    detection_index: int,
    coordinates: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> None:
    """Estime la couleur et dessine une boîte annotée pour un objet."""
    object_mask = get_object_mask(result, detection_index, coordinates, frame_shape)
    color_name, box_color = get_dominant_color(hsv_frame, object_mask)
    label = f"{OBJECT_LABELS[class_name]} {color_name}"
    x1, y1, x2, y2 = coordinates

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


def draw_object_detections(frame: MatLike, result: Results) -> MatLike:
    """Dessine chaque objet reconnu avec sa classe et sa couleur estimée."""
    height, width = frame.shape[:2]
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()

    if result.boxes is None:
        return output

    # L'index relie chaque boîte au masque de segmentation correspondant.
    detection_index = 0
    for box in result.boxes:
        current_index = detection_index
        detection_index += 1

        # On ignore les objets dont la classe n'est pas dans la liste des labels.
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        if class_name not in OBJECT_LABELS:
            continue

        raw_coordinates = box.xyxy[0].tolist()
        coordinates = (
            int(raw_coordinates[0]),
            int(raw_coordinates[1]),
            int(raw_coordinates[2]),
            int(raw_coordinates[3]),
        )
        draw_object_detection(
            output,
            hsv_frame,
            result,
            class_name,
            current_index,
            coordinates,
            (height, width),
        )

    return output


def draw_face_detections(
    frame: MatLike, output: MatLike, detector: cv2.CascadeClassifier
) -> None:
    """Dessine les visages en bleu, sans tenter d'en déduire le genre."""
    # On convertit l'image en niveaux de gris pour la détection de visages.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # On détecte les visages dans l'image en niveaux de gris.
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    # On dessine un rectangle bleu autour de chaque visage détecté.
    for x, y, width, height in faces:
        cv2.rectangle(output, (x, y), (x + width, y + height), BLUE, 3)
        cv2.putText(
            output,
            "Visage",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            BLUE,
            2,
        )


def process_frame(
    frame: MatLike, model: YOLO, face_detector: cv2.CascadeClassifier
) -> MatLike:
    """Applique YOLO puis ajoute les boîtes d'objets et de visages."""
    predictions = cast(
        list[Results],
        model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            stream=False,
        ),
    )
    output = draw_object_detections(frame, predictions[0])
    draw_face_detections(frame, output, face_detector)
    return output


def run_webcam(model: YOLO, face_detector: cv2.CascadeClassifier) -> None:
    """Lit et affiche les images de la webcam jusqu'à l'appui sur Q."""
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la webcam {CAMERA_INDEX}.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Impossible de lire une image depuis la webcam.")

            output = process_frame(frame, model, face_detector)
            cv2.imshow("Détection objets et visages", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Initialise les modèles et démarre la capture vidéo."""
    print("Chargement des modèles...", flush=True)
    model = YOLO(MODEL_PATH)
    face_detector = load_face_detector()
    print("Modèles chargés.", flush=True)
    run_webcam(model, face_detector)


if __name__ == "__main__":
    main()
