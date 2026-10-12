import cv2
from cv2.data import haarcascades
import numpy as np
from ultralytics import YOLO

# Modèle YOLO de segmentation préentraîné sur COCO
print("Chargement du modèle YOLO...", flush=True)
model = YOLO("yolo26n-seg.pt")
print("Modèle YOLO chargé.", flush=True)

# Détecteur de visages OpenCV
face_detector = cv2.CascadeClassifier(
    haarcascades + "haarcascade_frontalface_default.xml"
)

# Ouvrir la webcam
print("Ouverture de la webcam...", flush=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")
print("Webcam ouverte.", flush=True)

# Couleurs au format BGR : bleu, vert, rouge
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
first_frame_pending = True

try:
    while True:
        if first_frame_pending:
            print("Lecture de la première image...", flush=True)
        ret, frame = cap.read()
        if first_frame_pending:
            print(f"Lecture terminée (ret={ret}).", flush=True)

        if not ret:
            break

        height, width = frame.shape[:2]

        # 1. Détection et segmentation avec YOLO
        if first_frame_pending:
            print("Première prédiction YOLO...", flush=True)
        result = model.predict(frame, conf=0.35, verbose=False)[0]
        if first_frame_pending:
            print("Première prédiction terminée.", flush=True)
            first_frame_pending = False

        # 2. Masque des objets reconnus par YOLO
        foreground = np.zeros((height, width), dtype=np.uint8)

        if result.masks is not None:
            for mask in result.masks.data:
                mask = mask.cpu().numpy()
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
                foreground[mask > 0.5] = 255

        # 3. Teinter le fond en jaune
        yellow = np.full_like(frame, YELLOW)

        tinted = cv2.addWeighted(frame, 0.65, yellow, 0.35, 0)

        # Garder les objets segmentés dans leur couleur originale
        output = np.where(foreground[:, :, None] > 0, frame, tinted)

        # 4. Encadrer les pommes en rouge
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                if class_name != "apple":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(output, (x1, y1), (x2, y2), RED, 3)

                cv2.putText(
                    output,
                    "Pomme",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    RED,
                    2,
                )

        # 5. Détecter les visages sur l'image originale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        # Encadrer les visages en bleu
        for x, y, w, h in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), BLUE, 3)

            cv2.putText(
                output,
                "Visage",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                BLUE,
                2,
            )

        # 6. Afficher le résultat en temps réel
        cv2.imshow("YOLO + OpenCV", output)

        # Appuyer sur Q pour quitter
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
