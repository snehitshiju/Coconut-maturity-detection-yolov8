from ultralytics import YOLO
import cv2
import os

TIMELINE = {
    'Premature': {'days_to_mature': 40, 'days_to_overmature': 70},
    'Potential': {'days_to_mature': 10, 'days_to_overmature': 25},
    'Mature':    {'days_to_mature': 0,  'days_to_overmature': 15},
}

CLASS_COLORS = {
    'Premature': (0, 200, 255),
    'Potential': (0, 165, 255),
    'Mature':    (0, 200, 0),
}

COCONUT_CLASSES = ['Mature', 'Potential', 'Premature']

def get_timeline_message(label):
    t = TIMELINE.get(label, {})
    if label == 'Premature':
        return f"Approx. {t['days_to_mature']} days to mature | {t['days_to_overmature']} days to overmature"
    elif label == 'Potential':
        return f"Almost ready! Approx. {t['days_to_mature']} days to mature | {t['days_to_overmature']} days to overmature"
    elif label == 'Mature':
        return f"Ready to harvest now! | {t['days_to_overmature']} days before overmature"
    return "Unknown maturity stage"

def get_card_class(label):
    mapping = {
        'Premature': 'premature',
        'Potential': 'potential',
        'Mature':    'mature',
    }
    return mapping.get(label, 'premature')

def is_coconut_detected(results, model):
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in COCONUT_CLASSES:
            return True
    return False

def run_detection(image_path, save_path, model_path='runs/detect/coconut_model/weights/best.pt'):
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
        print(f"[WARNING] Custom model not found. Using default: {model_path}")

    model = YOLO(model_path)
    results = model(image_path)[0]

    # No detections at all or no coconut found
    if len(results.boxes) == 0 or not is_coconut_detected(results, model):
        return 'no_coconut'

    img = cv2.imread(image_path)
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        # Skip anything that is not a coconut class
        if label not in COCONUT_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(label, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)

        # Draw label text
        cv2.putText(img, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        timeline_msg = get_timeline_message(label)
        card_class = get_card_class(label)

        detections.append({
            'label':      label,
            'confidence': round(conf * 100, 1),
            'timeline':   timeline_msg,
            'card_class': card_class
        })

    # All detections were non-coconut
    if not detections:
        return 'no_coconut'

    cv2.imwrite(save_path, img)
    return detections
