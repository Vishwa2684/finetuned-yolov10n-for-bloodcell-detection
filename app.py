import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2

# Load the YOLO model
model = YOLO("/teamspace/studios/this_studio/model/yolov10n_blood.onnx")

def predict(image):
    """
    Run inference on the image and return the annotated image and detection details.
    """
    results = model(image)[0]
    # Annotated image
    annotated_image = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    # Extract class names and confidence scores
    boxes = results.boxes
    class_ids = boxes.cls.int().tolist()
    confidences = (boxes.conf * 100).tolist()
    
    # Map class IDs to class names (update according to your class labels)
    class_labels = {0: "WBC", 1: "RBC", 2: "Platelets"}
    detections = [
        f"Class: {class_labels.get(cls_id, 'Unknown')} | Confidence: {conf:.2f}%"
        for cls_id, conf in zip(class_ids, confidences)
    ]
    
    return annotated_image, "\n".join(detections)

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        "image",  # Annotated image
        gr.Textbox(label="Detections"),  # Textbox to display detections
    ],
    title="Blood Cell Detection",
    description="Upload an image to detect WBCs, RBCs, and Platelets using a YOLOv10n model.",
)

if __name__ == "__main__":
    interface.launch(share=True)
