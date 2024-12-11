import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2

# Load the YOLO model
model = YOLO("/teamspace/studios/this_studio/model/yolov10n_blood.onnx")

def predict(image):
    """
    Run inference on the image and return the annotated image.
    """
    results = model(image)[0]
    return cv2.cvtColor(results.plot(),cv2.COLOR_BGR2RGB)

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="Blood Cell Detection",
    description="Upload an image to detect WBCs, RBCs, and Platelets using a YOLOv10n model.",
)

if __name__ == "__main__":
    interface.launch(share=True)
