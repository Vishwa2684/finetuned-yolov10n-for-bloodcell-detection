# Blood Cell Detection using YOLOv10n

## Project Overview
This project implements a custom object detection model for identifying blood cell types using YOLOv10n, fine-tuned on the BCCD (Blood Cell Classification) Dataset.

## 🔬 Dataset
- **Source**: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- **Classes Detected**: 
  - White Blood Cells (WBC)
  - Red Blood Cells (RBC)
  - Platelets

## 🤖 Model Details
- **Base Model**: YOLOv10n
- **Hardware**: L4 GPU for fine-tuning
- **Training Epochs**: 100
- **Image Size**: 640x640

## 📊 Model Performance
### Overall Metrics
- **Precision**: 87.42%
- **Recall**: 94.61%
- **mAP50**: 95.76%
- **mAP50-95**: 76.03%
- **Fitness Score**: 0.780

### Dataset Configuration
```yaml
train: /teamspace/studios/this_studio/dataset/train/images  
test: /teamspace/studios/this_studio/dataset/test/images
val: /teamspace/studios/this_studio/dataset/val/images 
nc: 3                          
names:
- WBC
- RBC
- Platelets
```

## 🚀 Deployment
- **Platform**: Hugging Face Spaces
- **Interactive Demo**: [Blood Cells Detection](https://huggingface.co/spaces/vish26/bloodcells-detection)

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch
- Ultralytics
- CUDA (for GPU support)

### Setup
```bash
# Clone the repository
git clone <your-repo-link>

# Install dependencies
pip install ultralytics torch

```

## 📝 Usage

### Inference Example
```python
from ultralytics import YOLO

# Load the model
model = YOLO('yolov10n_blood.pt')

# Perform inference
results = model('path/to/blood/image.jpg')

# Visualize results
results.show()
```

## 🔍 Training Details
- **Base Weights**: YOLOv10n pre-trained weights
- **Fine-tuning Dataset**: BCCD Dataset
- **Training Environment**: L4 GPU

## 📦 Model Weights
- Available for download: [Model Link](https://github.com/Vishwa2684/finetuned-yolov10n-for-bloodcell-detection/blob/main/model/yolov10n_blood.onnx)

## 🙌 Acknowledgements
- [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- [YOLOv10 Repository](https://github.com/THU-MIG/yolov10)

## 🐛 Issues and Contributions
Please report any issues or submit pull requests to the project repository.