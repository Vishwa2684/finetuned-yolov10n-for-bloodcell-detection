# Blood Cell Detection using YOLOv10n

## Project Overview
This project implements a custom object detection model for identifying blood cell types using YOLOv10n, fine-tuned on the BCCD (Blood Cell Classification) Dataset.

## Dataset
- **Source**: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- **Classes Detected**: 
  - White Blood Cells (WBC)
  - Red Blood Cells (RBC)
  - Platelets

## Model Details
- **Base Model**: YOLOv10n
- **Hardware**: L4 GPU for fine-tuning

## Dataset Configuration
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

## Evaluation
Model performance was evaluated using `val.ipynb` notebook on the validation dataset.

## Deployment
The model is hosted on Hugging Face Spaces with a radio interface for interactive inference.

## How to Use
[Add instructions for using the model, loading weights, running inference, etc.]

## Requirements
- Python
- PyTorch
- YOLOv10
- [List any other specific dependencies]

## License
[Add license information]

## Acknowledgements
- Original BCCD Dataset: [GitHub Link]
- YOLOv10 Repository: [GitHub Link]