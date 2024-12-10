import os
import xml.etree.ElementTree as ET
import shutil

def prepare_dataset(input_image_dir, input_annotation_dir, output_dir, train_file, test_file, val_file=None):
    """
    Prepares the dataset by organizing images and annotations into train, test, and optionally validation sets in YOLO format.

    Args:
    - input_image_dir (str): Path to the directory containing original images.
    - input_annotation_dir (str): Path to the directory containing XML annotation files.
    - output_dir (str): Directory where the prepared dataset will be stored.
    - train_file (str): File containing names of training images (without extension).
    - test_file (str): File containing names of testing images (without extension).
    - val_file (str): File containing names of validation images (optional, without extension).
    """
    # Define output directories for images and labels
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    test_image_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')
    val_image_dir = os.path.join(output_dir, 'val', 'images')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')

    # Create output directories
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    if val_file:
        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

    # Load train, test, and validation image names
    def load_image_list(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() + '.jpg' for line in file.readlines()]

    train_images = load_image_list(train_file)
    test_images = load_image_list(test_file)
    val_images = load_image_list(val_file) if val_file else []

    # Class mapping
    class_map = {'WBC': 0, 'RBC': 1, 'Platelets': 2}

    # Process images and annotations
    def process_images(image_list, image_dir, label_dir):
        for img_filename in image_list:
            src_image_path = os.path.join(input_image_dir, img_filename)
            dst_image_path = os.path.join(image_dir, img_filename)

            # Check if image exists and copy it
            if not os.path.exists(src_image_path):
                print(f"Warning: Image {img_filename} not found. Skipping.")
                continue
            shutil.copy(src_image_path, dst_image_path)

            # Parse XML annotation
            xml_filename = img_filename.replace('.jpg', '.xml')
            xml_path = os.path.join(input_annotation_dir, xml_filename)

            if not os.path.exists(xml_path):
                print(f"Warning: No annotation found for {img_filename}. Skipping.")
                continue

            # Convert XML to YOLO format
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = os.path.join(label_dir, label_filename)

            with open(label_path, 'w') as label_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = class_map.get(class_name)

                    if class_id is None:
                        print(f"Warning: Unknown class {class_name} in {img_filename}. Skipping object.")
                        continue

                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # Convert to YOLO format
                    x_center = (xmin + xmax) / (2 * img_width)
                    y_center = (ymin + ymax) / (2 * img_height)
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # Write to label file
                    label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Process train, test, and validation datasets
    process_images(train_images, train_image_dir, train_label_dir)
    process_images(test_images, test_image_dir, test_label_dir)
    if val_file:
        process_images(val_images, val_image_dir, val_label_dir)

    print("Dataset preparation completed.")

def main():
    # Define paths
    input_image_dir = './BCCD_Dataset/BCCD/JPEGImages'
    input_annotation_dir = './BCCD_Dataset/BCCD/Annotations'
    output_dir = './dataset'
    train_file = './BCCD_Dataset/BCCD/ImageSets/Main/train.txt'
    test_file = './BCCD_Dataset/BCCD/ImageSets/Main/test.txt'
    val_file = './BCCD_Dataset/BCCD/ImageSets/Main/trainval.txt'  # Optional: List of validation image names

    # Prepare dataset
    prepare_dataset(input_image_dir, input_annotation_dir, output_dir, train_file, test_file, val_file)

if __name__ == '__main__':
    main()