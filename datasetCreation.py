import os
import random
import shutil

frames_dir = "/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/frames"
annotations_dir = "/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/Annotations"
dataset_dir = "/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/Dataset"

images_train = os.path.join(dataset_dir, "images/train")
images_val = os.path.join(dataset_dir, "images/val")
labels_train = os.path.join(dataset_dir, "labels/train")
labels_val = os.path.join(dataset_dir, "labels/val")

os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

images = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]

random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

# TRAIN DATA
for img in train_images:
    label = img.replace(".jpg", ".txt")

    img_path = os.path.join(frames_dir, img)
    label_path = os.path.join(annotations_dir, label)

    if os.path.exists(label_path):
        shutil.copy(img_path, images_train)
        shutil.copy(label_path, labels_train)

# VALIDATION DATA
for img in val_images:
    label = img.replace(".jpg", ".txt")

    img_path = os.path.join(frames_dir, img)
    label_path = os.path.join(annotations_dir, label)

    if os.path.exists(label_path):
        shutil.copy(img_path, images_val)
        shutil.copy(label_path, labels_val)

print("Dataset organized successfully!")