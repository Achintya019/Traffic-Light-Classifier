import os
import cv2
import albumentations as A

# Define augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=10, p=0.9),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
    A.HorizontalFlip(p=0.5)
])

# Path to the folder containing images
input_folder = '/home/achintya-trn0175/Downloads/datacopy/red'
output_folder = '/home/achintya-trn0175/Downloads/datacopy/augmented'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB (Albumentations uses RGB)

        # Perform augmentation
        augmented = transform(image=image)
        augmented_image = augmented['image']

        # Convert augmented image back to BGR format for OpenCV if needed
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        # Save the augmented image
        output_path = os.path.join(output_folder, f'augmented_{filename}')
        cv2.imwrite(output_path, augmented_image)

print('Augmentation complete.')
