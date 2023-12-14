from torchvision import transforms
import numpy as np
from PIL import Image

def load_image_and_transform(IMAGE_PATH, transform=None, img_size=224):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Open Image
    image = Image.open(IMAGE_PATH)

    # Resize Image
    resized_image = image.resize((img_size, img_size))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Normalize the image array to be in the range [0, 1]
    rgb_img = image_array.astype(np.float32) / 255.0
    
    # Transform to tensor.
    image_tensor = transform(image)
    
    return rgb_img, image_tensor