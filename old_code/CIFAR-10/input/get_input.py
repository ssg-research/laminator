from PIL import Image
import numpy as np



training_images = np.load("../test_data/test_images.npy")
last_image = training_images[-1]  # Get the last image
last_image_pil = Image.fromarray(last_image.astype('uint8'), 'RGB')  # Ensure data type conversion and specify RGB mode

# Save the image to a file
last_image_pil.save("last_image.png")