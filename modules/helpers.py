import os
import random
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image

# Takes the output of the CNN as an input. Returns the category
# with the greatest probability as the answer.
def get_output_category(output):
    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    max_value_index = np.argmax(output)

    return categories[max_value_index]

# Normalizes the image and adds a batch dimension. This is helpful
# when using the model with singular images.
def preprocess_image(input_image):
    copy_image = input_image.copy()
    copy_image_array = image.img_to_array(copy_image)

    if copy_image_array.shape != (150, 150, 3):
        raise ValueError(f"Unexpected image shape: {copy_image_array.shape}. Expected shape: (150, 150, 3)")
    
    image_array = np.expand_dims(copy_image_array, axis=0)
    image_array /= 255.0
    print(image_array.shape)
    
    return image_array

# Selects a random image from the user-entered category or a random category and
# returns it
def random_image(directory_path, category_name = 'random'):
    count = 0
    chosen_file = None 
    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    if category_name == 'random':
        category_name = random.choice(categories)

    folder_path = os.path.join(directory_path, category_name)

    with os.scandir(folder_path) as images:
        for image in images:
            count += 1
            if random.randint(1, count) ==1:
                chosen_file = image.name
     
    image_path = os.path.join(folder_path, chosen_file)
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.title(image_path)
    print(f'Image shape: {image.shape}')

    return image

# Generates an image from a random class and returns its base64 encoding
def encode_image(directory_path):
    image = random_image(directory_path)
    encoded_image = base64.b64encode(image.tobytes()).decode('utf-8')
    
    return encoded_image
    
# Decodes an input base64 image and returns it in its original shape
def decode_image(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    array_from_bytes = np.frombuffer(decoded_image, dtype=np.uint8)
    reshaped_array = array_from_bytes.reshape((150, 150, 3))

    return reshaped_array



    