import numpy as np
import cv2
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model 
from skimage import io

def image_features(url):
    image = np.resize(io.imread(url), (299, 299,3)).astype(float)
    print(image.shape)
    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through thenetwork
    image = np.expand_dims(image, axis=0)

    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess_input(image)
    base_model = Xception(weights='imagenet',input_shape=(299, 299, 3))
    image_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return image_model.predict(image)
