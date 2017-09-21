import json
import h5py
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from keras.models import Model

def image_preprocess(img_path):
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)

    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through thenetwork
    image = np.expand_dims(image, axis=0)

    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    image = preprocess_input(image)
    return image

base_model = Xception(weights='imagenet',input_shape=(299, 299, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


dataset = {}
with open('/home/ubuntu/data/data_processed/data_prepro.json','r') as data_file:
    data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]
N = len(dataset['unique_img_train'])
features_train = np.zeros((N,2048))
for i,img_path in enumerate(dataset['unique_img_train']):
    image = image_preprocess(img_path)
    features_train[i] = model.predict(image)[0]

M = len(dataset['unique_img_test'])
features_test = np.zeros((M,2048))
for j,img_path in enumerate(dataset['unique_img_test']):
    image = image_preprocess(img_path)
    features_test[j] = model.predict(image)[0]

with h5py.File('/home/ubuntu/data/data_processed/data_img.h5', 'w') as h:
    h.create_dataset('images_train', data= features_train)
    h.create_dataset('images_test', data = features_test)


