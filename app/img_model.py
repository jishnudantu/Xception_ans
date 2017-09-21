import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model 
def img_model():
	base_model = Xception(weights='imagenet',input_shape=(299, 299, 3))
	return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)