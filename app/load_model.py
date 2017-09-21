from keras.models import model_from_json
import numpy as np
import h5py, json
def loadmodel():
    model_filename = 'resources/model.json'
    model_weights_filename = 'resources/model_weights_xception.h5'
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
	# print "Reading Model..."
    model = model_from_json(loaded_model_json)
	# print "Loading Weights..."
    model.load_weights(model_weights_filename)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model
