import os
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import pandas as pd
import yaml

#######################################################################
# SETTING
#######################################################################

data_root = "C:\Data\Dev-Data\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")
model_path = data_root+"model\\"
img_path = data_root+"notes\\"
input_shape_images = (255,50,3)

#######################################################################
# LOAD MODEL
print ("LOAD MODEL")
#######################################################################

# load json and create model
json_file = open(model_path + 'model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# load weights into new model
model.load_weights(model_path + "model.h5")

# load model columns
train = pd.read_csv(model_path + "model-columns.csv")

print("--- Loaded model from disk")

#######################################################################
# PREDICT ON IMAGES
print ("PREDICT ON IMAGES")
#######################################################################

filenames = os.listdir(img_path)

for i in range(0, 10):
    filename = filenames[i]
    print(filename)
    img = image.load_img(img_path + filename, target_size=input_shape_images)
    img = image.img_to_array(img)
    img = img / 255
    img = img.reshape(1, 255, 50, 3)

    classes = np.array(train.columns[3:])
    proba = model.predict(img)
    top_3 = np.argsort(proba[0])[:-4:-1]
    for i in range(3):
        print("-- {}".format(classes[top_3[i]]) + " ({:.3})".format(proba[0][top_3[i]]))
