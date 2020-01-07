import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#######################################################################
# SETTING
#######################################################################
epochs = 50

data_root = "C:\Data\Dev-Data\music\\"
if os.getenv("ROCKETMAN_DATA") is not None:
    data_root = os.getenv("ROCKETMAN_DATA")
filename_labels = data_root + "labels\\labels.csv"
model_path_target = data_root + "smodel\\"
input_shape_images = (255,50,3)

#######################################################################
# LOAD LABELS
print ("LOAD LABELS")
#######################################################################

train = pd.read_csv(filename_labels)

#######################################################################
# ONE HOT ENCODING OF CATEGORIES
# http://www.insightsbot.com/blog/zuyVu/python-one-hot-encoding-with-pandas-made-simple
print ("ONE HOT ENCODING OF CATEGORIES")
#######################################################################

train ['label'] = pd.Categorical (train['label'])
trainOneHot = pd.get_dummies (train['label'], prefix = 'category')
train = pd.concat ([train, trainOneHot], axis=1)

#######################################################################
# PREPARE TRAINING IMAGES
print ("PREPARE IMAGES")
#######################################################################

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(train['path'][i]+'\\'+train['filename'][i],target_size=input_shape_images)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
y = np.array(train.drop(['label', 'path', 'filename'],axis=1))

#######################################################################
# SPLIT TRAINING AND TEST DATA
print ("SPLIT TRAINING AND TEST DATA")
#######################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

#######################################################################
# BUILD MODEL ARCHITECTUIRE
print ("BUILD MODEL ARCHITECTURE")
#
# The problems is that the output of layer conv2d_4 became zero or negative.
# To solve this problem, you must design the network so that the input data would not be highly downsampled.
# Here are some possible solutions:
#    Use less layers. Especially remove a max-pooling layer, which downsamples a lot (by one third under this setting).
#    Use smaller max-pooling, e.g. pool_size=(2, 2), which results in downsampling by a half.
#    Use "same padding" for Conv2D layer, which results in no downsampling during the convolution step.
#######################################################################

model = Sequential()
# replace: model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=input_shape_images))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# skip:
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# replace: model.add(Dense(25, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print (model.summary ())

#######################################################################
# TRAIN MODEL
print ("TRAIN MODEL")
#######################################################################

model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=64)

#######################################################################
# SAVE MODEL TO DISK
print ("SAVE MODEL TO DISK")
#######################################################################

# serialize model to JSON
model_json = model.to_json()
with open(model_path_target+"model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_path_target+"model.h5")

# serialize model columns / labels
train.to_csv(model_path_target+"model-columns.csv", index=None, header=True)

print("--- saved model to disk")
