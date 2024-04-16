import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron 
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

path_prefix = "/scratch/liu.jasm/"


###################
## LOAD THE DATA ##
###################

hemmorrhage_labels_df = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/hemorrhage-labels.csv")

# Flagged data
f = open(path_prefix + "Hemorrhage Segmentation Project/flagged.txt", "r")
flagged = [file.replace('\n','') for file in f.readlines()]
flagged += ['ID_6431af929.jpg']
flagged

# Segmented Epidural Data
epidural_results = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/Results_Epidural.csv")
epidural_results = epidural_results[(epidural_results['Majority Label'].str.len() != 0) & (epidural_results['Correct Label'].notna())]
segmented_epidural_images = epidural_results['Origin'].values
epidural_max_contrast = []
for dirname, _, filenames in os.walk(path_prefix + 'epidural/brain_window'):
    for filename in filenames:
        if (filename in segmented_epidural_images) & (filename not in flagged):
            epidural_max_contrast.append(os.path.join(dirname, filename))
        if (len(epidural_max_contrast)==300):
            break
    else:
        continue
    break

epidural_data = []
epidural_label = []
for path in epidural_max_contrast:
    img = Image.open(path)
    img = img.resize((300,300))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (300,300,3)) & (len(label)==1)):
        epidural_data.append(np.array(img))
        epidural_label.append(label.values.flatten().tolist()[2:])

epidural_data = np.array(epidural_data)
print(epidural_data.shape)
epidural_label = np.array(epidural_label)
print(epidural_label.shape)


# Segmented Intraparenchymal Data
intraparenchymal_results = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/Results_Intraparenchymal.csv")
intraparenchymal_results = intraparenchymal_results[(intraparenchymal_results['Majority Label'].str.len() != 0) & (intraparenchymal_results['Correct Label'].notna())]
segmented_intraparenchymal_images = intraparenchymal_results['Origin'].values
intraparenchymal_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'intraparenchymal/brain_window'):
    for filename in filenames:
        if (filename in segmented_intraparenchymal_images) & (filename not in flagged):
            intraparenchymal_max_constrast.append(os.path.join(dirname, filename))
        if (len(intraparenchymal_max_constrast)==300):
            break
    else:
        continue
    break

intraparenchymal_data = []
intraparenchymal_label = []
for path in intraparenchymal_max_constrast:
    img = Image.open(path)
    img = img.resize((300,300))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (300,300,3)) & (len(label)==1)):
        intraparenchymal_data.append(np.array(img))
        intraparenchymal_label.append(label.values.flatten().tolist()[2:])

intraparenchymal_data = np.array(intraparenchymal_data)
print(intraparenchymal_data.shape)
intraparenchymal_label = np.array(intraparenchymal_label)
print(intraparenchymal_label.shape)


# Segmented Subarachnoid Data
subarachnoid_results = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/Results_Subarachnoid.csv")
subarachnoid_results = subarachnoid_results[(subarachnoid_results['Majority Label'].str.len() != 0) & (subarachnoid_results['Correct Label'].notna())]
segmented_subarachnoid_images = subarachnoid_results['Origin'].values
subarachnoid_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'subarachnoid/brain_window'):
    for filename in filenames:
        if (filename in segmented_subarachnoid_images) & (filename not in flagged):
            subarachnoid_max_constrast.append(os.path.join(dirname, filename))
        if (len(subarachnoid_max_constrast)==300):
            break
    else:
        continue
    break

subarachnoid_data = []
subarachnoid_label = []
for path in subarachnoid_max_constrast:
    img = Image.open(path)
    img = img.resize((300,300))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (300,300,3)) & (len(label)==1)):
        subarachnoid_data.append(np.array(img))
        subarachnoid_label.append(label.values.flatten().tolist()[2:])

subarachnoid_data = np.array(subarachnoid_data)
print(subarachnoid_data.shape)
subarachnoid_label = np.array(subarachnoid_label)
print(subarachnoid_label.shape)


# Segmented Subdural Data
subdural_results = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv")
subdural_results = subdural_results[(subdural_results['Majority Label'].str.len() != 0) & (subdural_results['Correct Label'].notna())]
segmented_subdural_images = subdural_results['Origin'].values
subdural_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'subdural/brain_window'):
    for filename in filenames:
        if (filename in segmented_subdural_images) & (filename not in flagged):
            subdural_max_constrast.append(os.path.join(dirname, filename))
        if (len(subdural_max_constrast)==300):
            break
    else:
        continue
    break

subdural_data = []
subdural_label = []
for path in subdural_max_constrast:
    img = Image.open(path)
    img = img.resize((300,300))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (300,300,3)) & (len(label)==1)):
        subdural_data.append(np.array(img))
        subdural_label.append(label.values.flatten().tolist()[2:])

subdural_data = np.array(subdural_data)
print(subdural_data.shape)
subdural_label = np.array(subdural_label)
print(subdural_label.shape)


# Segmented Intraventricular Data
intraventricular_results = pd.read_csv(path_prefix + "Set2/Results_Brain.csv")
intraventricular_results = intraventricular_results[intraventricular_results['ROI'].notna()]
segmented_intraventricular_images = intraventricular_results['Origin'].values
intraventricular_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'intraventricular/brain_window'):
    for filename in filenames:
        if (filename in segmented_intraventricular_images) & (filename not in flagged):
            intraventricular_max_constrast.append(os.path.join(dirname, filename))
        if (len(intraventricular_max_constrast)==200):
            break
    else:
        continue
    break

intraventricular_data = []
intraventricular_label = []
for path in intraventricular_max_constrast:
    img = Image.open(path)
    img = img.resize((300,300))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (300,300,3)) & (len(label)==1)):
        intraventricular_data.append(np.array(img))
        intraventricular_label.append(label.values.flatten().tolist()[2:])
        
intraventricular_data = np.array(intraventricular_data)
print(intraventricular_data.shape)
intraventricular_label = np.array(intraventricular_label)
print(intraventricular_label.shape)



# Combine all the data
all_data = np.vstack((epidural_data,intraparenchymal_data,subarachnoid_data,subdural_data,intraventricular_data))
all_label = np.vstack((epidural_label,intraparenchymal_label,subarachnoid_label,subdural_label,intraventricular_label))
print(all_data.shape)
print(all_label.shape)

# Shuffle the data
shuffle_index = np.random.permutation(all_data.shape[0])
X_data = all_data[shuffle_index]
y_data = all_label[shuffle_index]

# Split train, test, val
[X_train, X_rest, y_train, y_rest] = train_test_split(X_data, y_data, test_size = 0.2, random_state=0)
[X_val, X_test, y_val, y_test] = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

# Normalize the data
X_train = X_train / 255.
X_test = X_test / 255.
X_val = X_val / 255.

## Add an extra dimension
X_train = X_train.reshape(-1, 300, 300, 3)
X_test = X_test.reshape(-1, 300, 300, 3)
X_val = X_val.reshape(-1, 300, 300, 3)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

print(X_val.shape)
print(y_val.shape)



#####################
## TRAIN THE MODEL ##
#####################

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Define the model
model = keras.models.Sequential()


# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same', input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
#model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
#model.add(BatchNormalization())
# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
# output layer
model.add(Dense(units=5, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

print(model.summary())


# Train the model
epochs = 30

history = model.fit(X_train, 
                    y_train, 
                    epochs=epochs, 
                    #batch_size=300,
                    verbose=1,
                    validation_data=(X_test, y_test))

