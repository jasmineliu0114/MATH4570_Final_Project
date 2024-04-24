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
import seaborn as sns
from keras.models import Sequential
from PIL import Image
from keras.layers import Conv2D,Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D
#from sklearn.preprocessing import OneHotEncoder
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
for dirname, _, filenames in os.walk(path_prefix + 'epidural/max_contrast_window'):
    for filename in filenames:
        if (filename in segmented_epidural_images) & (filename not in flagged):
            epidural_max_contrast.append(os.path.join(dirname, filename))

epidural_data = []
epidural_label = []
for path in epidural_max_contrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
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
for dirname, _, filenames in os.walk(path_prefix + 'intraparenchymal/max_contrast_window'):
    for filename in filenames:
        if (filename in segmented_intraparenchymal_images) & (filename not in flagged):
            intraparenchymal_max_constrast.append(os.path.join(dirname, filename))

intraparenchymal_data = []
intraparenchymal_label = []
for path in intraparenchymal_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
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
for dirname, _, filenames in os.walk(path_prefix + 'subarachnoid/max_contrast_window'):
    for filename in filenames:
        if (filename in segmented_subarachnoid_images) & (filename not in flagged):
            subarachnoid_max_constrast.append(os.path.join(dirname, filename))

subarachnoid_data = []
subarachnoid_label = []
for path in subarachnoid_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
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
for dirname, _, filenames in os.walk(path_prefix + 'subdural/max_contrast_window'):
    for filename in filenames:
        if (filename in segmented_subdural_images) & (filename not in flagged):
            subdural_max_constrast.append(os.path.join(dirname, filename))

subdural_data = []
subdural_label = []
for path in subdural_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
        subdural_data.append(np.array(img))
        subdural_label.append(label.values.flatten().tolist()[2:])

subdural_data = np.array(subdural_data)
print(subdural_data.shape)
subdural_label = np.array(subdural_label)
print(subdural_label.shape)

# Segmented Multiple Data
multiple_results = pd.read_csv(path_prefix + "Hemorrhage Segmentation Project/Results_Multiple.csv")
multiple_results = multiple_results[(multiple_results['Majority Label'].str.len() != 0) & (multiple_results['Correct Label'].notna())]
segmented_multiple_images = multiple_results['Origin'].values
multiple_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'multi/max_contrast_window'):
    for filename in filenames:
        if (filename in segmented_multiple_images) & (filename not in flagged):
            multiple_max_constrast.append(os.path.join(dirname, filename))

multiple_data = []
multiple_label = []
for path in multiple_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
        multiple_data.append(np.array(img))
        multiple_label.append(label.values.flatten().tolist()[2:])

multiple_data = np.array(multiple_data)
print(multiple_data.shape)
multiple_label = np.array(multiple_label)
print(multiple_label.shape)

# Intraventricular Data
intraventricular_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'intraventricular/max_contrast_window'):
    for filename in filenames:
	if (filename not in flagged):
	        intraventricular_max_constrast.append(os.path.join(dirname, filename))
intraventricular_data = []
intraventricular_label = []
for path in intraventricular_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    base_filename = os.path.basename(path)
    label = hemmorrhage_labels_df[hemmorrhage_labels_df['Image']==base_filename.replace('.jpg', '')]
    if((img.shape == (512,512,3)) & (len(label)==1)):
        intraventricular_data.append(np.array(img))
        intraventricular_label.append(label.values.flatten().tolist()[2:])
        
intraventricular_data = np.array(intraventricular_data)
print(intraventricular_data.shape)
intraventricular_label = np.array(intraventricular_label)
print(intraventricular_label.shape)

# Normal Data
normal_max_constrast = []
for dirname, _, filenames in os.walk(path_prefix + 'normal/max_contrast_window'):
    for filename in filenames:
	if (filename not in flagged):
		normal_max_constrast.append(os.path.join(dirname, filename))

normal_data = []
normal_label = []
for path in normal_max_constrast:
    img = Image.open(path)
    img = img.resize((512,512))
    img = np.array(img)
    if((img.shape == (512,512,3)) & (len(label)==1)):
        normal_data.append(np.array(img))
        normal_label.append([0,0,0,0,0])
        
normal_data = np.array(normal_data)
print(normal_data.shape)
normal_label = np.array(normal_label)
print(normal_label.shape)

# Combine all the data
all_data = np.vstack((epidural_data,intraparenchymal_data,subarachnoid_data,subdural_data,multiple_data,intraventricular_data,normal_data))
all_label = np.vstack((epidural_label,intraparenchymal_label,subarachnoid_label,subdural_label,multiple_label,intraventricular_label,normal_label))
print(all_data.shape)
print(all_label.shape)

# Shuffle the data
shuffle_index = np.random.permutation(19852)
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
X_train = X_train.reshape(-1, 512, 512, 3)
X_test = X_test.reshape(-1, 512, 512, 3)
X_val = X_val.reshape(-1, 512, 512, 3)

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

model.add(Conv2D(32, (8,8),
                 padding='valid',
                 input_shape=(512, 512, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


# Train the model
epochs = 5

history = model.fit(X_train, 
                    y_train, 
                    epochs=epochs, 
                    batch_size=128,
                    verbose=1,
                    validation_data=(X_test, y_test))
