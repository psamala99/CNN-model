#%% Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
 
#%% Identifying paths to Data
import glob

Tumor = glob.glob(r"C:\Users\pavan\Desktop\Git\MyWork\Brain Tumor Images Renamed\Tumor\*.*")

NoTumor = glob.glob(r"C:\Users\pavan\Desktop\Git\MyWork\Brain Tumor Images Renamed\No Tumor\*.*")


#%% Storing Data as Numpy Arrays
data = []
labels = []

for i in Tumor:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (28,28))
    image=np.array(image)
    data.append(image)
    labels.append(1)
for i in NoTumor:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (28,28))
    image=np.array(image)
    data.append(image)
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

#%%
number_no_tumors = np.count_nonzero(labels==0)
print((number_no_tumors/3762)*100)
number_tumors = np.count_nonzero(labels==1)
print((number_tumors/3762)*100)


#%%
from sklearn.model_selection import StratifiedShuffleSplit 

sss1=StratifiedShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8,random_state=42)
for train_index, test_index in sss1.split(data, labels):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.2, train_size=0.8,random_state=42)
for train_index, val_index in sss2.split(x_train, y_train):
    x_train, x_val = x_train[train_index], x_train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]   
#%%
number_no_tumors = np.count_nonzero(y_test==0)
print((number_no_tumors/len(y_test))*100)
number_tumors = np.count_nonzero(y_test==1)
print((number_tumors/len(y_test))*100)

number_no_tumors = np.count_nonzero(y_train==0)
print((number_no_tumors/len(y_train))*100)
number_tumors = np.count_nonzero(y_train==1)
print((number_tumors/len(y_train))*100)

number_no_tumors = np.count_nonzero(y_val==0)
print((number_no_tumors/len(y_val))*100)
number_tumors = np.count_nonzero(y_val==1)
print((number_tumors/len(y_val))*100)
#%% Splitting Data
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,random_state=42, shuffle=True)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
# random_state=42,  shuffle=True)
#%%


#%% Normalizing Data
x_train=x_train/255
x_val=x_val/255
x_test=x_test/255


#%% Creating CNN
import numpy as np
from tensorflow import keras
from glob import glob

model = keras.models.Sequential([
    keras.layers.Conv2D(32, 5, activation='relu', padding='same', input_shape=[28, 28, 3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 2, activation='relu', padding='same'),
    keras.layers.Conv2D(64, 2, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 2, activation='relu', padding='same'),
    keras.layers.Conv2D(128, 2, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

#%% Running Model
epochs=15
learning_rate = 0.001
decay_rate = learning_rate / epochs
momentum = 0.8
batch_size=128



model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

loss, accuracy = model.evaluate(x=x_test, y=y_test)

#%% Summarize History for Model Accuracy 
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#%% Summarize History for Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# %%
