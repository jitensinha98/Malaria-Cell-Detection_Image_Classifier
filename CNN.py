import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from sklearn.utils import shuffle
from tqdm import tqdm

train_path_infected = 'Train/Parasitized'
train_path_normal = 'Train/Uninfected'

img_size = 50 
n_classes = 2
batch_size = 64
n_epochs = 25

training_data = []

label_infected = 'Parasitized'
label_uninfected = 'Uninfected'

def img_label(label):
	if label == 'Parasitized' : return [1,0]
	elif label == 'Uninfected' : return [0,1]
	
for img_file1 in tqdm(os.listdir(train_path_infected)):
	label1=label_infected
	img_infected  = os.path.join(train_path_infected,img_file1)
	img_label1 = img_label(label1)
	img1 = cv2.resize(cv2.imread(img_infected,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
	training_data.append([np.array(img1),np.array(img_label1)])

for img_file2 in tqdm(os.listdir(train_path_normal)):
	label2=label_uninfected
	img_normal  = os.path.join(train_path_normal,img_file2)
	img_label2 = img_label(label2)
	img2 = cv2.resize(cv2.imread(img_normal,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
	training_data.append([np.array(img2),np.array(img_label2)])

shuffle(training_data,random_state=23)

train_data=training_data[:-500]
test_data=training_data[-500:]

train_X=np.array([i[0] for i in train_data]).reshape(-1,img_size,img_size,1)
train_Y=np.array([i[1] for i in train_data])

test_X=np.array([i[0] for i in test_data]).reshape(-1,img_size,img_size,1)
test_Y=np.array([i[1] for i in test_data])

def Conv_net(input_shape):
	model=Sequential()
	model.add(Conv2D(12,(10,10),padding='same',activation='relu',input_shape=input_shape))
	model.add(Conv2D(8,(6,6),padding='same',activation='relu'))
	model.add(Conv2D(5,(3,3),padding='same',activation='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(12,(10,10),padding='same',activation='relu'))
	model.add(Conv2D(8,(6,6),padding='same',activation='relu'))
	model.add(Conv2D(5,(3,3),padding='same',activation='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(12,(10,10),padding='same',activation='relu'))
	model.add(Conv2D(8,(6,6),padding='same',activation='relu'))
	model.add(Conv2D(5,(3,3),padding='same',activation='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(n_classes,activation='softmax'))

	return model

model=Conv_net(train_X.shape[1:])	

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
training_model = model.fit(train_X, train_Y,batch_size=batch_size,epochs=n_epochs,verbose=1)

score = model.evaluate(test_X, test_Y, verbose=0)

print("Test Loss = ",score[0])
print("Test Accuracy = ",score[1])

model.save('Saved_Model/Malaria_cells_Classifier.h5')


