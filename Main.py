import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.python.ops.gen_array_ops import shape
import pickle

#the complete dataset
#each row represents an image
df = pd.read_csv("digitdata.csv")

#labels: what each row represents. what we want to predict
y = df['label']

#one hot encoding
#the labels(categories) are nominal for digit recognition.
y = to_categorical(y, num_classes=10)


#input features - matrix
#the drop function creates a new dataset from the old one. label is what were dropping and axis=1 specifies label is a column.
#its important to know that pd.read_csv automatically assumes first row is column names(headers), so its not a part of the data
X = df.drop('label', axis=1)

#normalize the dataset. Its easier to process numbers between 0-1 so divide each value in X by 255
X = X.values / 255
X = X.reshape(-1, 28, 28, 1)

#divide the dataset into 2 sets
#trainng set 80% - teach the model patterns
#test set 20% - final evaluation gives a true performance estimate of the model

# from sklearn.model_selection import train_test_split splits the data for us
#fucntion train_test_split returns training features, testing features, training labels, and testing labels in that order.
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42)


#create the neural network model
#data flow: Image (28×28) → Flatten to 784 numbers → Process through 128 neurons → Process through 64 neurons → Get 10 probability scores → Highest score = predicted digit
model = Sequential([ #the model is sequential, meaning it processes data one after the other in order. [...] consists of all the layers
    Input(shape=(28,28,1)), #defiines the input and the format 28,28 is 28x28 image and 1 means its grayscale(black&white)
    Flatten(), #Flatten layer: takes 28x28 (2D) and makes it 784 (1D). Flattens it
    Dense(128,activation='relu'), #first hidden layer. each 784 connects to all 128 neurons. 128 neurons.
    Dense(64, activation='relu'),#2nd input layer. 64 neurons. narrows down the neurons
    Dense(10, activation='softmax'), #output layer. 10 neurons. open for each digit 0-9. Converts outputs to probabilities that sum to 1.0
])

#compiling the model. Telling it how to learn
#adam: algorithm that adjusts weight during training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#train the model
history = model.fit(X_train, y_train,epochs=10,batch_size=32,validation_data=(X_val,y_val))
#history stores the results of the training(loss and accuracy for each epoch) so you can analyze or visualize the models learning process.
#model.fit is a method that actually trains the neural net by showing it examples and adjusting weights
#An epoch is one complete pass through the entire dataset. With 10 epochs the model sees all the training images 10 times (each time, the it gets better at recognizing patterns.
#batch size: instead of processing all images at once, the model processes them in  small groups of 32. This makes training faster and more memory efficient.
#validation_data: after each epoch the model tests itself on this separate validation set (shows how model handles new unseen data)

#evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val) #model.evaluate returns two values(loss and accuracy). Lower the loss the better. Higher the accuracy the better.
#loss is a penalty score calculated by categorical_crossentropy. 0 is a perfect score
#val_accuracy is a decimal showing what percentage of predictions were correct.
print("Validation accuracy: " + str(val_accuracy * 100) + "%")

#save the model
model.save('model.h5')
#save the metrics
metrics = {'val_loss': val_loss, 'val_accuracy': val_accuracy}
with open('evaluation_metrics.pkl', 'wb') as file:
    pickle.dump(metrics, file)



#graph 1
plt.figure(figsize=(10, 6))
plt.title('Model Accuracy for Training and Validation sets')
plt.plot(history.history['accuracy'], label='Training Accuracy') # This is line 1, the training accuracy line(shows the accuracy of the training data after each epoch): ['accuracy'] gets the list of accuracy values for each epoch. label labels the name to Training Accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  #This is line 2(shows the accuracy of the validation set after each epoch): ['val_accuracy'] gets the validation history for each epoch.
plt.legend() #creates a legend box for which line is which
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy.png')
plt.show() #displays the window

#graph 2
plt.figure(figsize=(10, 6))
plt.title('Model Loss for Training and Validation sets')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('lost.png')
plt.show()



#now lets make predictions on new data
test_data = pd.read_csv("test.csv")
X_test = test_data.values / 255.0
X_test = X_test.reshape(-1,28,28,1)
predictions = model.predict(X_test) #predictions holds a 2D array. Each row is the probability for each digit ex: [0.01, 0.02, 0.85....] where each index is a digit it can be.
#predicitions holds the output values from the output layer of the model. Dense(10, activation='softmax'). It captures what the output layer produces
predicted_labels = np.argmax(predictions, axis=1) #gets the max of each row that the output layer returns



