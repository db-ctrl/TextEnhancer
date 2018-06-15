
#LSTM RNN for generating text predictions.

import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################### Parameters #############################
EPOCHS = 10
BATCH_SIZE = 300
SEQUENCE_LENGTH = 120
INPUT_TEXT_PATH = './input.txt'
############################################################

# Import text to train with and set to lowercase.

print("Importing text...")

input_text = open(INPUT_TEXT_PATH).read()
input_text = input_text.lower()

# Map all unique characters to a number.
chars = sorted(list(set(input_text)))

n_to_char = {n:char for n, char in enumerate(chars)}
char_to_n = {char:n for n, char in enumerate(chars)}

X = []
Y = []
length = len(input_text)
seq_length = SEQUENCE_LENGTH

for i in range(0, length - seq_length, 1):
	sequence = input_text[i:i + seq_length]
	label= input_text[i + seq_length]
	X.append([char_to_n[char] for char in sequence])
	Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(chars))
Y_modified = np_utils.to_categorical(Y)

print("Done.")
print("Unique Characters:",len(chars))

model = Sequential()
model.add(LSTM(250, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

if not os.path.isfile('./model.h5'):

	model.fit(X_modified, Y_modified, epochs=EPOCHS, batch_size=BATCH_SIZE)

	model.save_weights('./model.h5')
	print("Model saved.")

else:
	
	model.load_weights('./model.h5')
	print("Model loaded from file.")

print("Generating text...")

# string_mapped = X[np.random.randint(len(X))]
string_mapped = X[0]
full_string = [n_to_char[value] for value in string_mapped]

for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(chars))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

text=""

for char in full_string:
    text = text+char

print("-"*120)
print(text)
print("-"*120)