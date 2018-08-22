
# LSTM RNN for generating text predictions.

import numpy as np
import sys
import os
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Find next number in sequence for saving model.
model_list = (os.listdir('./Models/'))

biggest_num = 0

for i in range(len(model_list) - 1):
    x = model_list[i]
    x = int(x[6:-3])
    if x > biggest_num:
        biggest_num = x

model_num = biggest_num + 1

################################################################

##############
# Parameters #
##############

# Path of text to train from.
INPUT_TEXT_PATH = './Input/input.txt'
# Path where model will output to / load from.
MODEL_PATH_L = './Models/model ' + str(model_num - 1) + '.h5'
MODEL_PATH_O = './Models/model ' + str(model_num) + '.h5'
# Number of epochs to train for.
EPOCHS = 10
BATCH_SIZE = 150
# Sequence starting point.
# OUTPUT_START = 0
OUTPUT_START = 0
# Sequence length until prediction starts.
# SEQUENCE_LENGTH = 120
SEQUENCE_LENGTH = 120
# Length of prediction.
# PRED_LENGTH = 100
PRED_LENGTH = 100
# Flag for if loading model from file.
LOAD_MODEL = True
# Flag for if training model.
TRAINING = True

################################################################

# Import text to train with and set to lowercase.
print("Importing text...")

input_text = open(INPUT_TEXT_PATH).read()
input_text = input_text.lower()

# Map all unique characters to a number.
chars = sorted(list(set(input_text)))
n_to_char = {n: char for n, char in enumerate(chars)}
char_to_n = {char: n for n, char in enumerate(chars)}

X = []
Y = []
length = len(input_text)
seq_length = SEQUENCE_LENGTH

# Break text up into sequences.
for i in range(0, length - seq_length, 1):
    sequence = input_text[i:i + seq_length]
    label = input_text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

# Reshape X, scale X by character amount to improve training speed,
# and one-hot encode Y.
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(chars))
Y_modified = np_utils.to_categorical(Y)

print("Done.")
print("Unique Characters:", len(chars))

# Design model for training.
model = Sequential()
model.add(LSTM(250, input_shape=(X_modified.shape[1], X_modified.shape[2]),
          return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

if LOAD_MODEL:

    if model_num - 1 == 0:

        sys.exit("No model file to load.")

    else:

        model = load_model(MODEL_PATH_L)
        print("Model loaded from", MODEL_PATH_L)

if TRAINING:

    checkpoint = ModelCheckpoint(MODEL_PATH_O, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    cb_list = [checkpoint]
    model.fit(X_modified, Y_modified, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=cb_list)

if not TRAINING and not LOAD_MODEL:

    sys.exit("Can't generate text without training or loading model.")

print("Generating text...")

string_mapped = X[OUTPUT_START]
full_string = [n_to_char[value] for value in string_mapped]

for i in range(PRED_LENGTH):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(chars))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

text = ""

for char in full_string:
    text = text + char

print("-" * 100)
print(text)
print("-" * 100)
