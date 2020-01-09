
# LSTM RNN for generating text predictions.

import numpy as np
import sys
import os
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################################


##############
# Parameters #
##############


# Training text path.
INPUT_TEXT_PATH = '/Users/david/PycharmProjects/db-Text-Generator/Input/sonnets.txt'
# Name models will save to / load from.
# Make sure to include a space after the model name. ex: 'model '
MODEL_NAME = 'SimpleModel '
# Path where model will output to / load from. Will be h5 file-type.
model_path_l = '/Users/david/PycharmProjects/db-Text-Generator/Models/'
model_path_o = '/Users/david/PycharmProjects/db-Text-Generator/Models/'
# Sequential saving toggle.
# Sequential:
# Load: Highest saved model number.
# Save: Next highest (if training).
# Non-sequential:
# Load: Specified model number.
# Save: Specified model number (if training).
SEQUENTIAL = True
# Model number to load if using non-sequential mode.
MODEL_SPEC = 0
# Number of epochs to train for.
EPOCHS = 20
BATCH_SIZE = 120
# Adam optimizer learning rate.
# LEARN_RATE = 0.001
LEARN_RATE = 0.001
# Threshold of   allowed with no loss improvement.
EARLY_STOP = 3
# Sequence starting point.
# OUTPUT_START = 0
OUTPUT_START = 1200
# Sequence length until prediction starts.
# SEQUENCE_LENGTH = 120
SEQUENCE_LENGTH = 120
# Length of prediction.
# PRED_LENGTH = 100
PRED_LENGTH = 250
# Flag for if loading model from file.
LOAD_MODEL = False
# Flag for if training model.
TRAINING = True
# Flag for if checking accuracy of model.
TEST_ACC = False

################################################################

if SEQUENTIAL:

    model_list = (os.listdir(model_path_l))

    biggest_num = 0

    for i in range(len(model_list)):
        x = model_list[i]
        if x.find(MODEL_NAME) == -1:
            continue
        st = x.find(MODEL_NAME) + len(MODEL_NAME)
        x = int(x[st:-3])
        if x > biggest_num:
            biggest_num = x

    model_num = biggest_num + 1

    model_path_l += (MODEL_NAME + str(model_num - 1) + '.h5')
    model_path_o += (MODEL_NAME + str(model_num) + '.h5')

else:

    model_path_l += (MODEL_NAME + str(MODEL_SPEC) + '.h5')
    model_path_o = model_path_l

# Import text to train with and set to lowercase.
print("Importing and sorting text...")

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
model.add(Dropout(0.1))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
adam = optimizers.Adam(lr=LEARN_RATE)
model.compile(loss='categorical_crossentropy', optimizer=adam)

if LOAD_MODEL:

    if SEQUENTIAL:

        if model_num - 1 == 0:

            sys.exit("No model file to load.")

    try:
        model = load_model(model_path_l)
    except OSError:
        sys.exit("Specified model doesn't exist.")

    print("Model loaded from", model_path_l)

if TRAINING:
    checkpoint = ModelCheckpoint(model_path_o, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    early_s = EarlyStopping(monitor='loss', patience=EARLY_STOP, verbose=1)
    cb_list = [checkpoint, early_s]
    model.fit(X_modified, Y_modified, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=cb_list)

if not TRAINING and not LOAD_MODEL:

    sys.exit("Can't generate text without training/loading model.")

if TEST_ACC:

    print("Evaluating accuracy...")
    score = model.evaluate(X_modified, Y_modified, verbose=1)
    print("Model Accuracy: %.3f" % (score * 100))

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
