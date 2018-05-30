import numpy
from keras import backend as K
import keras as k
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Masking, Flatten, Convolution1D, MaxPooling1D, Flatten, concatenate, LSTM
from keras.models import Sequential
from keras.layers import Dropout, Input, SpatialDropout1D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import glob
import sys, os
import numpy as np
import csv
import pickle
from keras.layers.embeddings import Embedding
from itertools import islice



DATA_DIR = "directory which contain csv files with kdd preprocessed data"
DICTIONARY = "directory which contains the token:interger_code dictionary"
MODEL_DIRECTORY = "directory with the neural network model"
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
NB_EPOCH = 50
VERBOSE = 1
VALIDATION_SPLIT = 0.2
VOCAB_SIZE = 30886
MAX_LENGTH = 42 # osa einai kai ta logs
DROPOUT_RATE = 0.3
N_HIDDEN = 2 # einai to posa theloume na vgaloume
PATH = r'C:/Users/Nick/PycharmProjects/IntrusionDetector/kdd_preprocessed.csv'  # use your path
lr=0.001



class KerasNeuralNetwork():

    def __init__(self):
        self._model = None

    def load_data(self, allFiles, vocabulary):
        labels = []
        logs = []

        for f in allFiles:
            with open(f, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # the line below transforms the tokens of a log entry to their corresponding integer values
                    # according to the dict dictionary
                    #row_to_integer = map(vocabulary.get, row)# kathe tokean sto antistixo integer
                    row_to_integer = list(map(vocabulary.get, row[0].split(",")))# kathe tokean sto antistixo integer
                    logs.append(row_to_integer[:-1])
                    if(row_to_integer[-1] == "28166"):
                        labels.append(np.array([1,0]))
                    else:
                        labels.append(np.array([0, 1]))

        return logs, labels


    def model(self):

        # define the model
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, 80, input_length=41))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Bidirectional(LSTM(41, activation='tanh', use_bias=True)))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    def train(self, vocabulary):

        allFiles = glob.glob(PATH + "/*.csv")

        logs, labels = self.load_data(allFiles, vocabulary)

        model = self.model()
        # we split our data as batches

        batches = len(logs)//BATCH_SIZE
        training_logs = np.array(logs)
        training_labels = np.array(labels)

        # checkpoint
        filepath = "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,save_weights_only = False, mode='auto',period = 1)
        callbacks_list = [checkpoint]

        # fit the model
        fit_model_result = model.fit(training_logs, training_labels, batch_size = batches, epochs = NB_EPOCH, verbose = VERBOSE,callbacks=callbacks_list, validation_split = VALIDATION_SPLIT)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        return model

    def test(self, model):

        all_files = glob.glob("KDD_Test_Data_Preprocessed.csv/*.csv")

        test_logs, test_labels = None #self.load_data(all_files, vocabulary)

        model = self.model()

        # evaluate the model
        loss, accuracy = model.evaluate(test_logs, test_labels, verbose=VERBOSE)
        print('Accuracy: %f' % (accuracy * 100))

        return

    def main(self):
        with open('vocabulary.pickle', 'rb') as handle:
            vocabulary = pickle.load(handle)

            self.train(vocabulary)

# load the preprocessed KDD dataset
# dataset = numpy.loadtxt("kdd_preprocessed.csv", delimiter=",")


if __name__ == '__main__':
   nn = KerasNeuralNetwork()
   nn.main()



