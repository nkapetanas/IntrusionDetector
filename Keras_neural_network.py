import numpy
from keras import backend as K
from keras.layers import Dense, Masking,Flatten, Convolution1D, MaxPooling1D, Flatten, concatenate
from keras.models import Sequential
from keras.layers import Dropout, Input, SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
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
VOCAB_SIZE = 5000
MAX_LENGTH = 42 # osa einai kai ta logs
DROPOUT = 0.3
N_HIDDEN = 2 # einai to posa theloume na vgaloume
PATH = r'C:/Users/Nick/PycharmProjects/IntrusionDetector/kdd_preprocessed.csv'  # use your path



class KerasNeuralNetwork():

    def __init__(self):
        self._model = None

    def load_data(self, allFiles, vocabulary):
        labels = []
        logs = []
        '''
        Sto vocabulary kane load to dictionary poy tha ftiakseis me spark, twra to afhsa keno apla gia na kanei compile
        '''
        # for k,v in vocabulary.items():
        #     print(k,v)
        # for f in os.listdir(dir):
        for f in allFiles:
            with open(f, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # the line below transforms the tokens of a log entry to their corresponding integer values
                    # according to the dict dictionary
                    #row_to_integer = map(vocabulary.get, row)# kathe tokean sto antistixo integer
                    print(row)
                    row_to_integer = list(map(vocabulary.get, row[0].split(",")))# kathe tokean sto antistixo integer
                    print(row_to_integer)
                    logs.append(row_to_integer[:-1])
                    labels.append(row_to_integer[:-1])
        print(logs)
        print(labels)
        return logs, labels


    def model(self):

        # define the model
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, 8, input_length=MAX_LENGTH))
        model.add(Dropout(DROPOUT))
        # blstm tou ilia auto poy epistrefei prepei na einai dio output dld concatenated
        model.add(Flatten())
        model.add(Dropout(DROPOUT))
        model.add(Dense(N_HIDDEN, activation='sigmoid'))

        return model

    def train(self, vocabulary):


        allFiles = glob.glob(PATH + "/*.csv")
        logs, labels = self.load_data(allFiles, vocabulary)
        # split into input (X) and output (Y) variables
        #X = dataset[:, 0:8]
        #Y = dataset[:, 8]
        # we split our data as batches
        model = self.model()
        batches = len(logs)//BATCH_SIZE
        training_logs = np.array_split(logs, batches)
        training_labels = np.array_split(labels, batches)

        # call the model for training
        # compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        print(model.summary())
        # fit the model
        model.fit(training_logs, training_labels, epochs=50, verbose=0) # kane to modelo save
        # evaluate the model
        loss, accuracy = model.evaluate(training_logs, training_labels, verbose=0)
        print('Accuracy: %f' % (accuracy*100))

    def test(self):

    # tha paroume to modelo apo panw tou fit kai tha tou perasoume ta test dedomena test_logs kai test labels
    # edw pali to idio me panw alla adi na kaleseis to train operation toy grafou
    # kaleis to test
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



