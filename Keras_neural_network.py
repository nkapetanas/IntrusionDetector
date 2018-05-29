import numpy
from keras import backend as K
import keras as k
from keras.layers import Dense, Masking, Flatten, Convolution1D, MaxPooling1D, Flatten, concatenate
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
                    labels.append(row_to_integer[-1])
        print(logs)
        """
        
        labels kai opou dei 28166 na valei 0 enw ean einai kati allo vale to 1

        
        """
        return logs, labels


    def model(self):

        # define the model

        # Sentence Input (word embeddings) Layer


        #model.add(Embedding(VOCAB_SIZE, 80 ))
        #model.add(Dropout(DROPOUT))

        # word_embeddings = Embedding(VOCAB_SIZE, 80)
        #word_embeddings = Input(shape=(VOCAB_SIZE,80), name='input_layer_words')
        word_embeddings = Input(shape=(41,), name='input_layer_words')

        embedding_layer = Embedding(input_dim= VOCAB_SIZE, output_dim = 80, input_length=42)(word_embeddings)

        # Droupout over word embegdings
        noise_log_embeddings = Dropout(DROPOUT_RATE)(embedding_layer)

        #sentence_embedding = Dropout(DROPOUT_RATE)(word_embeddings)

        BLSTM = k.layers.Bidirectional(k.layers.LSTM(200, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, dropout=0.3, recurrent_dropout=0.3, implementation=1,stateful=False,  return_state=False, go_backwards=False), merge_mode='concat', weights=None)(noise_log_embeddings)
        blstm_dropout = Dropout(DROPOUT_RATE)(BLSTM)
        outputs = Dense(N_HIDDEN, activation='softmax')(blstm_dropout)


        # Wrap up Distributed Sentences Network
        self._model = Model(inputs=[word_embeddings], outputs=[outputs])

        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss='binary_crossentropy', metrics=['accuracy'])

        print(self._model.summary())

        return self._model

    def train(self, vocabulary):

        allFiles = glob.glob(PATH + "/*.csv")

        logs, labels = self.load_data(allFiles, vocabulary)

        model = self.model()
        # we split our data as batches

        batches = len(logs)//BATCH_SIZE
        training_logs = np.array_split(logs, batches)
        training_labels = np.array_split(labels, batches)

        # call the model for training
        # compile the model
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model

        # fit the model
        fit_model_result = model.fit(training_logs, training_labels, batch_size = BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE, validation_split = VALIDATION_SPLIT)

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



