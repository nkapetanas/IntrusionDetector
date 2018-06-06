from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM
from keras.models import Sequential, model_from_json
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import glob
import numpy as np
import csv
import pickle
from keras.layers.embeddings import Embedding
from sklearn.metrics import fbeta_score

DATA_DIR = "directory which contain csv files with kdd preprocessed data"
DICTIONARY = "directory which contains the token:interger_code dictionary"
MODEL_DIRECTORY = "directory with the neural network model"
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
NB_EPOCH = 5
VERBOSE = 1
VALIDATION_SPLIT = 0.2
VOCAB_SIZE = 30886
MAX_LENGTH = 42  # osa einai kai ta logs
DROPOUT_RATE = 0.3
N_HIDDEN = 2  # einai to posa theloume na vgaloume
PATH = r'C:/Users/Nick/PycharmProjects/IntrusionDetector/kdd_preprocessed.csv'  # use your path
lr = 0.001
from random import randint, random


class KerasNeuralNetwork():

    def __init__(self):
        self._model = None

    def load_data(self, allFiles, vocabulary, dictionaryOfFrequencies, trainning):
        labels = []
        logs = []

        for f in allFiles:
            with open(f, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # the line below transforms the tokens of a log entry to their corresponding integer values
                    # according to the dict dictionary
                    if (trainning):
                        randomIndex = randint(0, 39)
                        log = row[0].split(",")
                        if random() > (1 / int(dictionaryOfFrequencies.get(log[randomIndex]))):
                            log[randomIndex] = "unknown"

                    if (trainning == False):
                        log = row[0].split(",")
                        for n, token in enumerate(log):
                            if (vocabulary.get(token) == None):
                                log[n] = 'unknown'

                    row_to_integer = list(map(vocabulary.get, log))
                    logs.append(row_to_integer[:-1])

                    # if(row_to_integer[-1] == 28166):
                    if (row_to_integer[-1] == 2319):
                        labels.append(np.array([1, 0]))
                    else:
                        labels.append(np.array([0, 1]))
        # with open('logs.txt', 'w') as f:
        #     for _list in logs:
        #         for _string in _list:
        #             f.write(str(_string) + '\n')
        return logs, labels

    def model(self):

        # define the model
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, 80, input_length=41))
        model.add(Dropout(DROPOUT_RATE))
        model.add(LSTM(41, activation='tanh', use_bias=True))
        # model.add(Bidirectional(LSTM(41, activation='tanh', use_bias=True)))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    def train(self, vocabulary, dictionaryOfFrequencies):

        allFiles = glob.glob(PATH + "/*.csv")

        logs, labels = self.load_data(allFiles, vocabulary, dictionaryOfFrequencies, True)

        model = self.model()
        # we split our data as batches

        batches = len(logs) // BATCH_SIZE
        training_logs = np.array(logs)
        training_labels = np.array(labels)

        # checkpoint
        filepath = "weights2.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]

        # fit the model
        fit_model_result = model.fit(training_logs, training_labels, batch_size=batches, epochs=NB_EPOCH,
                                     verbose=VERBOSE, callbacks=callbacks_list, validation_split=VALIDATION_SPLIT)

        # serialize model to JSON
        model_json = model.to_json()
        with open("model2.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model2.h5")
        print("Saved model to disk")

        return model

    def test(self, vocabulary, dictionaryOfFrequencies):

        all_files = glob.glob("KDD_Test_Data_Preprocessed.csv/*.csv")

        test_logs, test_labels = self.load_data(all_files, vocabulary, dictionaryOfFrequencies, False)

        test_logs = np.array(test_logs)
        test_labels = np.array(test_labels)

        print(test_logs)
        loaded_model = self.load_saved_model()

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy',
                                                                                    self.recall,
                                                                                    self.precision,
                                                                                    self.precision_threshold(0.1),
                                                                                    self.precision_threshold(0.2),
                                                                                    self.precision_threshold(0.3),
                                                                                    self.precision_threshold(0.4),
                                                                                    self.precision_threshold(0.5),
                                                                                    self.precision_threshold(0.6),
                                                                                    self.precision_threshold(0.7),
                                                                                    self.precision_threshold(0.8),
                                                                                    self.precision_threshold(0.9),
                                                                                    self.recall_threshold(0.1),
                                                                                    self.recall_threshold(0.2),
                                                                                    self.recall_threshold(0.3),
                                                                                    self.recall_threshold(0.4),
                                                                                    self.recall_threshold(0.5),
                                                                                    self.recall_threshold(0.6),
                                                                                    self.recall_threshold(0.7),
                                                                                    self.recall_threshold(0.8),
                                                                                    self.recall_threshold(0.9)])
        loss, accuracy, recall_r, precision_r, precision_threshold1, precision_threshold2, precision_threshold3, precision_threshold4, precision_threshold5, precision_threshold6, \
                precision_threshold7, precision_threshold8, precision_threshold9, recall_threshold1, recall_threshold2, recall_threshold3, recall_threshold4, recall_threshold5, recall_threshold6, \
        recall_threshold7, recall_threshold8, recall_threshold9  = loaded_model.evaluate(test_logs, test_labels, verbose=VERBOSE)
        print('Accuracy: %f' % (accuracy * 100))
        print('F1: %f' % (self.f1measure(precision_r, recall_r) * 100))
        print('Recall: %f' % (recall_r * 100))
        print('Precision: %f' % (precision_r * 100))

        print(precision_threshold1 )

        print("")
        print(recall_threshold1)

        print("/////////////////////////")

        print(precision_threshold2)

        print("")
        print(recall_threshold2)

        print("/////////////////////////")

        print(precision_threshold3)

        print("")
        print(recall_threshold3)

        print("/////////////////////////")

        print(precision_threshold4)

        print("")
        print(recall_threshold4)

        print("/////////////////////////")

        print(precision_threshold5)

        print("")
        print(recall_threshold5)

        print("/////////////////////////")

        print(precision_threshold6)

        print("")
        print(recall_threshold6)

        print("/////////////////////////")

        print(precision_threshold7)

        print("")
        print(recall_threshold7)

        print("/////////////////////////")

        print(precision_threshold8)

        print("")
        print(recall_threshold8)

        print("/////////////////////////")

        print(precision_threshold9)

        print("")
        print(recall_threshold9)

        print("/////////////////////////")

        # print(precision_r, true_positives, false_positives, predicted_negatives, predicted_positives)
        # predict = loaded_model.predict(test_logs, verbose=1)

        # with open('output.txt', 'w') as f:
        #     for _list in predict:
        #         for _string in _list:
        #             f.write(str(_string) + '\n')

    def main(self):
        with open('vocabulary.pickle', 'rb') as handle:
            vocabulary = pickle.load(handle)

        with open('word_frequencies.pickle', 'rb') as handle:
            dictionaryOfFrequencies = pickle.load(handle)

            # self.train(vocabulary, dictionaryOfFrequencies)
            self.test(vocabulary, dictionaryOfFrequencies)

    def precision(self, y_true, y_pred):
        # Calculates the precision
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        false_positives = predicted_positives - true_positives
        # true_negatives
        predicted_negatives = K.sum(K.round(K.clip(y_pred, 1, 0)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        # return precision, true_positives, false_positives, predicted_negatives, predicted_positives

    def recall(self, y_true, y_pred):
        # Calculates the recall
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def load_saved_model(self):

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        return loaded_model

    def f1measure(self, precision, recall):
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def precision_recall_plot(self, y_true, y_pred):
        import tensorflow as tf
        y_true = tf.keras.backend.eval(y_true)
        y_pred = tf.keras.backend.eval(y_pred)
        print(y_true)
        print(y_pred)
        precision, recall, threashold = precision_recall_curve(y_true, y_pred)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    def precision_threshold(self, threshold=0.5):
        def precision(y_true, y_pred):
            """Precision metric.
            Computes the precision over the whole batch using threshold_value. """
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
            # predicted positives = true positives + false positives
            predicted_positives = K.sum(y_pred)
            precision_ratio = true_positives / (predicted_positives + K.epsilon())
            return precision_ratio
        return precision

    def recall_threshold(self, threshold=0.5):
        def recall(y_true, y_pred):
            """Recall metric. Computes the recall over the whole batch using threshold_value."""
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
            # all positives = true positives + false negatives, false negatives einai ayta poy enw htan intrusions, ta
            all_positives = K.sum(K.clip(y_true, 0, 1))
            recall_ratio = true_positives / (all_positives + K.epsilon())
            return recall_ratio
        return recall

    # usage model.compile(..., metrics = [precision_threshold(0.1), precision_threshold(0.2),precision_threshold(0.8), recall_threshold(0.2,...)])


if __name__ == '__main__':
    nn = KerasNeuralNetwork()
    nn.main()
