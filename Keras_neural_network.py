import csv
import glob
import pickle
from random import randint, random
import os
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_generator import DataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Reshape, TimeDistributed, Input
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, Model,model_from_json
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_curve
from attention import Attention

##--PATHS--#
INPUT_DIR_TRAIN = "data_test/"
INPUT_DIR_TEST = "test.csv"
VOCABULARY_FREQUENCIES = "word_frequencies.pickle"
VOCABULARY = "vocabulary.pickle"
MODEL_PATH = "kdd_models/model_attention_keras.json"
MODEL_PATH_WEIGHTS = "kdd_models/weights_attention.best.hdf5"

##-MODEL PARAMETERS--##
BATCH_SIZE = 128
NB_EPOCH = 2
VERBOSE = 1
VALIDATION_SPLIT = 0.2
VOCAB_SIZE = 30887
DROPOUT_RATE = 0.3
#PATH = r'C:/Users/Nick/PycharmProjects/IntrusionDetector/kdd_preprocessed.csv'
lr = 0.000001

# MODEL DIMENSIONS
TIME_STEPS = 50
EMB_DIMENSION = 40
NUM_OF_LOG_FIELDS = 41

train_examples  = 4200014+300021
val_examples  = 300021 #600002
test_examples = 311030


class KerasNeuralNetwork():

    def __init__(self):
        self._model = None

    def write_data(self, allFiles, vocabulary, dictionaryOfFrequencies, training = True, time_data = False, num_of_logs=TIME_STEPS):
        
        labels = []
        logs = []
        
        previous_logs = [[4044] * 41] * num_of_logs
        if time_data : previous_labels = [0] * num_of_logs
        id = 0
        
        for f in os.listdir(allFiles):
            _ = open(allFiles+f, 'r')
            row_count = len(_.readlines())
            _.close()
            
            with open(allFiles+f, 'r') as csvfile:
                
                reader = csv.reader(csvfile)
                r = 0

                for row in reader:

                    if (trainning):
                        randomIndex = randint(0, 39)
                        log = row[0].split(",")
                        if random() > (1 / int(dictionaryOfFrequencies.get(log[randomIndex]))):
                            log[randomIndex] = "unknown"

                    else:
                        log = row[0].split(",")
                        for n, token in enumerate(log):
                            if (vocabulary.get(token) == None):
                                log[n] = 'unknown'


                    current_log = list(map(vocabulary.get, log))
                    previous_logs.append(current_log[:-1])
                    previous_logs.pop(0)
                    logs.append(np.array(previous_logs).flatten())
                    
                    #previous_labels.pop(0)

                    if (current_log[-1] == vocabulary.get("normal.")):
                        if time_data :
                            previous_labels.pop(0)
                            previous_labels.append(0)
                            labels.append(np.array(previous_labels).flatten())
                        else:   
                            labels.append(0)
                    else:
                        if time_data :
                            previous_labels.pop(0)
                            previous_labels.append(1)
                            labels.append(np.array(previous_labels).flatten())
                        else:   
                            labels.append(1)                  
                    
                    r = r+1
                    
                    if(len(labels)>300000 or r == row_count-1):
                        with open(str(id)+"kdd_indexed_test.csv", 'w') as myfile:
                            wr = csv.writer(myfile, dialect='excel')
                            wr.writerows(logs[:])
                            
                        with open(str(id)+"labels_test.npy", 'w') as myfile:
                            np.save(myfile, labels)
                        print("Wrote "+str(id)+"kdd_indexed_test.csv")
                        logs = []
                        labels = []
                        id = id+1
                        if (r == row_count-1): r = 0


 
 
    def model(self):
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, EMB_DIMENSION, input_length=(NUM_OF_LOG_FIELDS * TIME_STEPS)))
        model.add(Reshape((TIME_STEPS, (EMB_DIMENSION * NUM_OF_LOG_FIELDS)),
                          input_shape=((NUM_OF_LOG_FIELDS * TIME_STEPS), EMB_DIMENSION)))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Bidirectional(LSTM(EMB_DIMENSION * NUM_OF_LOG_FIELDS, return_sequences=True)))
        model.add(Dropout(DROPOUT_RATE))
        model.add(TimeDistributed(Dense(1000, activation='softmax')))
        model.add(TimeDistributed(Dense(500, activation='softmax')))
        model.add(TimeDistributed(Dense(2, activation='softmax')))
 
        model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                      loss="binary_crossentropy",
                      metrics=['binary_accuracy'])
        print(model.summary())
 
        return model
     



    def model_attention(self):
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, EMB_DIMENSION, input_length=(NUM_OF_LOG_FIELDS * TIME_STEPS)))
        model.add(Reshape((TIME_STEPS, (EMB_DIMENSION * NUM_OF_LOG_FIELDS)),
                          input_shape=((NUM_OF_LOG_FIELDS * TIME_STEPS), EMB_DIMENSION)))
        model.add(Dropout(DROPOUT_RATE))
        
        model.add(Bidirectional(LSTM(EMB_DIMENSION * NUM_OF_LOG_FIELDS, return_sequences=True)))
        model.add(Attention())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                      loss="binary_crossentropy",
                      metrics=['binary_accuracy'])
        print(model.summary())

        return model
    


    def train(self, vocabulary, dictionaryOfFrequencies, model):
        
        training_generator = DataGenerator('0 1 2 3 4 5 6 7 8 9 10 11 12 13 14'.split(),batch_size=BATCH_SIZE)
        validation_generator = DataGenerator('15'.split(),batch_size=BATCH_SIZE)


        # checkpoint
        filepath = MODEL_PATH_WEIGHTS
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]

        print("Start model training")
        # fit the model
        fit_model_result = model.fit_generator(generator = training_generator.flow_from_directory(),
                                                validation_data = validation_generator.flow_from_directory(),
                                                steps_per_epoch=train_examples/BATCH_SIZE,validation_steps=val_examples/BATCH_SIZE,epochs=NB_EPOCH)

        # serialize model to JSON
        print("Saving model to disk.")
        model_json = model.to_json()
        with open(MODEL_PATH, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(MODEL_PATH_WEIGHTS)
        print("Saved model to disk")

        return model
    
    
    

    def test(self, vocabulary, dictionaryOfFrequencies):

        #all_files = glob.glob(INPUT_DIR_TEST)
        
        test_generator = DataGenerator('0 1'.split(),batch_size=BATCH_SIZE)
        loaded_model = self.load_saved_model()

        precision = self.as_keras_metric(tf.metrics.precision)
        recall = self.as_keras_metric(tf.metrics.recall)

        # evaluate loaded model on test data
        loaded_model.compile(loss=self.weighted_categorical_crossentropy([0.3, 0.8]), optimizer=Adam(),
                             metrics=['accuracy'] +
                                     [precision, recall] +
                                     [self.precision_threshold(i) for i in np.linspace(0.1, 0.9, 9)] +
                                     [self.recall_threshold(i) for i in np.linspace(0.1, 0.9, 9)])

        metrics = loaded_model.evaluate_generator(test_generator.flow_from_directory(), steps = test_examples/BATCH_SIZE, verbose=VERBOSE)
        print('Accuracy: %f' % (metrics[1] * 100))
        print('F1: %f' % (self.f1_measure(metrics[3], metrics[2]) * 100))
        print('Recall: %f' % (metrics[2] * 100))
        print('Precision: %f' % (metrics[3] * 100))

        print("Precision over different thresholds")
        print(metrics[4:12])
        print("Recall over different thresholds")
        print(metrics[13:])


        predict = loaded_model.predict_generator(test_generator.flow_from_directory(), steps = test_examples/BATCH_SIZE, verbose=VERBOSE)

        with open('output.txt', 'w') as f:
            for _list in predict:
                for _string in _list:
                    f.write(str(_string) + '\n')

  


    def as_keras_metric(self, metric):
        import functools

        @functools.wraps(metric)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = metric(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value

        return wrapper
    

    def load_saved_model(self):
        # load json and create model
        json_file = open(MODEL_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODEL_PATH_WEIGHTS)
        print("Loaded model from disk")

        return loaded_model

    def f1_measure(self, precision, recall):
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

    def weighted_categorical_crossentropy(self, weights):
        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss
    
    
    def main(self):
        with open(VOCABULARY, 'rb') as handle:
            vocabulary = pickle.load(handle)
        print("VOCABULARY SIZE:")
        print(len(vocabulary))

        with open(VOCABULARY_FREQUENCIES, 'rb') as handle:
            dictionaryOfFrequencies = pickle.load(handle)
        
        self.load_data("data_1/", vocabulary, dictionaryOfFrequencies, True, num_of_logs=TIME_STEPS)
        #self.train(vocabulary, dictionaryOfFrequencies)
        #self.test(vocabulary, dictionaryOfFrequencies)


if __name__ == '__main__':
    nn = KerasNeuralNetwork()
    nn.main()
