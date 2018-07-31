import numpy as np
import keras
import csv


class DataGenerator():
    'Generates data for Keras'
    def __init__(self, IDs ,batch_size=128, n_classes=2):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.IDs = IDs


    def flow_from_directory(self):
        #classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            X = []
            Y = []
            inputs = np.empty(shape=())
            targets = np.empty(shape=())
            for id in self.IDs:
                print(id)
                with open('data/'+id+'kdd_indexed.csv', 'r') as csvfile:
                    X = []
                    Y = []
                    X = list(csv.reader(csvfile))
                    Y = keras.utils.to_categorical(np.load('data/' + id + 'labels.npy'), num_classes=self.n_classes)
                    print("File "+id+" loaded.. Batching starts..") 
                    b = 0 
                    inputs = np.empty(shape=())
                    targets = np.empty(shape=())
                    for i in xrange(0, len(X), self.batch_size): 
                        inputs = np.array(X[i:i + self.batch_size])
                        targets = np.array(Y[i:i + self.batch_size])
                        if b%200==0: print("I'm in batch "+str(b))
                        b = b+1
                        yield inputs, targets