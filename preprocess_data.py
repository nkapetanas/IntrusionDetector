from __future__ import print_function

import csv
from itertools import islice
import numpy as np
import os
import pickle

from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import Bucketizer
from pyspark.sql import SQLContext
from pyspark.sql import functions as func
from pyspark.sql.types import StructField, StructType, StringType
import sys


class PreprocessData():

    def __init__(self):
        return


    # we pass a df and the field column we want to bucketize
    def bucketize(self, df, field):
        df = df.withColumn(field, df[field].cast("double"))
        max = df.agg({field: "max"}).collect()[0][0]
        min = df.agg({field: "min"}).collect()[0][0]
        stddev = df.agg({field: "stddev"}).collect()[0][0]
        number_of_buckets = 1
        if stddev != 0:
            number_of_buckets = ((max - min) // (stddev))
        buckets = np.arange(number_of_buckets, dtype=np.float).tolist()
        buckets = [-float('inf')] + buckets + [float('inf')]
        bucketizer = Bucketizer(splits=buckets, inputCol=field,
                                outputCol=field + '_bucketized')
        print("Bucketizing column: ", field)
        bucketized_features = bucketizer.transform(df)
        return bucketized_features


    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))

    def field_name_changer(self, df, field):
        def func_inner(value):
            return str(value) + field
        field_udf = func.udf(func_inner, StringType())
        return df.withColumn(field + '_enc', field_udf(field))
    
    
    
     
    def write_data(self, allFiles, training=True, time_data=False, num_of_logs=40):
        config = model_parameters.train_config()
        
        with open(config.vocabulary_dir, 'rb') as handle:
            vocabulary = pickle.load(handle)

        with open(config.frequency_dict_dir, 'rb') as handle:
            dictionaryOfFrequencies = pickle.load(handle)
            
        print("Vocabulary and dictionary of frequencies loaded.")
        
        labels = []
        logs = []
        previous_logs = [[4044] * 41] * num_of_logs
        if time_data : previous_labels = [0] * num_of_logs
        id = 0
         
        for f in os.listdir(allFiles):
            _ = open(allFiles + f, 'r')
            row_count = len(_.readlines())
            _.close()
             
            with open(allFiles + f, 'r') as csvfile:   
                reader = csv.reader(csvfile)
                r = 0
                
                for row in reader:
                    if (training):
                        randomIndex = np.random.randint(0, 39)
                        log = row[0].split(",")
                        if np.random() > (1 / int(dictionaryOfFrequencies.get(log[randomIndex]))):
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
                     
                    r = r + 1
                     
                    if(len(labels) > 50000 or r == row_count - 1):
                        with open(str(id) + "kdd_indexed.csv", 'w') as myfile:
                            wr = csv.writer(myfile, dialect='excel')
                            wr.writerows(logs[:])
                             
                        with open(str(id) + "labels.npy", 'w') as myfile:
                            np.save(myfile, labels)
                        print("Wrote " + str(id) + "kdd_indexed.csv")
                        logs = []
                        labels = []
                        id = id + 1
                        if (r == row_count - 1): r = 0
       


if __name__ == '__main__':

    _conf = SparkConf().set("spark.driver.maxResultSize", "2G").setAppName('KDD preprocessing')
    _sc = SparkContext(conf=_conf)
    _sqlC = SQLContext(_sc)
    

 
    schema = StructType([
        StructField("duration", StringType(), True),
        StructField("protocol_type", StringType(), True),
        StructField("service", StringType(), True),
        StructField("flag", StringType(), True),
        StructField("src_bytes", StringType(), True),
        StructField("dst_bytes", StringType(), True),
        StructField("land", StringType(), True),
        StructField("wrong_fragment", StringType(), True),
        StructField("urgent", StringType(), True),
        StructField("hot", StringType(), True),
        StructField("num_failed_logins", StringType(), True),
        StructField("logged_in", StringType(), True),
        StructField("num_compromised", StringType(), True),
        StructField("root_shell", StringType(), True),
        StructField("su_attempted", StringType(), True),
        StructField("num_root", StringType(), True),
        StructField("num_file_creations", StringType(), True),
        StructField("num_shells", StringType(), True),
        StructField("num_access_files", StringType(), True),
        StructField("num_outbound_cmds", StringType(), True),
        StructField("is_host_login", StringType(), True),
        StructField("is_guest_login", StringType(), True),
        StructField("count", StringType(), True),
        StructField("srv_count", StringType(), True),
        StructField("serror_rate", StringType(), True),
        StructField("srv_serror_rate", StringType(), True),
        StructField("rerror_rate", StringType(), True),
        StructField("srv_rerror_rate", StringType(), True),
        StructField("same_srv_rate", StringType(), True),
        StructField("diff_srv_rate", StringType(), True),
        StructField("srv_diff_host_rate", StringType(), True),
        StructField("dst_host_count", StringType(), True),
        StructField("dst_host_srv_count", StringType(), True),
        StructField("dst_host_same_srv_rate", StringType(), True),
        StructField("dst_host_diff_srv_rate", StringType(), True),
        StructField("dst_host_same_src_port_rate", StringType(), True),
        StructField("dst_host_srv_diff_host_rate", StringType(), True),
        StructField("dst_host_serror_rate", StringType(), True),
        StructField("dst_host_srv_serror_rate", StringType(), True),
        StructField("dst_host_rerror_rate", StringType(), True),
        StructField("dst_host_srv_rerror_rate", StringType(), True),
        StructField("answer", StringType(), True)
    ])

    

    final_struc = StructType(fields=schema)
    rdd_kdd  = _sc.textFile(sys.argv[1]).map(lambda x : x.split(","))
    df = _sqlC.createDataFrame(rdd_kdd, schema=schema)
    preprocessing = PreprocessData()
    
    continues_data_for_bucket_labels = ["duration", "dst_bytes", "count", "serror_rate", "rerror_rate", "same_srv_rate",
                                         "diff_srv_rate", "srv_count", "srv_serror_rate", "srv_rerror_rate"]


    for col in continues_data_for_bucket_labels:
        df = preprocessing.bucketize(df, col)

    field_names = ["duration_bucketized","src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
                   "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                   "is_guest_login", "count_bucketized", "srv_count_bucketized", "serror_rate_bucketized",
                   "srv_serror_rate_bucketized", "rerror_rate_bucketized", "srv_rerror_rate_bucketized",
                   "same_srv_rate_bucketized", "diff_srv_rate_bucketized", "srv_diff_host_rate", "dst_host_count",
                   "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                   "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                   "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

   
    for field in field_names:
        df = preprocessing.field_name_changer(df, field)

 
    logs_fields = ["protocol_type","flag","service"]+[ item + "_enc" for item in field_names] + ["answer"]
 
    df = df.select(func.concat_ws(",", *logs_fields))
    
    df.write.csv(sys.argv[2])
    
    alltokens = df.rdd.map(list).flatMap(lambda row : row[0].split(','))
    vocabulary = alltokens.distinct().zipWithIndex().collectAsMap()
    vocabulary['unknown'] = 404404
    print(vocabulary)
    with open('vocabulary.pickle', 'wb') as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    word_frequencies = alltokens.map(lambda token : (token, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x :-x[1])
    with open("word_frequencies.pickle", 'wb') as handle:
        pickle.dump(word_frequencies.collectAsMap(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    _sc.stop()
