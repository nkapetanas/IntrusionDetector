from __future__ import print_function

import numpy as np
from pyspark.python.pyspark.shell import sc
from pyspark.sql import SparkSession, types
from pyspark.mllib.feature import Word2Vec
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as func
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import csv
from itertools import islice

#DICTIONARY_WORD2VEC = "dict_word2vec_win5.pickle"
#DICTIONARY_WORD2VEC = "dict_word2vec_win10.pickle"
DICTIONARY_WORD2VEC = "dict_word2vec.pickle"
DICTIONARY_FREQUENT_WORD2VEC = "dict_frequent_word2vec.pickle"
#KDD_DATA = "data/kddcup.data_10_percent_corrected"
KDD_DATA = "data/kddcup.data.corrected"


class PreprocessData():

    def __init__(self):
        return

    def to_plot(self, num, dictionary, embs):
        plot_embs = []
        i = 0
        while i < num:
            plot_embs.append(embs[dictionary.keys()[i]])
            # if you run the above line with python 3 and above change it to
            #  plot_embs.append(embs[list(dictionary.keys())[i]])
            i = i + 1
        return plot_embs

    # we pass a df and the field column we want to bucketize
    def bucketize(self, df, field, step=100):
        df = df.withColumn(field, df[field].cast("int"))
        max = df.agg({field: "max"}).collect()[0][0]
        min = df.agg({field: "min"}).collect()[0][0]
        std = df.agg({field: "stddev"}).collect()[0][0]
        number_of_buckets = max - min / std
        buckets = np.arange(min, number_of_buckets, step, dtype=np.float).tolist()
        buckets = np.concatenate([[-float('inf')], buckets, [float('inf')]]).tolist()
        bucketizer = Bucketizer(splits=buckets, inputCol=field,
                                outputCol=field + '_bucketized')
        print("column is: ",field)
        bucketized_features = bucketizer.transform(df)
        return bucketized_features


    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))

    def field_name_changer(selfe, df, field):
        def func_inner(value):
            return str(value) + field
        field_udf = func.udf(func_inner, types.StringType())
        return df.withColumn(field + '_enc', field_udf(field))


    def save_word2vec(self, vectors, dictionaryOfFrequentWords):
        vecs_python = {}
        for key in vectors.keys():
            vecs_python[key] = list(vectors[key])

        word2vec_results_dict = {key : vecs_python[key] for key in vectors.keys()}

        with open(DICTIONARY_WORD2VEC, 'wb') as handle:
            pickle.dump(word2vec_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DICTIONARY_FREQUENT_WORD2VEC, 'wb') as handle:
            pickle.dump(dictionaryOfFrequentWords, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return word2vec_results_dict

    def results_visualization(self):

        with open(DICTIONARY_WORD2VEC, 'rb') as handle:
            word2vec_results_dict = pickle.load(handle)

        with open(DICTIONARY_FREQUENT_WORD2VEC, 'rb') as handle:
            dictionary = pickle.load(handle)

        #word2vec_results_dict = dict(map(lambda kv: (kv[0], f(kv[1])), word2vec_results_dict.items()))

        # X = np.vstack(word2vec_results_dict.values())

        # for k,i in dictionary.items():
        #     new_key = unicodedata.normalize('NFKD', k).encode('ascii', 'ignore')
        #     dictionary[new_key] = dictionary.pop(k)
        #     dictionary[new_key] = str(i)

        plot = self.to_plot(500, dictionary, word2vec_results_dict)

        np.set_printoptions(suppress=True)
        x_Embedded = TSNE(perplexity=100, n_iter=5000).fit_transform(plot)
        lol = []
        for i in range(1000):
            lol.append('')

        plt.scatter(x_Embedded[:, 0], x_Embedded[:, 1])
        for label, x, y in zip(lol[:500], x_Embedded[:, 0], x_Embedded[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()
        plt.scatter(x_Embedded[:, 0], x_Embedded[:, 1])

        mostFrequentKeys = sorted(dictionary, key=dictionary.get)
        for label, x, y in zip(mostFrequentKeys[:500], x_Embedded[:, 0], x_Embedded[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.show()

    def Word2_vec(self, logs_rdd):

        # Input data: Each row is a bag of words from a sentence or document.

        word2vec = Word2Vec()
        word2vec.setNumPartitions(9).setVectorSize(100).setMinCount(1).setWindowSize(3).setLearningRate(0.00005)
        model = word2vec.fit(logs_rdd)

        wordCount = logs_rdd.flatMap(lambda line: line).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: -x[1])
        frequent_words = wordCount.top(1000, key=None)
        temp_dict1 = [i[0] for i in frequent_words]
        temp_dict2 = [i[1] for i in frequent_words]

        dictionaryOfFrequentWords = dict(zip(temp_dict1, temp_dict2))

        return model.getVectors(), dictionaryOfFrequentWords


if __name__ == '__main__':
    spark = SparkSession.builder.appName('ids').getOrCreate()

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
    df = spark.read.csv(KDD_DATA, schema=final_struc, inferSchema=True, header=True)

    preprocessing = PreprocessData()
    # uncomment the above line, if you want to bucketize also the field dst_bytes
    # continues_data_for_bucket_labels = ["duration", "dst_bytes", "count", "serror_rate", "rerror_rate", "same_srv_rate",
    #                                     "diff_srv_rate", "srv_count", "srv_serror_rate", "srv_rerror_rate"]

    continues_data_for_bucket_labels = ["duration", "count", "serror_rate", "rerror_rate", "same_srv_rate",
                                        "diff_srv_rate", "srv_count", "srv_serror_rate", "srv_rerror_rate"]
    dataframe_with_bucket = df
    for col in continues_data_for_bucket_labels:
        dataframe_with_bucket = preprocessing.bucketize(dataframe_with_bucket, col)

    field_names = ["duration_bucketized", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
                   "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                   "is_guest_login", "count_bucketized", "srv_count_bucketized", "serror_rate_bucketized",
                   "srv_serror_rate_bucketized", "rerror_rate_bucketized", "srv_rerror_rate_bucketized",
                   "same_srv_rate_bucketized", "diff_srv_rate_bucketized", "srv_diff_host_rate", "dst_host_count",
                   "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                   "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                   "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

    for field in field_names:
         dataframe_with_bucket = preprocessing.field_name_changer(dataframe_with_bucket, field)

    # logs_fields = """duration_bucketized_enc,protocol_type,service,flag,src_bytes_enc,dst_bytes_bucketized_enc,land_enc,wrong_fragment_enc,urgent_enc,hot_enc,num_failed_logins_enc,logged_in_enc,num_compromised_enc,root_shell_enc,su_attempted_enc,num_root_enc,num_file_creations_enc,num_shells_enc,num_access_files_enc,num_outbound_cmds_enc,is_host_login_enc,is_guest_login_enc,count_bucketized_enc,srv_count_bucketized_enc,serror_rate_bucketized_enc,srv_serror_rate_bucketized_enc,rerror_rate_bucketized_enc,srv_rerror_rate_bucketized_enc,same_srv_rate_bucketized_enc,diff_srv_rate_bucketized_enc,srv_diff_host_rate_enc,dst_host_count_enc,dst_host_srv_count_enc,dst_host_same_srv_rate_enc,dst_host_diff_srv_rate_enc,dst_host_same_src_port_rate_enc,dst_host_srv_diff_host_rate_enc,dst_host_serror_rate_enc,dst_host_srv_serror_rate_enc,dst_host_rerror_rate_enc,dst_host_srv_rerror_rate_enc""".split(
    #     ',')
    # "dst_bytes_bucketized"
    logs_fields = """duration_bucketized_enc,protocol_type,service,flag,src_bytes_enc,dst_bytes_enc,land_enc,wrong_fragment_enc,urgent_enc,hot_enc,num_failed_logins_enc,logged_in_enc,num_compromised_enc,root_shell_enc,su_attempted_enc,num_root_enc,num_file_creations_enc,num_shells_enc,num_access_files_enc,num_outbound_cmds_enc,is_host_login_enc,is_guest_login_enc,count_bucketized_enc,srv_count_bucketized_enc,serror_rate_bucketized_enc,srv_serror_rate_bucketized_enc,rerror_rate_bucketized_enc,srv_rerror_rate_bucketized_enc,same_srv_rate_bucketized_enc,diff_srv_rate_bucketized_enc,srv_diff_host_rate_enc,dst_host_count_enc,dst_host_srv_count_enc,dst_host_same_srv_rate_enc,dst_host_diff_srv_rate_enc,dst_host_same_src_port_rate_enc,dst_host_srv_diff_host_rate_enc,dst_host_serror_rate_enc,dst_host_srv_serror_rate_enc,dst_host_rerror_rate_enc,dst_host_srv_rerror_rate_enc,answer""".split(
    ',')

    dataframe_with_bucket = dataframe_with_bucket.select(func.concat_ws(",", *logs_fields)).alias("lxplus")

    # dataframe_with_bucket.to_csv('kdd_Preprocessed.csv',sep=',',encoding='utf-8')

    #dataframe_with_bucket.write.csv('kdd_preprocessed.csv')

    # In order to call Word2Vec function, uncomment the above 3 lines.

    # logs_rdd = dataframe_with_bucket.rdd.map(lambda s: s[0].split('%'))
    # model_getVectors, dictionaryOfFrequentWords = Word2_vec(logs_rdd)
    # word2vec_results_dict = save_word2vec(model_getVectors,dictionaryOfFrequentWords)

    # To produce and visualize the results from the Word2Vec, uncomment the above line.
    #results_visualization(word2vec_results_dict,dictionaryOfFrequentWords)

    # To visualize the results from the Word2Vec, from the pickle files, uncomment the above line.
    # preprocessing.results_visualization()

    spark.stop()