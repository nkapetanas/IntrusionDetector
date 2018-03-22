from __future__ import print_function

import lit as lit
import numpy as np
from pyspark.shell import sqlContext, sc
from pyspark.sql import SparkSession, types, dataframe
#from pyspark.ml.feature import Word2Vec
from pyspark.mllib.feature import Word2Vec
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as func
from pyspark import SQLContext
import pickle
import csv


spark = SparkSession.builder.appName('ids').getOrCreate()



schema = StructType([
    #StructField("start_Signal", StringType(), True),
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
    StructField("anws", StringType(), True)
])

final_struc = StructType(fields = schema)
df = spark.read.csv('data/kddcup.data_10_percent_corrected',schema=final_struc, inferSchema = True, header=True)

def write_txt(df):
    df.write.csv('preprocessed.csv')

# we pass a df and the field column we want to bucketize
def bucketize(df, field, min=-10, max=10, step=100):
    df = df.withColumn(field, df[field].cast("int"))
    max = df.agg({field: "max"}).collect()[0][0]
    min = df.agg({field: "min"}).collect()[0][0]
    std = df.agg({field: "stddev"}).collect()[0][0]
    #print (max)
    #print (min)
    number_of_buckets = max - min / std
    buckets = np.arange(min, number_of_buckets, step, dtype=np.float).tolist()
    buckets = np.concatenate([[-float('inf')], buckets, [float('inf')]]).tolist()
    bucketizer = Bucketizer(splits=buckets, inputCol=field,
                            outputCol=field + '_bucketized')
    bucketized_features = bucketizer.transform(df)
    #bucketized_features.show()
    return bucketized_features


continues_data_for_bucket_labels = ["duration","dst_bytes","count","serror_rate","rerror_rate","same_srv_rate","diff_srv_rate","srv_count","srv_serror_rate","srv_rerror_rate"]
dataframe_with_bucket = df
for col in continues_data_for_bucket_labels:
    dataframe_with_bucket = bucketize(dataframe_with_bucket, col)

#dataframe_with_bucket.show()



def field_name_changer(df, field):
    def func_inner(value):
        return str(value) + field
    field_udf = func.udf(func_inner, types.StringType())
    return df.withColumn(field + '_enc', field_udf(field))

field_names = [ "duration_bucketized","src_bytes", "dst_bytes_bucketized", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count_bucketized", "srv_count_bucketized", "serror_rate_bucketized", "srv_serror_rate_bucketized", "rerror_rate_bucketized", "srv_rerror_rate_bucketized", "same_srv_rate_bucketized", "diff_srv_rate_bucketized", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

for field in field_names:
    #dataframe_with_bucket.show()
    dataframe_with_bucket = field_name_changer(dataframe_with_bucket,field)

logs_fields = """duration_bucketized_enc,protocol_type,service,flag,src_bytes_enc,dst_bytes_bucketized_enc,land_enc,wrong_fragment_enc,urgent_enc,hot_enc,num_failed_logins_enc,logged_in_enc,num_compromised_enc,root_shell_enc,su_attempted_enc,num_root_enc,num_file_creations_enc,num_shells_enc,num_access_files_enc,num_outbound_cmds_enc,is_host_login_enc,is_guest_login_enc,count_bucketized_enc,srv_count_bucketized_enc,serror_rate_bucketized_enc,srv_serror_rate_bucketized_enc,rerror_rate_bucketized_enc,srv_rerror_rate_bucketized_enc,same_srv_rate_bucketized_enc,diff_srv_rate_bucketized_enc,srv_diff_host_rate_enc,dst_host_count_enc,dst_host_srv_count_enc,dst_host_same_srv_rate_enc,dst_host_diff_srv_rate_enc,dst_host_same_src_port_rate_enc,dst_host_srv_diff_host_rate_enc,dst_host_serror_rate_enc,dst_host_srv_serror_rate_enc,dst_host_rerror_rate_enc,dst_host_srv_rerror_rate_enc""".split(',')

dataframe_with_bucket = dataframe_with_bucket.select(func.concat_ws("%", *logs_fields)).alias("lxplus")

#dataframe_with_bucket = dataframe_with_bucket.select(func.concat_ws(" ", *logs_fields)).alias("lxplus")
#dataframe_with_bucket = dataframe_with_bucket.select(*logs_fields).alias("lxplus")

column_temp = "concat_ws(%, duration_bucketized_enc, protocol_type, service, flag, src_bytes_enc, dst_bytes_bucketized_enc, land_enc, wrong_fragment_enc, urgent_enc, hot_enc, num_failed_logins_enc, logged_in_enc, num_compromised_enc, root_shell_enc, su_attempted_enc, num_root_enc, num_file_creations_enc, num_shells_enc, num_access_files_enc, num_outbound_cmds_enc, is_host_login_enc, is_guest_login_enc, count_bucketized_enc, srv_count_bucketized_enc, serror_rate_bucketized_enc, srv_serror_rate_bucketized_enc, rerror_rate_bucketized_enc, srv_rerror_rate_bucketized_enc, same_srv_rate_bucketized_enc, diff_srv_rate_bucketized_enc, srv_diff_host_rate_enc, dst_host_count_enc, dst_host_srv_count_enc, dst_host_same_srv_rate_enc, dst_host_diff_srv_rate_enc, dst_host_same_src_port_rate_enc, dst_host_srv_diff_host_rate_enc, dst_host_serror_rate_enc, dst_host_srv_serror_rate_enc, dst_host_rerror_rate_enc, dst_host_srv_rerror_rate_enc)"

#dataframe_with_bucket.show(20,False)


logs_rdd = dataframe_with_bucket.rdd.map(lambda s : s[0].split('%'))


'''
f = open("data/kddcup.data_10_percent_corrected", "r")
logs = []
answers = []
for log in f:
    tokens = log.split(',')
    features = log.split(',')[ :-1]
    answers.append(tokens[-1]) # -1 means that we take the last element from the list
    #for i in range(0 , len(field_names)):
        #features[i] = str(features[i]) + field_names[i]
    logs.append(features)

for index, answer in enumerate(answers):
    answers[index]= answer.rstrip()
    answers[index] = answers[index].strip('.')

'''





#sqlContext = SQLContext(sc)

#logsTest_rdd = sc.textFile("data/kddcup.data_10_percent_corrected").map(lambda line: line.split(","))

#Employee_df = logsTest_rdd.toDF(schema)



#word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate, anws", outputCol="result")
#model = word2Vec.fit(Employee_df)
#model.getVectors().show()


# for counting unique values
# #lstemp = [log[22] for log in logs]
#temp = list(set(lstemp))
#print(len(temp))

def save_word2vec(vectors):
    vecs_python = {}
    for key in vectors.keys():
        # print(key)
        vecs_python[key] = list(vectors[key])
    print('The vocabulary size is:' + str(len(vecs_python)))

    #np.array(vecs_python).dump(open('array.npy', 'wb'))
    a = np.array(vecs_python)
    #np.set_printoptions(threshold=np.nan)
    #print(a)
    #for key in vectors.keys():
        #print("The key is:", key, "and the values are:", vecs_python[key] )

    word2vec_results_dict = {key : vecs_python[key] for key in vectors.keys()}

    #with open('word2vec_dict.txt', 'wb') as handle:
        #wr = csv.writer(handle, quoting=csv.QUOTE_ALL)
        #wr.writerow(vecs_python)
        #for item in vecs_python:
            #handle.write(item)
    with open('dict.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in word2vec_results_dict.items():
            writer.writerow([key, value])
    #return vecs_python


def Word2_vec():
    # Input data: Each row is a bag of words from a sentence or document.

    # train word2vec
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "),),
        ("I wish Java could use case classes".split(" "),),
        ("Logistic regression models are neat".split(" "),)
    ], ["text"])



    #documentDF_temp.show()
    # df = sqlContext.createDataFrame(logs_rdd, schema)
    #listT = ["duration_bucketized_enc", "protocol_type", "service, flag", "src_bytes_enc", "dst_bytes_bucketized_enc", "land_enc, wrong_fragment_enc", "urgent_enc", "hot_enc", "num_failed_logins_enc", "logged_in_enc", "num_compromised_enc", "root_shell_enc", "su_attempted_enc", "num_root_enc", "num_file_creations_enc", "num_shells_enc", "num_access_files_enc", "num_outbound_cmds_enc", "is_host_login_enc", "is_guest_login_enc", "count_bucketized_enc", "srv_count_bucketized_enc", "serror_rate_bucketized_enc", "srv_serror_rate_bucketized_enc", "rerror_rate_bucketized_enc", "srv_rerror_rate_bucketized_enc", "same_srv_rate_bucketized_enc", "diff_srv_rate_bucketized_enc", "srv_diff_host_rate_enc", "dst_host_count_enc", "dst_host_srv_count_enc", "dst_host_same_srv_rate_enc", "dst_host_diff_srv_rate_enc", "dst_host_same_src_port_rate_enc", "dst_host_srv_diff_host_rate_enc", "dst_host_serror_rate_enc", "dst_host_srv_serror_rate_enc", "dst_host_rerror_rate_enc", "dst_host_srv_rerror_rate_enc"]
    #word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol=(col for col in listT), outputCol="result")
    #model = word2Vec.fit(dataframe_with_bucket)
    #model.getVectors().show()

    #dataframe_bucked_map = dataframe_with_bucket.rdd.map(lambda x: (x['duration_bucketized_enc'], x['protocol_type'],x['service'], x['flag'],x['src_bytes_enc'], x['dst_bytes_bucketized_enc'], x['land_enc'], x['wrong_fragment_enc'],x['urgent_enc'], x['hot_enc'],x['num_failed_logins_enc'], x['logged_in_enc'],x['num_compromised_enc'], x['root_shell_enc'], x['su_attempted_enc'],x['num_root_enc'], x['num_file_creations_enc'], x['num_shells_enc'],x['num_access_files_enc'], x['num_outbound_cmds_enc'], x['is_host_login_enc'], x['is_guest_login_enc'], x['count_bucketized_enc'], x['srv_count_bucketized_enc'],x['serror_rate_bucketized_enc'], x['srv_serror_rate_bucketized_enc'], x['rerror_rate_bucketized_enc'], x['srv_rerror_rate_bucketized_enc'], x['same_srv_rate_bucketized_enc'], x['diff_srv_rate_bucketized_enc'],x['srv_diff_host_rate_enc'], x['dst_host_count_enc'], x['dst_host_srv_count_enc'], x['dst_host_same_srv_rate_enc'],x['dst_host_diff_srv_rate_enc'], x['dst_host_same_src_port_rate_enc'],x['dst_host_srv_diff_host_rate_enc'], x['dst_host_serror_rate_enc'], x['dst_host_srv_serror_rate_enc'], x['dst_host_rerror_rate_enc'],x['dst_host_srv_rerror_rate_enc']))
    #mtcars_map.take(5)
    #rdd = dataframe_with_bucket.rdd

    word2vec = Word2Vec()
    word2vec.setNumPartitions(9).setVectorSize(100).setMinCount(1).setWindowSize(3).setLearningRate(0.00005)
    model = word2vec.fit(logs_rdd)

    #word2vec_dict= save_word2vec(model.getVectors())
    save_word2vec(model.getVectors())
    #print(word2vec_dict.values())
    ###########################################
    #wordCount = logs_rdd.flatMap(lambda line: line).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: -x[1])
    #with open("word2vec_dir.pickle" + '_vocab.pickle', 'wb') as handle:
        #pickle.dump(wordCount.collect(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    #synonyms = model.findSynonyms('tcp', 40)

    #for word, cosine_distance in synonyms:
        #print("{}: {}".format(word, cosine_distance))


    ##########################################
    #dataframe_bucked_map.foreach(print)
    #logs_rdd.foreach(print)
    # Learn a mapping from words to Vectors.
    #word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol=column_temp, outputCol="result")
    #model = word2Vec.fit(dataframe_with_bucket)

    #result = model.transform(df)
    #for row in result.collect():
        #text, vector = row
        #print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
    #model.getVectors()



Word2_vec()

spark.stop()

