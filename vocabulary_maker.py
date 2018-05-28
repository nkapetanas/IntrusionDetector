from __future__ import print_function

from pyspark.shell import sc
from pyspark.sql import SparkSession
import csv

from pyspark.sql.types import StructField, StringType, StructType

if __name__ == '__main__':
    spark = SparkSession.builder.appName('ids').getOrCreate()


    alltokens = sc.textFile("kdd_preprocessed.csv/*.csv").flatMap(lambda line: line.replace('"','').split(","))
#wordsRDD = sc.parallelize(alltokens, 4)
# wordsDataframe = spark.read.csv(
#     "kdd_preprocessed.csv/*.csv", header=True, mode="DROPMALFORMED", schema=schema
# )
# wordsRDD = wordsDataframe.rdd.map(list)

    print(type(alltokens))
# Print out the type of wordsRDD
    words = alltokens.distinct().zipWithIndex().collectAsMap()

    import pickle
    with open('vocabulary.pickle', 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)
    spark.stop()