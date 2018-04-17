import shelve
import pickle
import numpy as np
from scipy.sparse import *
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext

class MatrixFactorization:
    def __init__(self, maxIter=15, regParam=0.01, rank=10):
        self.maxIter = maxIter
        self.regParam = regParam
        self.rank = rank
        conf = SparkConf().setAppName("appName").setMaster("local[*]")
        # self.spark = SparkSession.builder.master("local[*]").appName("Example").getOrCreate()
        conf.set("spark.driver.memory","8g")
        conf.set("spark.executor.memory","8g")
        self.spark = SparkContext(conf=conf)
        print("New SparkSession started...")

    def change_parameter(self, regParam):
        self.regParam = regParam

    def matrix_factorization(self, train_lst):
        ratings = self.spark.parallelize(train_lst)
        model = ALS.train(ratings, rank=self.rank, seed=10, \
                          iterations=self.maxIter, \
                          lambda_=self.regParam)
        print("MF DONE")
        userFeatures = sorted(model.userFeatures().collect(), key=lambda d: d[0], reverse=False)
        productFeatures = sorted(model.productFeatures().collect(), key=lambda d: d[0], reverse=False)
        return userFeatures, productFeatures

    def end(self):
        self.spark.stop()
        print("SparkSession stopped.")