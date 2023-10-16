# Make sure following job paramters set before running this script 

# "jobParameters": [
# 			{
# 				"key": "--additional-python-modules",
# 				"value": "datasets==2.14.5, transformers==4.33.2, torch==1.13.1",
# 			},
# 			{
# 				"key": "--conf",
# 				"value": "spark.driver.maxResultSize=4g",
# 			},
# 			{
# 				"key": "--input-csv-path",
# 				"value": "s3://<bucket>/sample/sample.csv/part-00000-6ace383c-54dd-4929-bf8a-aa7797f227ee-c000.csv",
# 			},
# 			{
# 				"key": "--output_tokenized_path",
# 				"value": "s3://<bucket>/sample-v2/torkenized/",

# 			},
# 			{
# 				"key": "--sequence-length",
# 				"value": "142",
# 			},
# 			{
# 				"key": "--torkenizer-name",
# 				"value": "facebook/esm2_t6_8M_UR50D",
# 			}
# 		]

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import concat_ws

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
import os

# Initialize a GlueContext
sc = SparkContext()
sc.getConf().set("spark.driver.maxResultSize", "3g")
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'torkenizer_name', "sequence_length", "input_csv_path", "output_tokenized_path"])

tokenizer = AutoTokenizer.from_pretrained(args["torkenizer_name"])

def tokenize_sequence(sequence):
    sequence_length = int(args["sequence_length"])
    tokens = tokenizer(
        sequence, 
        padding="max_length",
        truncation=True,
        max_length=sequence_length
    )
    return (tokens["input_ids"], tokens["attention_mask"])


tokenize_schema = ArrayType(ArrayType(IntegerType()))
tokenize_udf = udf(tokenize_sequence, tokenize_schema)

# Path to the CSV in S3

input_csv_path = args["input_csv_path"]
output_tokenized_path = args["output_tokenized_path"]

# Read the CSV
df = spark.read.option("header", "true").csv(input_csv_path)

df_tokenized = df.withColumn("tokens", tokenize_udf(df["sequence"]))
df_tokenized = df_tokenized.withColumn("input_ids", df_tokenized.tokens[0])\
                           .withColumn("attention_mask", df_tokenized.tokens[1])
                           
#df_tokenized.select("input_ids", "attention_mask").show(truncate=False)
# Splitting into train and test
train_df, test_df = df_tokenized.randomSplit([0.8, 0.2])

num_train_partitions = 150000  # You might need to adjust this value based on your data and experimentation
num_test_partitions = 25000

# Repartition the DataFrame
train_df_repartitioned = train_df.repartition(num_train_partitions)
test_df_repartitioned = test_df.repartition(num_test_partitions)

train_df_repartitioned.write.parquet(output_tokenized_path + "/train")
test_df_repartitioned.write.parquet(output_tokenized_path + "/test")

job.commit()