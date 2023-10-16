# Make sure following ob paramters are set before running this job.

# "jobParameters": [
# 			{
# 				"key": "--input_torkenized_path",
# 				"value": "s3://<bucket>/uniref100/torkenized-1mb-650m-v1/",
# 			}
# 	
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import input_file_name, count, sum, lit, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql import functions as F
from pyspark.sql.functions import regexp_extract


def create_index_map(input_path, output_path):
    # Step 1: Read the saved Parquet files and associate each record with its file name.
    df = spark.read.parquet(input_path)
    df_with_file = df.withColumn("file_name", F.input_file_name())
    
    # Extract just the file name from the full path
    df_with_file = df_with_file.withColumn("file_name", regexp_extract(F.col("file_name"), ".*\/([^\/]+)$", 1))

    # Step 2: Compute counts and create a running total
    aggregated_df = df_with_file.groupBy("file_name").agg(F.count("*").alias("num_sequences"))

    window_spec = Window.orderBy("file_name")
    aggregated_df = aggregated_df.withColumn("cum_sum", F.sum("num_sequences").over(window_spec))

    # Step 3: Compute start_line and end_line
    aggregated_df = aggregated_df.withColumn("start_line", F.lag("cum_sum", 1, 0).over(window_spec))
    aggregated_df = aggregated_df.withColumn("end_line", F.col("cum_sum") - 1)

    # Write the result to S3 as a single file
    aggregated_df.select("file_name", "num_sequences", "start_line", "end_line")\
                 .coalesce(1)\
                 .write.option("header", "true")\
                 .csv(output_path)
                 

## Initialize GlueContext and SparkContext
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Initialize Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_torkenized_path'])
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define your input and output paths
input_parquet_path = args["input_torkenized_path"]

# Generate index maps for train and test
create_index_map(input_parquet_path + "train/", input_parquet_path + "train/train_index_map")
create_index_map(input_parquet_path + "test/", input_parquet_path + "test/test_index_map")

# Complete the Glue job
job.commit()
