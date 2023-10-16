# Make sure following ob paramters are set before running this job.

# "jobParameters": [
# 			{
# 				"key": "--input_path",
# 				"value": "s3://<bucket>/sample/sample.fasta.gz",
# 			},
# 			{
# 				"key": "--output_path",
# 				"value": "s3://<bucket>/sample-output/sample.csv",
# 			}
# 		]

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize a GlueContext
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME', "input_path", "output_path"])

# The previous PySpark code
def lines_to_records(lines):
    record = None
    for line in lines:
        if line.startswith(">"):
            if record:
                record["sequence"] = "".join(record["sequence"])  # Convert sequence array to string here
                yield record
            record = {"sequence_id": line[1:], "sequence": []}
        else:
            record["sequence"].append(line)
    if record:
        record["sequence"] = "".join(record["sequence"])
        yield record

# Reading from the gzipped FASTA file on S3
input_path = args["input_path"]
output_path = args["output_path"]

rdd = spark.sparkContext.textFile(input_path)
records = rdd.mapPartitions(lines_to_records)
df = records.toDF()

# Saving the DataFrame to S3 as CSV
df.write.option("header", "true").csv(output_path)

job.commit()


