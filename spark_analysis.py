
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count, avg, desc

# Start Spark
spark = SparkSession.builder \
    .appName("PublicHealthOutbreakAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# Load cleaned dataset
df = spark.read.csv(
    "cleaned_health_data.csv",
    header=True,
    inferSchema=True
)

print("\n===== DATASET OVERVIEW =====")
print("Total records:", df.count())
df.printSchema()

# 1. Disease-wise case analysis
print("\n===== DISEASE-WISE CASE TOTALS =====")
disease_summary = df.groupBy("Disease").agg(
    sum("Cases").alias("Total_Cases"),
    sum("Deaths").alias("Reported_Deaths"),
    count("*").alias("Records")
).orderBy(desc("Total_Cases"))

disease_summary.show(10, truncate=False)

# 2. State-wise case analysis
print("\n===== STATE-WISE CASE TOTALS =====")
state_summary = df.groupBy("state_ut").agg(
    sum("Cases").alias("Total_Cases"),
    count("*").alias("Records")
).orderBy(desc("Total_Cases"))

state_summary.show(10, truncate=False)

# 3. Year-wise case trends
print("\n===== YEAR-WISE CASE TOTALS =====")
year_summary = df.groupBy("year").agg(
    sum("Cases").alias("Total_Cases"),
    avg("Cases").alias("Average_Cases")
).orderBy("year")

year_summary.show(truncate=False)

# Save results
disease_summary.toPandas().to_csv(
    "disease_summary.csv", index=False
)
state_summary.toPandas().to_csv(
    "state_summary.csv", index=False
)
year_summary.toPandas().to_csv(
    "year_summary.csv", index=False
)

print("\nAnalysis completed. Summary CSV files saved.")

spark.stop()
