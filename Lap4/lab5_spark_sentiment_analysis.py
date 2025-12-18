from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .getOrCreate()

    data_path = "sentiments.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

    df = df.dropna(subset=["text", "label"])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    lr = LogisticRegression(maxIter=20, regParam=0.01, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
