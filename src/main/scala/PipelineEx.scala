import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}



object PipelineEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Pipeline01")
    val sc = new SparkContext(conf)
    val sql = new SQLContext(sc)

    val training = sql.createDataFrame(Seq(
      (0L, "a b c d e", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")


    // Pipelien: tokenizer -> hashingTF -> LogisticReg
    val tokenizer = new Tokenizer()
                          .setInputCol("text")
                          .setOutputCol("words")

    val hashingTF = new HashingTF()
                          .setNumFeatures(1000)
                          .setInputCol(tokenizer.getOutputCol)
                          .setOutputCol("features")

    val lr = new LogisticRegression()
                   .setMaxIter(10)
                   .setRegParam(0.01)

    val pipeline = new Pipeline()
                        .setStages(Array(tokenizer, hashingTF, lr))

    val model: PipelineModel = pipeline.fit(training)

    val test: DataFrame = sql.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    model.transform(test.toDF())
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    }


  }
}
