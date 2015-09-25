/**
 * Created by gabe on 2015-09-21.
 */


import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SQLContext, Row}

object Estimators01 {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setAppName("Estimators01")
    val sc = new SparkContext(conf)
    val sqlCx = new SQLContext(sc)
    import sqlCx.implicits._

    val trainingSet: Seq[(Double, Vector)] = Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )

    val training: DataFrame = sqlCx.createDataFrame(trainingSet)
                                   .toDF("label", "features")

    val lr = new LogisticRegression()
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    lr.setMaxIter(10)
      .setRegParam(0.01)

    val model1 = lr.fit(training)
    println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

    // or override parameters
    val paramMap = ParamMap(lr.maxIter -> 20)
                     .put(lr.maxIter, 30)
                     .put(lr.regParam -> 0.1, lr.threshold -> 0.55)
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
    val paramMapCombined = paramMap ++ paramMap2

    val model2 = lr.fit(training, paramMapCombined)
    println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

    val test = sqlCx.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -1.0)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    model2.transform(test)
      .select("features", "label", "myProbability", "prediction")
      .collect()
      .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
      }
  }
 }

