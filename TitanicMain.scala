package ro.comanitza.kaggle

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType

/**
 *
 * Simple implementation of the solution for the Titanic machine learning from Kaggle
 *
 * More info here https://www.kaggle.com/c/titanic/overview
 *
 * @author Stefan Comanita
 */
object TitanicMain {

  def main(args: Array[String]): Unit = {
    println("kaggle titanic challenge...")

    /*
     * if you run on windows you will probably need something similar
     */
    System.setProperty("hadoop.home.dir", "D:\\stuff\\hadoopHome\\winutils-master\\winutils-master\\hadoop-3.0.0")

    testTitanic()
  }

  private def testTitanic(): Unit = {

    val session = SparkSession.builder()
      .config(new SparkConf().setAppName("spark-kaggle-titanic").setMaster("local[*]"))
      .getOrCreate()

    /*
     * set the log level to warn
     */
    session.sparkContext.setLogLevel("WARN")

    /*
     * read the train data
     */
    val df = session.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("D:\\stuff\\texts\\titanic\\train.csv")

    /*
     * read the test data
     */
    val testDf = session.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("D:\\stuff\\texts\\titanic\\test.csv")

    df.show()

    println(s"df count: ${df.count()} and testDf count ${testDf.count()}")

    /*
     * we must encode the gender in order to be able to use it, as it's currently a string
     */
    val genderEncoder = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("encodedSex")
      .setHandleInvalid("keep")

    /*
     * assemble the features that we will use
     */
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "encodedSex", "Age", "Fare"))
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val classifier = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("Survived")
      .setMaxBins(64)

    val pipeline = new Pipeline().setStages(Array(genderEncoder, assembler, classifier))

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")
      .setRawPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(32, 64))
      .addGrid(classifier.maxDepth, Array(16, 24, 30))
      .addGrid(classifier.numTrees, Array(10, 18))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)
      .setParallelism(2)

    /*
     * save some of the train data to use it for a small accuracy check
     */
    val Array(trainDataDf, testDataDf) = df.randomSplit(Array(0.9, 0.1))

    val model = crossValidator.fit(trainDataDf)

    val testPredictions = model.transform(testDataDf)

    val accuracy = evaluator.evaluate(testPredictions)

    println(s"test data accuracy $accuracy")

    val actualPredictions = model.transform(testDf)

    /*
     * write the output result in the Kaggle format
     */
    actualPredictions.withColumn("Survived", actualPredictions("prediction").cast(IntegerType))
      .select("PassengerId", "Survived").write.format("csv")
      .option("header", "true").save(s"D:\\stuff\\texts\\titanic\\result-$accuracy")
  }
}
