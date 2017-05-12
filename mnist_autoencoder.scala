import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.StackedAutoencoder

val mnist = sc.textFile("/Users/jnoxon/Dropbox/python/DLFrameworks/mnist_x_train")
val parsedData = mnist.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
val tupleData = parsedData.map(x => Tuple1(x))
val mnistDF = spark.createDataFrame(tupleData).toDF("input")

val stackedAutoencoder = new StackedAutoencoder().setLayers(Array(784, 32)).setBlockSize(1).setMaxIter(100).setSeed(11L).setTol(1e-6).setInputCol("input").setOutputCol("output").setDataIn01Interval(true).setBuildDecoder(true)

val saModel = stackedAutoencoder.fit(mnistDF)
saModel.setInputCol("input").setOutputCol("encoded")
val encodedData = saModel.transform(mnistDF)
saModel.setInputCol("encoded").setOutputCol("decoded")
val decodedData = saModel.decode(encodedData)

val decoded = decodedData.select("decoded")
decoded.rdd.saveAsTextFile("/Users/jnoxon/Dropbox/python/DLFrameworks/spark_sigmoid_top_gradient.txt")
// decoded.write.repartition(1).format("com.databricks.spark.csv").option("header", "true").save("/Users/jnoxon/Dropbox/python/DLFrameworks/spark_sigmoid1.csv")