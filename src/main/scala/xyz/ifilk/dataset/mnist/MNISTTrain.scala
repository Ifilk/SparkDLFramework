package xyz.ifilk.dataset.mnist

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Result
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import xyz.ifilk.distributed.spark.SparkDist
import xyz.ifilk.distributed.{GradPacket, Sample}
import xyz.ifilk.functions.CrossEntropyLoss
import xyz.ifilk.nn.Parameter
import xyz.ifilk.optim.Adam
import xyz.ifilk.tensor.Tensor
import xyz.ifilk.utils.ModuleUtilKt.oneHot
import xyz.ifilk.utils.TensorUtilKt.{deserializeTensor, fromIntArray}

object MNISTTrain {
  private def readMNISTFromHBase(sc: JavaSparkContext, tableName: String): JavaRDD[Sample] = {
    val zkQuorum = System.getenv("HBASE_ZOOKEEPER_QUORUM")
    if (zkQuorum == null)
      throw new IllegalStateException("HBASE_ZOOKEEPER_QUORUM environment variable is not set")

    val hbaseConf = HBaseConfiguration.create()
    hbaseConf.set("hbase.zookeeper.quorum", zkQuorum)
    hbaseConf.set(TableInputFormat.INPUT_TABLE, tableName)

    val hbaseRDD = sc.newAPIHadoopRDD(
      hbaseConf,
      classOf[TableInputFormat],
      classOf[ImmutableBytesWritable],
      classOf[Result]
    )

    hbaseRDD.map { case (_, result) =>
      val featureBytes = result.getValue(Bytes.toBytes("data"), Bytes.toBytes("image"))
      val labelBytes = result.getValue(Bytes.toBytes("data"), Bytes.toBytes("label"))

      if (featureBytes == null || labelBytes == null) {
        throw new IllegalArgumentException("Missing features or label column in HBase row")
      }

      val featureTensor = deserializeTensor(featureBytes)
      val labelInt = Bytes.toInt(labelBytes)
      val labelTensor = oneHot(fromIntArray(Array(labelInt), Array(1)), 10)

      new Sample(featureTensor, labelTensor)
    }
  }

  def main(args: Array[String]): Unit = {
    val epochs = System.getenv("EPOCHS")
    if (epochs == null)
      throw new IllegalStateException("EPOCHS")

    val conf = new SparkConf()
      .setAppName("NcclRpcServer")
      .setMaster("local[*]") // 或 spark://spark:7077

    conf.registerKryoClasses(Array(
      classOf[Tensor],
      classOf[GradPacket],
      classOf[Parameter]
    ))

    val jsc = new JavaSparkContext(conf)
    val dataset = readMNISTFromHBase(jsc, "mnist_dataset")

    SparkDist.INSTANCE.train(
      jsc,
      dataset,
      () => new MNISTMLP(),                        // 模型构造器
      new CrossEntropyLoss(),
      epochs.toInt,                                // epochs
      64,                                          // batchSize
      model => new Adam(model, 0.001, 0.9, 0.999, 1e-8),// 优化器构造器
      false                                         // barrierMode
    )
  }
}

