package xyz.ifilk.distributed.spark

import org.apache.spark.BarrierTaskContext
import org.apache.spark.TaskContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import scala.reflect.ClassTag
import xyz.ifilk.data.DataLoader
import xyz.ifilk.distributed.GradPacket
import xyz.ifilk.distributed.ModuleBuilder
import xyz.ifilk.distributed.OptimizerBuilder
import xyz.ifilk.distributed.Sample
import xyz.ifilk.functions.Criticizer
import xyz.ifilk.tensor.Tensor

object SparkDist {

    fun train(
        sc: JavaSparkContext,
        dataset: JavaRDD<Sample>,
        moduleBuilder: ModuleBuilder,
        criticizer: Criticizer,
        epochs: Int,
        batchSize: Int,
        optimizerBuilder: OptimizerBuilder,
        barrierMode: Boolean = false,
    ) {
        // 1. Initial parameter snapshot built on driver
        val model = moduleBuilder()
        model.train()
        val globalOpt = optimizerBuilder(model)

        val totalSamples = dataset.count().toDouble()

        repeat(epochs) { epoch ->
            println("=== Epoch ${epoch + 1} / $epochs ===")

            val epochStart = System.currentTimeMillis()
            // 2. Broadcast parameters to executors
            val params = model.parameters.map { it.clone() }.toTypedArray()
            val bcParams = sc.broadcast(params)
            println("Broadcasted model parameters")

            // 3. Compute partition‑level gradients
            println("Computing gradients across partitions...")
            val gradPacket = if (barrierMode) {
                // 需要提前广播每个分区的数据
                val partitionedSamples = dataset
                    .mapPartitionsWithIndex<Pair<Int, Sample>>({ index, iter ->
                        iter.asSequence().map { index to it }.iterator()
                    }, true)
                    .groupBy { it.first }
                    .collectAsMap()
                    .mapValues { it.value.toList() }

                val bcPartitionedSamples = sc.broadcast(partitionedSamples)

                dataset.rdd().barrier()
                    .mapPartitions({
                        val ctx = BarrierTaskContext.get()
                        val partitionId = ctx.partitionId()

                        val localMod = moduleBuilder()
                        val localOpt = optimizerBuilder(model)
                        localMod.loadParameters(bcParams.value)
                        localMod.isTraining = true

                        val samples = bcPartitionedSamples.value[partitionId]!!.map { it.second }
                        val loader = createDataLoader(samples, batchSize, shuffle = false)
                        var batchLossSum = 0.0
                        var batchCount = 0

                        val accGrad = GradPacket.zeroLike(bcParams.value)
                        for ((features, labels) in loader) {
                            val pred = localMod.forward(features)
                            val loss = criticizer.call(pred, labels)
                            loss.backward()
                            localOpt.step()
                            localOpt.zeroGrad()

                            batchLossSum += loss.data.average()
                            batchCount += 1

                            localMod.parameters.indices.forEach { idx ->
                                val g = localMod.parameters[idx].grad ?: Tensor.zerosLike(localMod.parameters[idx])
                                val acc = accGrad.grads[idx]
                                for (k in acc.data.indices) acc.data[k] += g.data[k]
                            }
                        }
                        println("Partition $partitionId: avg batch loss = ${"%.6f".format(batchLossSum / (batchCount.coerceAtLeast(1)))}")

                        ctx.barrier()
                        JavaToScalaIterator(listOf(accGrad).iterator())
                    }, true, ClassTag.apply(GradPacket::class.java))
                    .treeReduce({ a, b -> a.apply { addInPlace(b) } }, 2)
            } else {
                dataset.mapPartitions { iter ->
                    val ctx = TaskContext.get()
                    val partitionId = ctx.partitionId()
                    val localMod = moduleBuilder()
                    val localOpt = optimizerBuilder(model)
                    localMod.loadParameters(bcParams.value)
                    localMod.train()

                    val accGrad = GradPacket.zeroLike(bcParams.value)
                    val samples = iter.asSequence().toList()
                    val loader = createDataLoader(samples, batchSize, shuffle = false)
                    var batchLossSum = 0.0
                    var batchCount = 0

                    for ((features, labels) in loader) {
                        val pred = localMod.forward(features)
                        val loss = criticizer.call(pred, labels)
                        loss.backward()
                        localOpt.step()
                        localOpt.zeroGrad()

                        batchLossSum += loss.data.average()
                        batchCount += 1

                        localMod.parameters.indices.forEach { idx ->
                            val g = localMod.parameters[idx].grad ?: Tensor.zerosLike(localMod.parameters[idx])
                            val acc = accGrad.grads[idx]
                            for (k in acc.data.indices) acc.data[k] += g.data[k]
                        }
                    }
                    println("Partition $partitionId: avg batch loss = ${"%.6f".format(batchLossSum / (batchCount.coerceAtLeast(1)))}")

                    listOf(accGrad).iterator()
                }.treeReduce({ a, b -> a.apply { addInPlace(b) } }, 2)
            }

            // 4. Average gradient & optimizer step on driver
            println("Aggregating gradients on driver and applying optimizer step...")
            gradPacket.grads.forEach { g ->
                for (j in g.data.indices) {
                    g.data[j] /= totalSamples
                }
            }
            globalOpt.step(gradPacket.grads)

            bcParams.unpersist(true)
//            val accuracy = totalCorrect.toDouble() / totalCount
            val epochEnd = System.currentTimeMillis()
            println("Epoch ${epoch + 1} finished in ${(epochEnd - epochStart)} ms")
            println()
        }
    }

    private fun createDataLoader(
        samples: List<Sample>,
        batchSize: Int = 32,
        shuffle: Boolean = true
    ): DataLoader {
        return SampleListDataLoader(samples, batchSize, shuffle)
    }
}
