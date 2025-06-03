package xyz.ifilk.distributed.spark

import xyz.ifilk.data.DataLoader
import xyz.ifilk.distributed.Sample
import xyz.ifilk.tensor.Tensor

class SampleListDataLoader(
    private val samples: List<Sample>,
    private val batchSize: Int = 32,
    private val shuffle: Boolean = true
) : DataLoader {

    private val sampleCount = samples.size
    private val indices = IntArray(sampleCount) { it }

    override val numBatches: Int
        get() = (sampleCount + batchSize - 1) / batchSize

    override fun iterator(): Iterator<Pair<Tensor, Tensor>> = iterator {
        if (shuffle) indices.shuffle()

        for (start in 0 until sampleCount step batchSize) {
            val end = minOf(start + batchSize, sampleCount)
            val batch = indices.slice(start until end).map { samples[it] }
            val xb = Tensor.stack(batch.map { it.features })
            val yb = Tensor.stack(batch.map { it.target })
            yield(xb to yb)
        }
    }
}
