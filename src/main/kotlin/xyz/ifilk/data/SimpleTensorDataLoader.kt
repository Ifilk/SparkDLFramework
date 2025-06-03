package xyz.ifilk.data

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.utils.takeRows

class SimpleTensorDataLoader(
    private val x: Tensor,
    private val y: Tensor,
    private val batchSize: Int = 32,
    private val shuffle: Boolean = true
) : DataLoader {

    private val sampleCount = x.shape[0]
    private val indices = IntArray(sampleCount) { it }

    override val numBatches: Int
        get() = (sampleCount + batchSize - 1) / batchSize

    override fun iterator(): Iterator<Pair<Tensor, Tensor>> = iterator {
        if (shuffle) indices.shuffle()

        for (start in 0 until sampleCount step batchSize) {
            val end = minOf(start + batchSize, sampleCount)
            val batchIndices = indices.sliceArray(start until end)
            val xb = x.takeRows(batchIndices)
            val yb = y.takeRows(batchIndices)
            yield(xb to yb)
        }
    }
}
