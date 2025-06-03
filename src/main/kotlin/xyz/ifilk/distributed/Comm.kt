package xyz.ifilk.distributed

import xyz.ifilk.tensor.Tensor
import java.io.Serializable

/* =============== Helpers =============== */

/** Simple data sample — adapt fields to your task. */
data class Sample(val features: Tensor, val target: Tensor) : Serializable

typealias Params = Array<Tensor>

/** Wrapper that transports an array of gradients through Spark’s shuffle. */
class GradPacket(var grads: Params) : Serializable {
    /** In‑place addition (used by the tree aggregator). */
    fun addInPlace(other: GradPacket) {
        grads.indices.forEach { i ->
            val g = grads[i]
            val o = other.grads[i]
            for (j in g.data.indices) g.data[j] += o.data[j]
        }
    }

    companion object {
        fun zeroLike(params: Params): GradPacket = GradPacket(params.map { Tensor.zerosLike(it) }.toTypedArray())
    }
}