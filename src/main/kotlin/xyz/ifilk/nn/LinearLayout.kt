package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.utils.copyFrom
import kotlin.math.sqrt
import kotlin.random.Random

class LinearLayout(
    private val inFeatures: Int,
    private val outFeatures: Int,
    private val useBias: Boolean = true
): DModule() {

    // 权重参数：shape = [in, out]
    val weight = Parameter(
        name = "linear.weight",
        data = DoubleArray(inFeatures * outFeatures) {
            val bound = sqrt(6.0 / (inFeatures + outFeatures))
            Random.nextDouble(-bound, bound)
        },
        shape = intArrayOf(inFeatures, outFeatures),
        requiresGrad = true
    )

    // 偏置参数：shape = [out]
    val bias = if (useBias) {
        Parameter(
            name = "linear.bias",
            data = DoubleArray(outFeatures) { Random.nextDouble(0.0, 1.0) },
            shape = intArrayOf(outFeatures),
            requiresGrad = true
        )
    } else null

    init {
        registerParameter(weight)
        if (useBias) {
            registerParameter(bias!!)
        }
    }

    // 前向传播
    override fun forward(input: Tensor): Tensor {
        val output = input.matmul(weight)
        return if (useBias) {
            output + bias!! // 需支持 broadcast，加法自动图已修正
        } else {
            output
        }
    }

    override fun loadParameters(value: Array<Tensor>) {
        require(value.size == if (useBias) 2 else 1) {
            "Expected ${if (useBias) 2 else 1} tensors but got ${value.size}"
        }
        weight.copyFrom(value[0])
        if (useBias) bias!!.copyFrom(value[1])
    }
}
