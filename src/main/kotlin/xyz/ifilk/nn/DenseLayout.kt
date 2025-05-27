package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor
import kotlin.math.sqrt
import kotlin.random.Random

class DenseLayout(
    private val inFeatures: Int,
    private val outFeatures: Int,
    private val useBias: Boolean = true
): Module() {
    // 权重参数：shape = [in, out]
    val weight = Tensor(
        data = DoubleArray(inFeatures * outFeatures) {
            // Xavier 初始化：U(−sqrt(6 / (in + out)), sqrt(6 / (in + out)))
            val bound = sqrt(6.0 / (inFeatures + outFeatures))
            Random.nextDouble(-bound, bound)
        },
        shape = intArrayOf(inFeatures, outFeatures),
        requiresGrad = true
    )

    // 偏置参数：shape = [out]
    val bias = if (useBias) {
        Tensor(
            data = DoubleArray(outFeatures) { 0.0 },
            shape = intArrayOf(outFeatures),
            requiresGrad = true
        )
    } else null

    // 前向传播
    override fun forward(input: Tensor): Tensor {
        var output = input.matmul(weight)
        if (useBias) {
            output += bias!!
        }
        return output
    }
}
