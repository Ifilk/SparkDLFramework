package xyz.ifilk.optim

import xyz.ifilk.tensor.Tensor
import kotlin.math.pow
import kotlin.math.sqrt

class Adam(
    private val parameters: List<Tensor>,
    private val lr: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8
) : Optimizer {

    private val m = parameters.map { DoubleArray(it.data.size) { 0.0 } }
    private val v = parameters.map { DoubleArray(it.data.size) { 0.0 } }
    private var t = 0

    override fun step() {
        t += 1
        for ((index, param) in parameters.withIndex()) {
            val grad = param.grad?.data ?: continue
            val m_t = m[index]
            val v_t = v[index]

            for (i in grad.indices) {
                m_t[i] = beta1 * m_t[i] + (1 - beta1) * grad[i]
                v_t[i] = beta2 * v_t[i] + (1 - beta2) * grad[i] * grad[i]

                val mHat = m_t[i] / (1 - beta1.pow(t))
                val vHat = v_t[i] / (1 - beta2.pow(t))

                param.data[i] -= lr * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }

    override fun zeroGrad() {
        for (param in parameters) {
            param.grad?.zeroGrad()
        }
    }
}
