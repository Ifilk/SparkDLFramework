package xyz.ifilk.optim

import xyz.ifilk.nn.DModule
import xyz.ifilk.tensor.Tensor
import kotlin.math.pow
import kotlin.math.sqrt

class Adam(
    model: DModule,
    private val lr: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8
) : Optimizer {
    private val parameters = model.parameters

    private val m = parameters.map { DoubleArray(it.data.size) { 0.0 } }
    private val v = parameters.map { DoubleArray(it.data.size) { 0.0 } }
    private var t = 0

    override fun step(grads: Array<Tensor>?) {
        t += 1
        if (grads != null) {
            for (i in parameters.indices) {
                val param = parameters[i]
                val grad = grads[i].data
                val m_t = m[i]
                val v_t = v[i]

                for (j in grad.indices) {
                    m_t[j] = beta1 * m_t[j] + (1 - beta1) * grad[j]
                    v_t[j] = beta2 * v_t[j] + (1 - beta2) * grad[j] * grad[j]

                    val mHat = m_t[j] / (1 - beta1.pow(t))
                    val vHat = v_t[j] / (1 - beta2.pow(t))

                    param.data[j] -= lr * mHat / (sqrt(vHat) + epsilon)
                }
            }
        } else {
            // 如果没传入梯度，使用参数内部的grad
            for (i in parameters.indices) {
                val param = parameters[i]
                val gradData = param.grad?.data ?: continue
                val m_t = m[i]
                val v_t = v[i]

                for (j in gradData.indices) {
                    m_t[j] = beta1 * m_t[j] + (1 - beta1) * gradData[j]
                    v_t[j] = beta2 * v_t[j] + (1 - beta2) * gradData[j] * gradData[j]

                    val mHat = m_t[j] / (1 - beta1.pow(t))
                    val vHat = v_t[j] / (1 - beta2.pow(t))

                    param.data[j] -= lr * mHat / (sqrt(vHat) + epsilon)
                }
            }
        }
    }

    override fun zeroGrad() {
        for (param in parameters) {
            param.grad?.zeroGrad()
        }
    }
}
