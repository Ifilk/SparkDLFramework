package xyz.ifilk.optim

import xyz.ifilk.tensor.Tensor

class SGD(
    private val parameters: List<Tensor>,
    private val lr: Double
) : Optimizer {

    override fun step() {
        for (param in parameters) {
            if (param.grad != null) {
                for (i in param.data.indices) {
                    param.data[i] -= lr * param.grad!!.data[i]
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
