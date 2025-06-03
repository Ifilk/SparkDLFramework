package xyz.ifilk.optim

import xyz.ifilk.nn.DModule
import xyz.ifilk.tensor.Tensor
import xyz.ifilk.utils.times

class SGD(
    module: DModule,
    private var lr: Double
) : Optimizer {

    private val parameters = module.parameters

    override fun step(grads: Array<Tensor>?) {
        if (grads != null) {
            // 用传入的梯度更新参数
            for (i in parameters.indices) {
                val param = parameters[i]
                val grad = grads[i]
                for (j in param.data.indices) {
                    param.data[j] -= lr * grad.data[j]
                }
            }
        } else {
            // 如果没传梯度，就用参数自身的grad字段
            for (param in parameters) {
                val grad = param.grad ?: continue
                for (j in param.data.indices) {
                    param.data[j] -= lr * grad.data[j]
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
