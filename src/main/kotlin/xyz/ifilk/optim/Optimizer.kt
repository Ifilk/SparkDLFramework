package xyz.ifilk.optim

import xyz.ifilk.tensor.Tensor
import java.io.Serializable

interface Optimizer: Serializable {
    fun step(grads: Array<Tensor>? = null)
    fun zeroGrad()
}
