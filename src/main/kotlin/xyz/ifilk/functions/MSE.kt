package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor

class MSE: Criticizer {
    override fun call(logits: Tensor, target: Tensor): Tensor {
        return mse(logits, target)
    }
}