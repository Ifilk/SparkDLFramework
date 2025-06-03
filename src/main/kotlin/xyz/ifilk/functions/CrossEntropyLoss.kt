package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor

class CrossEntropyLoss: Criticizer {
    override fun call(logits: Tensor, target: Tensor): Tensor {
        return crossEntropyLoss(logits, target)
    }
}