package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor
import java.io.Serializable

interface Criticizer: Serializable {
    fun call(logits: Tensor, target: Tensor): Tensor
}