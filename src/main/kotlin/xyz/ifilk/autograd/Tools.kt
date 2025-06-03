package xyz.ifilk.autograd

import xyz.ifilk.tensor.Tensor

fun toDotGraph(tensor: Tensor): String {
    return AutogradEngine.toDotGraph(tensor)
}