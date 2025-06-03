package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor

fun mse(x: Tensor, y: Tensor): Tensor {
    return ((x - y).pow(2)).mean()
}

fun crossEntropyLoss(logits: Tensor, target: Tensor): Tensor {
    // 数值稳定：softmax 前先减去每行最大值
    val probs = logits.softmax()              // [batch, C]，自动求导 OK
    val eps = 1e-12                           // 防止 log(0)
    val logP = (probs + eps).log()            // 对数概率

    val t = target.reshape(*logP.shape)
    return (-(t * logP)).mean()               // 交叉熵 → 取 batch 均值
}