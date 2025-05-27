package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor


abstract class Module {
    val parameters: MutableList<Tensor> = mutableListOf()

    abstract fun forward(input: Tensor): Tensor

    fun zeroGrad() {
        parameters.forEach { param ->
            if (param.requiresGrad) {
                // 这里假设 Tensor 类有 grad 属性和 zeroGrad() 方法
                param.grad?.zeroGrad()
            }
        }
    }
}