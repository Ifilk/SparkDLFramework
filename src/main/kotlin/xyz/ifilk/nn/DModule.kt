package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor

abstract class DModule {
    protected val _parameters: MutableList<Tensor> = mutableListOf()

    val parameters: List<Tensor>
        get() = _parameters.toList()

    /** 子类必须实现前向传播逻辑 */
    abstract fun forward(input: Tensor): Tensor

    /** 便于分布式训练框架调用参数清零 */
    fun zeroGrad() {
        for (param in _parameters) {
            if (param.requiresGrad) {
                param.grad?.zeroGrad()
            }
        }
    }

    /** 手动触发参数反向传播（如非 Autograd 管理的情况） */
    fun backward(loss: Tensor? = null) {
        loss?.backward()
    }

    /** 参数更新方法留空，通常由优化器管理 */
    open fun step() {
        // no-op; optim.step() 应该在外部控制
    }

    /** 注册一个需要梯度更新的参数 */
    protected fun registerParameter(tensor: Tensor) {
        if (!tensor.requiresGrad) {
            throw IllegalArgumentException("All model parameters must require gradients.")
        }
        _parameters.add(tensor)
    }
}
