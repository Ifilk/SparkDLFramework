package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor
import java.io.Serializable

abstract class DModule: Serializable {
    /* ---------- 可训练参数 ---------- */
    val _parameters: MutableList<Parameter> = mutableListOf()
    val parameters: List<Parameter>
        get() = _parameters.toList()

    /* ---------- 非训练 Buffer ---------- */
    private val _buffers: MutableMap<String, Tensor> = mutableMapOf()
    val buffers: Map<String, Tensor>
        get() = _buffers.toMap()

    // 训练模式标志
    var isTraining: Boolean = true

    open fun train() {
        isTraining = true
        onModeChanged()
    }

    open fun eval() {
        isTraining = false
        onModeChanged()
    }

    /**
     * 子类可以重写此方法，在 train/eval 状态切换时响应
     */
    protected open fun onModeChanged() {}

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
    protected fun registerParameter(tensor: Parameter) {
        if (!tensor.requiresGrad) {
            throw IllegalArgumentException("All model parameters must require gradients.")
        }
        _parameters.add(tensor)
    }

    /**
     * 注册一个 **不参与梯度**、但希望随模型保存/加载的张量。
     *
     * @param name        唯一名称
     * @param tensor      需注册的张量 (requiresGrad 必须为 false)
     * @param persistent  若为 false，则在保存 state_dict 时跳过
     */
    protected fun registerBuffer(
        name: String,
        tensor: Tensor,
        persistent: Boolean = true
    ) {
        require(!tensor.requiresGrad) { "Buffers must not require gradients." }
        require(name !in _buffers) { "Buffer '$name' already registered." }

        _buffers[name] = tensor.apply { isPersistent = persistent }
        // Tensor 可加一个可选字段 isPersistent，或由调用者自行管理
    }

    abstract fun loadParameters(value: Array<Tensor>)
}
