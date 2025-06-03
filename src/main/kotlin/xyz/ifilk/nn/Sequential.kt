package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor

/**
 * 按顺序串联多个 DModule 的容器
 *
 * @param modules 需要顺序执行的子模块，可变参数或列表皆可。
 */
class Sequential(private vararg val modules: DModule) : DModule() {

    init {
        // 收集所有子模块的参数，供 Optimizer 统一访问
        modules.forEach { _parameters.addAll(it.parameters) }
    }

    override fun train() {
        modules.forEach { it.train() }
    }

    override fun eval() {
        modules.forEach { it.eval() }
    }

    override fun forward(input: Tensor): Tensor {
        var out = input
        for (m in modules) {
            out = m.forward(out)
        }
        return out
    }

    /**
     * 按顺序把外部提供的参数分段加载进各子模块。
     * @param value 整个 Sequential 的参数数组（顺序与 parameters 属性一致）
     */
    override fun loadParameters(value: Array<Tensor>) {
        var cursor = 0
        for (m in modules) {
            val subCount = m.parameters.size
            val slice = value.sliceArray(cursor until cursor + subCount)
            m.loadParameters(slice)
            cursor += subCount
        }
        // 重新同步自身的参数引用（子模块可能换了新 Tensor）
        _parameters.clear()
        modules.forEach { _parameters.addAll(it.parameters) }
    }

    /**
     * 如果希望手动更新参数（很少用到，因为通常由 Optimizer 管理），
     * 可以把请求下发到各子模块：
     */
    override fun step() {
        modules.forEach { it.step() }
    }
}
