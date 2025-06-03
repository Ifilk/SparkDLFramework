package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.tensor.TensorFunction

/**
 * 支持按轴 (axis) 求平均值的 MeanFunction。
 *
 * - 当 axis == null 时，返回所有元素的均值（标量）。
 * - 当 axis 指定时，沿给定轴求平均，返回该轴被压缩后的张量；
 *   当前实现聚焦于 **2‑D 输入张量**，且常见用法 axis == 0/1。
 *   如需更高维度扩展，可在 FIXME 标记处补充递归展开规则。
 */
class MeanFunction private constructor(private val axis: Int?) : TensorFunction() {

    companion object {
        /**
         * @param x     需要求均值的张量
         * @param axis  沿哪个轴求平均；null 表示所有元素
         */
        fun apply(x: Tensor, axis: Int? = null): Tensor {
            val fn = MeanFunction(axis)
            fn.inputs = arrayOf(x)
            val out = fn.forward(x)
            fn.attachCreator(out)
            return out
        }
    }

    /* 反向传播时需要用到的缓存 ------------------------------ */
    private lateinit var inputShape: IntArray
    private var axisSize: Int = 0         // 指定轴上的元素个数

    /* 前向传播 --------------------------------------------- */
    override fun forward(vararg inputs: Tensor): Tensor {
        val x = inputs[0]
        inputShape = x.shape

        // ➤ 情况一：整体平均（标量，与旧版一致）
        if (axis == null) {
            axisSize = x.data.size
            val meanVal = x.data.average()
            return Tensor(doubleArrayOf(meanVal), intArrayOf())
        }

        // ➤ 情况二：按轴平均（只实现 2‑D，常用于 BatchNorm）
        require(inputShape.size == 2) {
            "MeanFunction(axis) 目前仅支持 2‑D 输入，收到 shape=${inputShape.toList()}"
        }
        require(axis == 0 || axis == 1) { "axis 必须为 0 或 1" }

        val (batch, feat) = inputShape
        axisSize = if (axis == 0) batch else feat

        return if (axis == 0) {
            // → 结果 shape = [feat]
            val outData = DoubleArray(feat) { j ->
                var sum = 0.0
                var idx = j
                repeat(batch) {
                    sum += x.data[idx]
                    idx += feat
                }
                sum / batch
            }
            Tensor(outData, intArrayOf(feat))
        } else {
            // axis == 1 → 结果 shape = [batch]
            val outData = DoubleArray(batch) { i ->
                var sum = 0.0
                val base = i * feat
                for (j in 0 until feat) sum += x.data[base + j]
                sum / feat
            }
            Tensor(outData, intArrayOf(batch))
        }
    }

    /* 反向传播 --------------------------------------------- */
    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val totalElems = inputShape.reduce(Int::times)
        val gradData = DoubleArray(totalElems)

        if (axis == null) {
            // 标量均值：每个元素分得 grad / N
            val gradPerElem = gradOutput.data[0] / axisSize
            java.util.Arrays.fill(gradData, gradPerElem)
            return arrayOf(Tensor(gradData, inputShape))
        }

        // axis != null，仍假设 2‑D 输入
        val (batch, feat) = inputShape
        if (axis == 0) {
            // gradOutput shape = [feat]
            for (j in 0 until feat) {
                val g = gradOutput.data[j] / batch
                var idx = j
                repeat(batch) {
                    gradData[idx] = g
                    idx += feat
                }
            }
        } else {
            // axis == 1，gradOutput shape = [batch]
            for (i in 0 until batch) {
                val g = gradOutput.data[i] / feat
                val base = i * feat
                for (j in 0 until feat) gradData[base + j] = g
            }
        }
        return arrayOf(Tensor(gradData, inputShape))
    }
}
