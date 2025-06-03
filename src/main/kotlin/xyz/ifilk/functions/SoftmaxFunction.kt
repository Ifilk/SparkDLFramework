package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.tensor.TensorFunction
import xyz.ifilk.utils.isVector
import kotlin.math.exp

class SoftmaxFunction(private val axis: Int = -1) : TensorFunction() {

    companion object {
        fun apply(input: Tensor, axis: Int = -1): Tensor {
            val fn = SoftmaxFunction(axis)
            fn.inputs = arrayOf(input)
            val out = fn.forward(input)
            fn.attachCreator(out)
            return out
        }
    }

    private lateinit var softmaxOutput: DoubleArray
    private lateinit var shape: IntArray

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val x = input.data
        shape = input.shape
        val ndim = shape.size

        // 处理负轴
        val axisIndex = if (axis < 0) ndim + axis else axis
        require(axisIndex in 0 until ndim) { "Invalid axis $axis for shape ${shape.contentToString()}" }

        val outerSize = shape.take(axisIndex).fold(1) { a, b -> a * b }
        val axisSize = shape[axisIndex]
        val innerSize = shape.drop(axisIndex + 1).fold(1) { a, b -> a * b }

        val output = DoubleArray(x.size)
        softmaxOutput = DoubleArray(x.size)

        for (outer in 0 until outerSize) {
            for (inner in 0 until innerSize) {
                val offset = outer * axisSize * innerSize + inner
                // 找最大值用于数值稳定
                var maxVal = Double.NEGATIVE_INFINITY
                for (i in 0 until axisSize) {
                    val idx = offset + i * innerSize
                    maxVal = maxOf(maxVal, x[idx])
                }
                // 计算 softmax
                var sumExp = 0.0
                for (i in 0 until axisSize) {
                    val idx = offset + i * innerSize
                    val e = exp(x[idx] - maxVal)
                    softmaxOutput[idx] = e
                    sumExp += e
                }
                for (i in 0 until axisSize) {
                    val idx = offset + i * innerSize
                    output[idx] = softmaxOutput[idx] / sumExp
                    softmaxOutput[idx] = output[idx] // 便于 backward 使用
                }
            }
        }

        return Tensor(output, shape.clone(), input.requiresGrad)
    }


    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val gradInput = DoubleArray(softmaxOutput.size)

        val axisIndex = if (axis < 0) shape.size + axis else axis
        val outerSize = shape.take(axisIndex).fold(1) { a, b -> a * b }
        val axisSize = shape[axisIndex]
        val innerSize = shape.drop(axisIndex + 1).fold(1) { a, b -> a * b }

        for (outer in 0 until outerSize) {
            for (inner in 0 until innerSize) {
                val offset = outer * axisSize * innerSize + inner
                for (i in 0 until axisSize) {
                    val idx_i = offset + i * innerSize
                    var sum = 0.0
                    for (j in 0 until axisSize) {
                        val idx_j = offset + j * innerSize
                        val delta = if (i == j) 1.0 else 0.0
                        val dyj_dxi = softmaxOutput[idx_j] * (delta - softmaxOutput[idx_i])
                        sum += gradOutput.data[idx_j] * dyj_dxi
                    }
                    gradInput[idx_i] = sum
                }
            }
        }

        return arrayOf(Tensor(gradInput, shape.clone(), false))
    }
}
