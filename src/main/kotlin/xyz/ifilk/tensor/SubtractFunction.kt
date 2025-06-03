package xyz.ifilk.tensor

import xyz.ifilk.utils.isScaler

/**
 * 执行逐元素减法：output = a - b
 * 支持：
 *   1. 形状完全一致
 *   2. [m,n] - [n]     （向量沿最后一维广播）
 *   3. [m,n] - [m,1]   （向量沿第一维广播）
 *   4. 标量广播
 */
class SubtractFunction : TensorFunction() {

    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val fn = SubtractFunction()
            fn.inputs = arrayOf(a, b)
            val out = fn.forward(a, b)
            fn.attachCreator(out)
            return out
        }

        fun apply(a: Tensor, b: Double): Tensor {
            val tb = Tensor(b)
            val fn = SubtractFunction()
            fn.inputs = arrayOf(a, tb)
            val out = fn.forward(a, tb)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]

        require(a.shape.size <= 2 && b.shape.size <= 2) {
            "Only supports scalar, vector, or matrix subtraction"
        }

        val outputShape = a.shape
        val outputData = DoubleArray(a.data.size)

        when {
            // Case 1: same shape
            a.shape.contentEquals(b.shape) -> {
                for (i in outputData.indices) {
                    outputData[i] = a.data[i] - b.data[i]
                }
            }

            // Case 2: [m,n] - [n]
            b.shape.size == 1 && b.shape[0] == a.shape.last() -> {
                val m = a.shape[0]
                val n = a.shape[1]
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        outputData[i * n + j] = a.data[i * n + j] - b.data[j]
                    }
                }
            }

            // Case 3: [m,n] - [m,1]  or  [m] - [m]
            b.shape.size == 1 && b.shape[0] == a.shape.first() -> {
                val m = a.shape[0]
                val n = if (a.shape.size > 1) a.shape[1] else 1
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        val idx = if (a.shape.size > 1) i * n + j else i
                        outputData[idx] = a.data[idx] - b.data[i]
                    }
                }
            }

            // Case 4: scalar broadcast
            b.isScaler() -> {
                for (i in outputData.indices) {
                    outputData[i] = a.data[i] - b.data[0]
                }
            }

            else -> throw IllegalArgumentException(
                "Unsupported broadcasting: a.shape=${a.shape.contentToString()}, b.shape=${b.shape.contentToString()}"
            )
        }

        return Tensor(outputData, outputShape)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val a = inputs[0]
        val b = inputs[1]

        // ∂L/∂a = gradOutput
        val gradA = gradOutput.clone()

        // ∂L/∂b = -gradOutput（再根据广播规则进行缩并）
        val gradB: Tensor = when {
            // same shape
            a.shape.contentEquals(b.shape) -> {
                gradOutput.clone().apply { for (i in data.indices) data[i] = -data[i] }
            }

            // b 是最后一维向量 [n]
            b.shape.size == 1 && b.shape[0] == a.shape.last() -> {
                val m = a.shape[0]
                val n = a.shape[1]
                val data = DoubleArray(n)
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        data[j] -= gradOutput.data[i * n + j]      // 注意加负号
                    }
                }
                Tensor(data, b.shape)
            }

            // b 是第一维向量 [m] 或 [m,1]
            b.shape.size == 1 && b.shape[0] == a.shape.first() -> {
                val m = a.shape[0]
                val n = if (a.shape.size > 1) a.shape[1] else 1
                val data = DoubleArray(m)
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        val idx = if (a.shape.size > 1) i * n + j else i
                        data[i] -= gradOutput.data[idx]
                    }
                }
                Tensor(data, b.shape)
            }

            // scalar
            b.shape.isEmpty() -> {
                Tensor(doubleArrayOf(-gradOutput.data.sum()), intArrayOf(1))
            }

            else -> throw IllegalStateException(
                "Unsupported backward broadcasting: a.shape=${a.shape.contentToString()}, b.shape=${b.shape.contentToString()}"
            )
        }

        return arrayOf(gradA, gradB)
    }
}


