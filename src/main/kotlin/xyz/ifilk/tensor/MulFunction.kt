package xyz.ifilk.tensor

import xyz.ifilk.utils.haveSameShape
import xyz.ifilk.utils.isScaler

class MulFunction : TensorFunction() {

    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val fn = MulFunction()
            fn.inputs = arrayOf(a, b)
            val out = fn.forward(a, b)
            fn.attachCreator(out)
            return out
        }

        fun apply(a: Tensor, b: Double): Tensor {
            return apply(a, Tensor(doubleArrayOf(b), intArrayOf(1), false))
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]

        require(
            haveSameShape(a, b) || a.isScaler() || b.isScaler()
        ) {
            "Shape mismatch: ${a.shape.contentToString()} x ${b.shape.contentToString()}"
        }

        // 复制大的那端，避免无谓扩容
        val out = if (!a.isScaler()) a.clone() else b.clone()

        when {
            haveSameShape(a, b)   -> out.mul_(b)                             // 逐元素
            b.isScaler()          -> out.mul_(b.data[0])                     // a * scalar
            a.isScaler()          -> { out.mul_(a.data[0]) }                 // scalar * b
        }
        return out
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val a = inputs[0]
        val b = inputs[1]

        val aIsScalar = a.isScaler()
        val bIsScalar = b.isScaler()

        val gradA = DoubleArray(a.data.size)
        val gradB = DoubleArray(b.data.size)

        when {
            // 普通逐元素乘
            !aIsScalar && !bIsScalar -> {
                for (i in gradA.indices) {
                    gradA[i] = b.data[i] * gradOutput.data[i]
                    gradB[i] = a.data[i] * gradOutput.data[i]
                }
            }

            // a 是张量, b 是标量
            !aIsScalar && bIsScalar -> {
                val k = b.data[0]
                var sum = 0.0
                for (i in gradA.indices) {
                    gradA[i] = k * gradOutput.data[i]
                    sum += a.data[i] * gradOutput.data[i]
                }
                gradB[0] = sum
            }

            // a 是标量, b 是张量
            aIsScalar && !bIsScalar -> {
                val k = a.data[0]
                var sum = 0.0
                for (i in gradB.indices) {
                    gradB[i] = k * gradOutput.data[i]
                    sum += b.data[i] * gradOutput.data[i]
                }
                gradA[0] = sum
            }

            // 两边都是标量（退化情况）
            else -> {
                gradA[0] = b.data[0] * gradOutput.data[0]
                gradB[0] = a.data[0] * gradOutput.data[0]
            }
        }

        return arrayOf(
            Tensor(gradA, a.shape.clone(), false),
            Tensor(gradB, b.shape.clone(), false)
        )
    }
}
