package xyz.ifilk.tensor

class MatmulFunction : TensorFunction() {
    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val f = MatmulFunction()
            f.inputs = arrayOf(a, b)
            val out = f.forward(a, b)
            f.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]
        val aShape = if (a.shape.size == 1) intArrayOf(1, a.shape[0]) else a.shape
        val bShape = if (b.shape.size == 1) intArrayOf(b.shape[0], 1) else b.shape
        val (m, k) = aShape
        val (k2, n) = bShape
        require(k == k2) {
            "Shape mismatch: ${a.shape.contentToString()} x ${b.shape.contentToString()}"
        }

        val result = DoubleArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0.0
                for (t in 0 until k) {
                    sum += a.data[i * k + t] * b.data[t * n + j]
                }
                result[i * n + j] = sum
            }
        }
        return Tensor(result, intArrayOf(m, n), requiresGrad = false)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor?> {
        val a = inputs[0]
        val b = inputs[1]
        val (m, k) = a.shape
        val n = b.shape[1]

        val gradA = DoubleArray(m * k)
        for (i in 0 until m) {
            for (j in 0 until k) {
                var sum = 0.0
                for (t in 0 until n) {
                    sum += gradOutput.data[i * n + t] * b.data[j * n + t]
                }
                gradA[i * k + j] = sum
            }
        }

        val gradB = DoubleArray(k * n)
        for (i in 0 until k) {
            for (j in 0 until n) {
                var sum = 0.0
                for (t in 0 until m) {
                    sum += a.data[t * k + i] * gradOutput.data[t * n + j]
                }
                gradB[i * n + j] = sum
            }
        }

        return arrayOf(
            Tensor(gradA, intArrayOf(m, k), requiresGrad = false),
            Tensor(gradB, intArrayOf(k, n), requiresGrad = false)
        )
    }
}
