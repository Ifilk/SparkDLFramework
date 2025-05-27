package xyz.ifilk.tensor

class AddFunction : TensorFunction() {
    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val function = AddFunction()
            function.inputs = arrayOf(a, b)
            val output = function.forward(a, b)
            function.attachCreator(output)
            return output
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]

        require(a.shape.size <= 2 && b.shape.size <= 2) {
            "Only supports scalar, vector, or matrix addition"
        }

        val outputShape = a.shape
        val outputData = DoubleArray(a.data.size)

        when {
            a.shape.contentEquals(b.shape) -> {
                // Case 1: shape equal
                for (i in outputData.indices) {
                    outputData[i] = a.data[i] + b.data[i]
                }
            }

            b.shape.size == 1 && b.shape[0] == a.shape.last() -> {
                // Case 2: broadcast vector (e.g., [m,n] + [n])
                val m = a.shape[0]
                val n = a.shape[1]
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        outputData[i * n + j] = a.data[i * n + j] + b.data[j]
                    }
                }
            }

            b.shape.isEmpty() -> {
                // Case 3: scalar broadcast
                for (i in outputData.indices) {
                    outputData[i] = a.data[i] + b.data[0]
                }
            }

            else -> throw IllegalArgumentException("Unsupported broadcasting: a.shape=${a.shape.contentToString()}, b.shape=${b.shape.contentToString()}")
        }

        return Tensor(outputData, outputShape)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val a = inputs[0]
        val b = inputs[1]

        val gradA = gradOutput.clone()
        val gradB: Tensor = when {
            a.shape.contentEquals(b.shape) -> {
                // Case 1: same shapes
                gradOutput.clone()
            }

            b.shape.size == 1 && b.shape[0] == a.shape.last() -> {
                // Case 2: broadcast vector along last dimension (e.g., [m,n] + [n])
                val m = if (a.shape.size > 0) a.shape[0] else 1
                val n = if (a.shape.size > 1) a.shape[1] else a.shape[0]
                val data = DoubleArray(n)
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        val idx = if (a.shape.size > 1) i * n + j else j
                        data[j] += gradOutput.data[idx]
                    }
                }
                Tensor(data, b.shape)
            }

            b.shape.size == 1 && b.shape[0] == a.shape.first() -> {
                // Case 3: broadcast vector along first dimension (e.g., [m,n] + [m,1])
                val m = a.shape[0]
                val n = if (a.shape.size > 1) a.shape[1] else 1
                val data = DoubleArray(m)
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        val idx = if (a.shape.size > 1) i * n + j else i
                        data[i] += gradOutput.data[idx]
                    }
                }
                Tensor(data, b.shape)
            }

            b.shape.isEmpty() -> {
                // Case 4: scalar broadcast
                val sum = gradOutput.data.sum()
                Tensor(doubleArrayOf(sum), intArrayOf(1))
            }

            else -> throw IllegalStateException(
                "Unsupported backward broadcasting for a.shape=${a.shape.contentToString()}, " +
                        "b.shape=${b.shape.contentToString()}"
            )
        }

        return arrayOf(gradA, gradB)
    }
}
