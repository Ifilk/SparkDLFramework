package xyz.ifilk.tensor

class SumFunction : TensorFunction() {
    companion object {
        fun apply(input: Tensor): Tensor {
            val f = SumFunction()
            f.inputs = arrayOf(input)
            val out = f.forward(input)
            f.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        require(inputs.size == 1) { "SumFunction requires exactly 1 input" }
        val input = inputs[0]

        // Handle empty tensor case
        if (input.data.isEmpty()) {
            return Tensor(doubleArrayOf(0.0), intArrayOf(), requiresGrad = input.requiresGrad)
        }

        val sum = input.data.sum()
        return Tensor(doubleArrayOf(sum), intArrayOf(), requiresGrad = input.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        require(gradOutput.shape.isEmpty() || gradOutput.shape.size == 1) {
            "Gradient for sum operation must be scalar (empty shape)"
        }
        require(gradOutput.data.size == 1) {
            "Gradient for sum operation must have exactly 1 value"
        }

        val input = inputs[0]
        val gradValue = gradOutput.data[0]
        val grad = DoubleArray(input.data.size) { gradValue }

        return arrayOf(Tensor(grad, input.shape, requiresGrad = false))
    }
}