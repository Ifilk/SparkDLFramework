package xyz.ifilk.tensor

class NegFunction : TensorFunction() {
    companion object {
        fun apply(input: Tensor): Tensor {
            val fn = NegFunction()
            fn.inputs = arrayOf(input)
            val out = fn.forward(input)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val data = DoubleArray(input.data.size) { i -> -input.data[i] }
        return Tensor(data, input.shape.clone(), input.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val grad = DoubleArray(gradOutput.data.size) { i -> -gradOutput.data[i] }
        return arrayOf(Tensor(grad, gradOutput.shape.clone(), false))
    }
}
