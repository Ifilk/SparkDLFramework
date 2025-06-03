package xyz.ifilk.tensor

class ReciprocalFunction : TensorFunction() {

    companion object {
        fun apply(x: Tensor): Tensor {
            val fn = ReciprocalFunction()
            fn.inputs = arrayOf(x)
            val out = fn.forward(x)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val x = inputs[0]
        val data = DoubleArray(x.data.size) { i -> 1.0 / x.data[i] }
        return Tensor(data, x.shape)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val x = inputs[0]
        val grad = DoubleArray(x.data.size) { i ->
            // ∂(1/x)/∂x = -1/x^2
            -gradOutput.data[i] / (x.data[i] * x.data[i])
        }
        return arrayOf(Tensor(grad, x.shape))
    }
}
