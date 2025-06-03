package xyz.ifilk.tensor

class ReLUFunction : TensorFunction() {

    companion object {
        fun apply(input: Tensor): Tensor {
            val fn = ReLUFunction()
            fn.inputs = arrayOf(input)
            val out = fn.forward(input)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val outData = DoubleArray(input.data.size) { i ->
            if (input.data[i] > 0.0) input.data[i] else 0.0
        }
        return Tensor(outData, input.shape.clone(), input.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val input = inputs[0]
        val gradInput = DoubleArray(input.data.size) { i ->
            if (input.data[i] > 0.0) gradOutput.data[i] else 0.0
        }
        return arrayOf(Tensor(gradInput, input.shape.clone(), false))
    }
}
