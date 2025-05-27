package xyz.ifilk.tensor

class AddFunction: TensorFunction() {
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
        require(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]) {
            "Shape mismatch: ${a.shape.contentToString()} x ${b.shape.contentToString()}"
        }
        return Tensor(a.data.mapIndexed { i, _ -> a.data[i] + b.data[i] }.toDoubleArray(), *a.shape)
    }


    override fun backward(gradOutput: Tensor): Array<Tensor?> {
        // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        val gradA = gradOutput.clone()
        val gradB = gradOutput.clone()
        return arrayOf(gradA, gradB)
    }
}