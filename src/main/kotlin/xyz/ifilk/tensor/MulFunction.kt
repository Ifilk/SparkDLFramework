package xyz.ifilk.tensor

class MulFunction: TensorFunction() {
    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val function = MulFunction()
            function.inputs = arrayOf(a, b)
            val output = function.forward(a, b)
            function.attachCreator(output)
            return output
        }
     }
    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]
        val out = a.clone()
        require(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]) {
            "Shape mismatch: ${a.shape.contentToString()} x ${b.shape.contentToString()}"
        }
        for (i in 0 until out.data.size) {
            out.data[i] *= b.data[i]
        }
        return out
    }

    override fun backward(gradOutput: Tensor): Array<Tensor?> {
        val a = inputs[0]
        val b = inputs[1]
        val gradA = DoubleArray(a.data.size)
        val gradB = DoubleArray(b.data.size)
        for (i in 0 until a.data.size) {
            gradA[i] = b.data[i] * gradOutput.data[i]
            gradB[i] = a.data[i] * gradOutput.data[i]
        }
        return arrayOf(
            Tensor(gradA, a.shape.clone(), false),
            Tensor(gradB, b.shape.clone(), false)
        )
    }
}