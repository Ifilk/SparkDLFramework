package xyz.ifilk.tensor

class FlattenFunction : TensorFunction() {

    companion object {
        fun apply(input: Tensor): Tensor {
            val fn = FlattenFunction()
            fn.inputs = arrayOf(input)
            val out = fn.forward(input)
            fn.attachCreator(out)
            return out
        }
    }

    private lateinit var originalShape: IntArray

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        originalShape = input.shape.clone()
        return Tensor(input.data.copyOf(), intArrayOf(input.data.size), input.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        // 反向传播：将梯度还原为原始形状
        return arrayOf(
            Tensor(gradOutput.data.copyOf(), originalShape, false)
        )
    }
}
