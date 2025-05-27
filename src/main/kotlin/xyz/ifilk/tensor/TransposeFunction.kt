package xyz.ifilk.tensor

class TransposeFunction : TensorFunction() {
    companion object {
        fun apply(input: Tensor, axes: IntArray? = null): Tensor {
            val f = TransposeFunction()
            f.inputs = arrayOf(input)
            f.axes = axes ?: input.shape.indices.reversed().toList().toIntArray() // 默认反转维度
            val out = f.forward(input)
            f.attachCreator(out)
            return out
        }
    }

    private lateinit var axes: IntArray

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        return input.clone().transpose_(axes)


    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val input = inputs[0]
        val inverseAxes = IntArray(axes.size)
        for (i in axes.indices) {
            inverseAxes[axes[i]] = i
        }
        return arrayOf(gradOutput.transpose_(inverseAxes).also { it.requiresGrad = input.requiresGrad })
    }
}
