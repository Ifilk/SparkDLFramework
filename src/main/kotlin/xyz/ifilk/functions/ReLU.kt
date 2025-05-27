package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.tensor.TensorFunction

class ReLU : TensorFunction() {
    companion object {
        fun apply(input: Tensor): Tensor {
            val function = ReLU()
            function.inputs = arrayOf(input)  // 这里赋值inputs，必须有
            val output = function.forward(input)
            function.attachCreator(output)
            return output
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val outputData = input.data.map { if (it > 0) it else 0.0 }.toDoubleArray()
        return Tensor(outputData, input.shape)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val input = inputs[0]
        val gradInputData = DoubleArray(input.data.size) { i ->
            if (input.data[i] > 0) gradOutput.data[i] else 0.0
        }
        return arrayOf(Tensor(gradInputData, input.shape))
    }
}
