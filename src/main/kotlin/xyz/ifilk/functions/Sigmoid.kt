package xyz.ifilk.functions

import xyz.ifilk.tensor.Tensor
import xyz.ifilk.tensor.TensorFunction

class Sigmoid : TensorFunction() {
    companion object {
        fun apply(input: Tensor): Tensor {
            val function = Sigmoid()
            function.inputs = arrayOf(input)
            val output = function.forward(input)
            function.attachCreator(output)
            return output
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val outputData = input.data.map { x -> 1.0 / (1.0 + kotlin.math.exp(-x)) }.toDoubleArray()
        return Tensor(outputData, input.shape)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val output = output ?: throw IllegalStateException("Output tensor not found")
        val gradInputData = DoubleArray(output.data.size) { i ->
            val y = output.data[i]
            gradOutput.data[i] * y * (1 - y)
        }
        return arrayOf(Tensor(gradInputData, output.shape))
    }
}
