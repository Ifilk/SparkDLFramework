package xyz.ifilk.tensor

import kotlin.math.E
import kotlin.math.log

class LogFunction : TensorFunction() {

    companion object {
        fun apply(input: Tensor): Tensor {
            val fn = LogFunction()
            fn.inputs = arrayOf(input)
            val out = fn.forward(input)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val input = inputs[0]
        val data = input.data.map { log(it, E) }.toDoubleArray()
        return Tensor(data, input.shape.clone(), input.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val input = inputs[0]
        val grad = DoubleArray(input.data.size) { i ->
            gradOutput.data[i] / input.data[i]
        }
        return arrayOf(Tensor(grad, input.shape.clone(), false))
    }
}
