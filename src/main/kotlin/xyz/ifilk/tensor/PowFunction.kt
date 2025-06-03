package xyz.ifilk.tensor


import kotlin.math.pow

class PowFunction(private val p: Double) : TensorFunction() {
    companion object {
        fun apply(x: Tensor, power: Double): Tensor {
            val fn = PowFunction(power)
            fn.inputs = arrayOf(x)
            val out = fn.forward(x)
            fn.attachCreator(out)
            return out
        }
    }

    override fun forward(vararg inputs: Tensor): Tensor {
        val x = inputs[0]
        val data = DoubleArray(x.data.size) { idx -> x.data[idx].pow(p) }
        return Tensor(data, x.shape, requiresGrad = x.requiresGrad)
    }

    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val x = inputs[0]
        val gradData = DoubleArray(x.data.size) { idx ->
            p * gradOutput.data[idx] * x.data[idx].pow(p - 1)
        }
        return arrayOf(Tensor(gradData, x.shape))
    }
}