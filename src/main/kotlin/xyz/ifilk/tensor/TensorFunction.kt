package xyz.ifilk.tensor

import xyz.ifilk.tensor.Tensor
import java.util.*

/**
 * Base class for every differentiable operation.
 */
abstract class TensorFunction {
    lateinit var inputs: Array<Tensor>
    var output: Tensor? = null

    /** Performs forward pass and returns the output tensor.  */
    abstract fun forward(vararg inputs: Tensor): Tensor

    /** Given upstream gradient, compute gradients w.r.t inputs.  */
    abstract fun backward(gradOutput: Tensor): Array<Tensor?>

    /** Utility: attach creator info to output  */
    fun attachCreator(out: Tensor) {
        out.creator = this
        out.requiresGrad = Arrays.stream(inputs).anyMatch { t: Tensor -> t.requiresGrad }
        out.isLeaf = false
        this.output = out
    }
}