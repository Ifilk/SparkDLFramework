import xyz.ifilk.tensor.Tensor
import xyz.ifilk.tensor.TensorFunction

/**
 * Matrix-multiplication with autograd support.
 *
 * Supports the following shapes
 *   • (m × k)  · (k × n)  → (m × n)
 *   • (m × k)  · (k)      → (m)      (matrix–vector)
 *   • (k)      · (k × n)  → (n)      (vector–matrix)
 *   • (k)      · (k)      → ()       (inner product → scalar)
 */
class MatmulFunction : TensorFunction() {

    companion object {
        fun apply(a: Tensor, b: Tensor): Tensor {
            val fn = MatmulFunction()
            fn.inputs = arrayOf(a, b)
            val out = fn.forward(a, b)
            fn.attachCreator(out)
            return out
        }
    }

    /* ---------- forward ---------- */

    override fun forward(vararg inputs: Tensor): Tensor {
        val a = inputs[0]
        val b = inputs[1]
        return a.clone().matmul_(b).also { it.requiresGrad = (a.requiresGrad || b.requiresGrad) }
    }

    /* ---------- backward ---------- */
    override fun backward(gradOutput: Tensor): Array<Tensor> {
        val a = inputs[0]
        val b = inputs[1]

        /* grad w.r.t. A  :  dL/dA = gradOut · Bᵀ   → shape (m,k) */
        val gradA = gradOutput.clone().matmul_(b.clone().T_())

        /* grad w.r.t. B  :  dL/dB = Aᵀ · gradOut   → shape (k,n) */
        val gradB = a.clone().T_().matmul_(gradOutput)


        return arrayOf(gradA, gradB)
    }
}
