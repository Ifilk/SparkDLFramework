package xyz.ifilk.tensor

import xyz.ifilk.autograd.AutogradEngine

/**
 * Core Tensor data structure with minimal autograd support.
 * Note: This is a *starting skeleton* – many optimizations (broadcasting, shape inference,
 * memory layout, GPU off‑loading, etc.) can be layered on later.
 */
class Tensor(
    /** Flat data buffer in row‑major order.  */
    val data: DoubleArray,
    /** Shape of the tensor, e.g. {2,3}.  */
    val shape: IntArray,
    /** Whether to track gradients.  */
    var requiresGrad: Boolean
) {
    /* ============== Fields ============== */
    /** Gradient accumulated during back‑prop.  */
    var grad: Tensor? = null

    /** Function that created this tensor (null for leaf).  */
    var creator: TensorFunction? = null

    /** Whether this tensor is a leaf (parameters / inputs).  */
    var isLeaf: Boolean = true

    /* ============== Constructors ============== */
    constructor(data: DoubleArray, vararg shape: Int) : this(data, shape, false)

    /** Convenience scalar constructor (leaf).  */
    constructor(scalar: Double, requiresGrad: Boolean) : this(doubleArrayOf(scalar), intArrayOf(1), requiresGrad)

    fun add(other: Tensor): Tensor {
        return AddFunction.apply(this, other)
    }

    fun _add(other: Tensor): Tensor {
        for (i in data.indices)
            data[i] += other.data[i]
        return this
    }

    fun mul(other: Tensor): Tensor {
        return MulFunction.apply(this, other)
    }

    fun matmul(other: Tensor): Tensor {
        return MatmulFunction.apply(this, other)
    }



    /** Kick‑off backward pass (only valid on scalar outputs).  */
    fun backward() {
        check(data.size == 1) { "backward() can only be called on scalar outputs" }
        // Seed grad with 1.0
        this.grad = Tensor(doubleArrayOf(1.0), intArrayOf(1), false)
        AutogradEngine.backward(this)
    }

    override fun toString(): String {
        return "Tensor(data=${data.contentToString()}, shape=${shape.contentToString()}, requiresGrad=$requiresGrad)"
    }

    fun clone(): Tensor {
        return Tensor(data.copyOf(), shape.copyOf(), requiresGrad)
    }
}

