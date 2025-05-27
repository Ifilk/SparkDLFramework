package xyz.ifilk.tensor

import xyz.ifilk.autograd.AutogradEngine
import xyz.ifilk.utils.haveSameShape

/**
 * Core Tensor data structure with minimal autograd support.
 * Note: This is a *starting skeleton* – many optimizations (broadcasting, shape inference,
 * memory layout, GPU off‑loading, etc.) can be layered on later.
 */
class Tensor(
    /** Flat data buffer in row‑major order.  */
    var data: DoubleArray,
    /** Shape of the tensor, e.g. {2,3}.  */
    var shape: IntArray,
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
    constructor(data: DoubleArray, shape: IntArray) : this(data, shape, false)

    /** Convenience scalar constructor (leaf).  */
    constructor(scalar: Double, requiresGrad: Boolean) : this(doubleArrayOf(scalar), intArrayOf(1), requiresGrad)

    fun zeroGrad() {
        if (requiresGrad && grad != null) {
            grad = Tensor(
                data = DoubleArray(data.size) { 0.0 },
                shape = shape.copyOf(),
                requiresGrad = false
            )
        }
    }

    fun T(): Tensor {
        return transpose()
    }

    fun T_(): Tensor {
        return transpose_()
    }


    fun transpose(axes: IntArray? = null): Tensor {
        return TransposeFunction.apply(this, axes)
    }

    fun transpose_(axes: IntArray? = null): Tensor {
        val _axes = axes?: shape.indices.reversed().toList().toIntArray()
        val originalStrides = computeStrides(shape)
        val newShape = _axes.map { shape[it] }.toIntArray()
        val newStrides = computeStrides(newShape)

        val result = DoubleArray(data.size)
        val idx = IntArray(shape.size)

        for (i in data.indices) {
            var remainder = i
            for (j in shape.indices) {
                idx[j] = remainder / originalStrides[j]
                remainder %= originalStrides[j]
            }
            val newIdx = _axes.map { idx[it] }.toIntArray()
            val newPos = newIdx.indices.sumOf { newIdx[it] * newStrides[it] }
            result[newPos] = data[i]
        }

        data = result
        shape = newShape
        return this
    }

    fun add(other: Tensor): Tensor {
        return AddFunction.apply(this, other)
    }

    operator fun plus(other: Tensor): Tensor {
        return this.add(other)
    }

    fun add_(other: Tensor): Tensor {
        require(haveSameShape(this, other)) {
            "Shape mismatch: ${shape.contentToString()} x ${other.shape.contentToString()}"
        }
        for (i in data.indices)
            data[i] += other.data[i]
        return this
    }

    fun mul(other: Tensor): Tensor {
        return MulFunction.apply(this, other)
    }

    operator fun times(other: Tensor): Tensor {
        return this.mul(other)
    }

    infix fun mm(other: Tensor): Tensor {
        return this.matmul(other)
    }

    fun mul_(other: Tensor): Tensor {
        require(haveSameShape(this, other)) {
            "Shape mismatch: ${shape.contentToString()} x ${other.shape.contentToString()}"
        }
        for (i in data.indices)
            data[i] *= other.data[i]
        return this
    }

    fun matmul(other: Tensor): Tensor {
        return MatmulFunction.apply(this, other)
    }

    /**
     * In-place matrix multiplication: this ← this · other
     *
     * 约定：
     *   • 允许 1-D 向量参与运算（自动晋升为行/列张量）
     *   • 只支持 2-D/1-D 情况；更高维 batch-matmul 需另行扩展
     *   • requiresGrad 取两输入的逻辑或
     *
     * 使用示例：
     *   val a = Tensor(doubleArrayOf(1.0,2.0,3.0,4.0), intArrayOf(2,2))
     *   val b = Tensor(doubleArrayOf(5.0,6.0,7.0,8.0), intArrayOf(2,2))
     *   a.matmul_(b)     // a 被就地更新
     */
    fun matmul_(other: Tensor): Tensor {
        val aShape = if (this.shape.size == 1) intArrayOf(1, this.shape[0]) else this.shape
        val bShape = if (other.shape.size == 1) intArrayOf(other.shape[0], 1) else other.shape
        val (m, k) = aShape
        val (k2, n) = bShape
        require(k == k2) {
            "Shape mismatch for matmul_: ${this.shape.contentToString()} x ${other.shape.contentToString()}"
        }

        /* --- 计算结果 --- */
        val out = DoubleArray(m * n)
        for (i in 0 until m)            // rows of A
            for (j in 0 until n) {      // cols of B
                var sum = 0.0
                for (t in 0 until k)    // shared dim
                    sum += this.data[i * k + t] * other.data[t * n + j]
                out[i * n + j] = sum
            }

        /* --- 还原输出形状（与 MatmulFunction.forward 完全一致） --- */
        val outShape = when {
            this.shape.size == 1 && other.shape.size == 1 -> intArrayOf()        // scalar
            this.shape.size == 1 -> intArrayOf(n)                                // row-vector
            other.shape.size == 1 -> intArrayOf(m)                               // column-vector
            else -> intArrayOf(m, n)                                             // matrix
        }

        /* --- 就地更新当前张量 --- */
        this.data = out
        this.shape = outShape
        this.requiresGrad = this.requiresGrad || other.requiresGrad

        return this
    }


    fun sum(): Tensor {
        return SumFunction.apply(this)
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

    private fun computeStrides(shape: IntArray): IntArray {
        val strides = IntArray(shape.size)
        var acc = 1
        for (i in shape.indices.reversed()) {
            strides[i] = acc
            acc *= shape[i]
        }
        return strides
    }

}

