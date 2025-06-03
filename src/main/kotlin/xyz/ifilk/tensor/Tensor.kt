package xyz.ifilk.tensor

import xyz.ifilk.autograd.AutogradEngine
import xyz.ifilk.functions.MeanFunction
import xyz.ifilk.functions.SoftmaxFunction
import xyz.ifilk.utils.haveSameShape
import java.io.Serializable
import kotlin.math.abs
import kotlin.math.sign
import kotlin.math.sqrt

/**
 * Core Tensor data structure with minimal autograd support.
 * Note: This is a *starting skeleton* – many optimizations (broadcasting, shape inference,
 * memory layout, GPU off‑loading, etc.) can be layered on later.
 */
open class Tensor(
    /** Flat data buffer in row‑major order.  */
    open var data: DoubleArray,
    /** Shape of the tensor, e.g. {2,3}.  */
    open var shape: IntArray,
    /** Whether to track gradients.  */
    open var requiresGrad: Boolean
): Serializable {
    /* ============== Fields ============== */
    /** Gradient accumulated during back‑prop.  */
    var grad: Tensor? = null

    /** Function that created this tensor (null for leaf).  */
    @Transient
    var creator: TensorFunction? = null

    /** Whether this tensor is a leaf (parameters / inputs).  */
    var isLeaf: Boolean = true

    var isPersistent: Boolean = false

    /* ============== Constructors ============== */
    constructor(data: DoubleArray, shape: IntArray) : this(data, shape, false)

    /** Convenience scalar constructor (leaf).  */
    constructor(scalar: Double, requiresGrad: Boolean=true) : this(doubleArrayOf(scalar), intArrayOf(1), requiresGrad)

    companion object {
        fun zerosLike(tensor: Tensor): Tensor{
            val newData = DoubleArray(tensor.data.size) { 0.0 }
            return Tensor(newData, tensor.shape)
        }

        fun stack(tensors: List<Tensor>): Tensor {
            require(tensors.isNotEmpty()) { "Tensor list cannot be empty" }

            // 所有 tensor 形状必须一致
            val baseShape = tensors[0].shape
            for (t in tensors) {
                require(t.shape.contentEquals(baseShape)) { "All tensors must have the same shape" }
            }

            val numTensors = tensors.size
            val newShape = intArrayOf(numTensors) + baseShape

            // 计算每个 tensor 中元素数目
            val singleSize = baseShape.fold(1) { acc, dim -> acc * dim }

            // 创建合并数据数组
            val stackedData = DoubleArray(numTensors * singleSize)

            for ((i, t) in tensors.withIndex()) {
                System.arraycopy(t.data, 0, stackedData, i * singleSize, singleSize)
            }

            return Tensor(stackedData, newShape)
        }

        fun fromIntArray(ints: IntArray, shape: IntArray): Tensor {
            require(ints.size == shape.reduce { a, b -> a * b }) {
                "Data size (${ints.size}) does not match shape ${shape.contentToString()}"
            }
            val data = DoubleArray(ints.size) { i -> ints[i].toDouble() }
            return Tensor(data, shape, requiresGrad = false)
        }
    }

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

    fun add(other: Double): Tensor {
        return AddFunction.apply(this, other)
    }

    fun substract(other: Tensor): Tensor {
        return SubtractFunction.apply(this, other)
    }

    fun substract(other: Double): Tensor {
        return SubtractFunction.apply(this, other)
    }

    operator fun plus(other: Tensor): Tensor {
        return this.add(other)
    }

    operator fun plus(other: Double): Tensor {
        return this.add(other)
    }

    operator fun minus(other: Tensor): Tensor {
        return this.substract(other)
    }

    operator fun minus(other: Double): Tensor {
        return this.substract(other)
    }

    operator fun unaryMinus(): Tensor {
        return NegFunction.apply(this)
    }

    fun add_(other: Tensor): Tensor {
        require(haveSameShape(this, other)) {
            "Shape mismatch: ${shape.contentToString()} x ${other.shape.contentToString()}"
        }
        for (i in data.indices)
            data[i] += other.data[i]
        return this
    }

    fun add_(other: Double): Tensor {
        for (i in data.indices)
            data[i] += other
        return this
    }

    fun mul(other: Tensor): Tensor {
        return MulFunction.apply(this, other)
    }

    fun mul(other: Double): Tensor {
        return MulFunction.apply(this, other)
    }

    operator fun times(other: Tensor): Tensor {
        return this.mul(other)
    }

    operator fun times(other: Double): Tensor {
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

    fun mul_(other: Double): Tensor {
        for (i in data.indices)
            data[i] *= other
        return this
    }

    fun matmul(other: Tensor): Tensor {
        return MatmulFunction.apply(this, other)
    }

    /**
     * 只支持 2-D/1-D 情况；更高维 batch-matmul 需另行扩展
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

    fun mean(axis: Int? = null): Tensor {
        return MeanFunction.apply(this, axis)
    }

    fun pow(i: Double): Tensor {
        return PowFunction.apply(this, i)
    }

    fun pow(i: Int): Tensor {
        return PowFunction.apply(this, i.toDouble())
    }

    /** Kick‑off backward pass (only valid   on scalar outputs).  */
    fun backward(grad: Tensor? = null) {
        check(data.size == 1) { "backward() can only be called on scalar outputs" }
        // Seed grad with 1.0
        this.grad = grad ?: Tensor(doubleArrayOf(1.0), intArrayOf(1), false)
        AutogradEngine.backward(this)
    }

    override fun toString(): String {
        return "Tensor(data=${data.contentToString()}, shape=${shape.contentToString()}, requiresGrad=$requiresGrad)"
    }

    val string: String
        get() {
            return formatString()
        }

    private fun formatString(): String {
        val sb = StringBuilder()
        sb.append("Tensor(data=[\n")
        for (i in data.indices) {
            if (i % shape[shape.size - 1] == 0)
                sb.append("[")
            sb.append(data[i])
            if (i != data.size - 1 && i % shape[shape.size - 1] != shape[shape.size - 1] - 1)
                sb.append(", ")
            if (i % shape[shape.size - 1] == shape[shape.size - 1] - 1)
                sb.append("]\n")
        }
        sb.append("])")
        return sb.toString()
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

    fun coerceIn(d: Double, maxNorm: Double) {
        val norm = sqrt(data.sumOf { it * it })
        if (norm > maxNorm) {
            val scale = maxNorm / norm
            for (i in data.indices) {
                data[i] *= scale
                if (abs(data[i]) < d)
                    data[i] = 0.0
                else
                    data[i] = sign(data[i]) * d
            }
        }
    }

    fun sqrt(): Tensor {
        return pow(0.5)
    }

    fun reciprocal(): Tensor {
        return ReciprocalFunction.apply(this)
    }

    fun fill(value: Double): Tensor {
        for (i in data.indices) {
            data[i] = value
        }
        return this
    }

    fun flatten(): Tensor {
        return FlattenFunction.apply(this)
    }

    fun softmax(axis: Int = -1): Tensor {
        return SoftmaxFunction.apply(this, axis)
    }

    fun log(): Tensor {
        return LogFunction.apply(this)
    }

    fun relu(): Tensor {
        return ReLUFunction.apply(this)
    }

    fun reshape(vararg newShape: Int): Tensor {
        val totalElements = this.data.size
        var inferredShape = newShape

        // 自动推导维度 -1
        val negOneCount = newShape.count { it == -1 }
        require(negOneCount <= 1) { "Only one dimension can be -1" }

        if (negOneCount == 1) {
            val knownProduct = newShape.filter { it != -1 }.fold(1) { acc, i -> acc * i }
            require(totalElements % knownProduct == 0) {
                "Cannot infer shape: total elements $totalElements not divisible by $knownProduct"
            }
            val inferred = totalElements / knownProduct
            inferredShape = newShape.map { if (it == -1) inferred else it }.toIntArray()
        }

        // 验证新形状匹配
        val expectedSize = inferredShape.fold(1) { acc, i -> acc * i }
        require(expectedSize == totalElements) {
            "Shape mismatch: expected $expectedSize elements, got $totalElements"
        }

        return Tensor(this.data.copyOf(), inferredShape, this.requiresGrad)
    }
}

