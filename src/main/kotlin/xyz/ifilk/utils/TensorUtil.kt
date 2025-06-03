package xyz.ifilk.utils

import xyz.ifilk.tensor.Tensor
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random


internal object TensorSerde {
    fun serialize(tensor: Tensor, out: DataOutputStream) {
        out.writeInt(tensor.shape.size)
        tensor.shape.forEach(out::writeInt)
        out.writeInt(tensor.data.size)
        tensor.data.forEach(out::writeDouble)
    }

    fun deserialize(inp: DataInputStream): Tensor {
        val dim = inp.readInt()
        val shape = IntArray(dim) { inp.readInt() }
        val len = inp.readInt()
        val data = DoubleArray(len) { inp.readDouble() }
        return Tensor(data, shape)
    }
}

/**
 * 将 Tensor 打包为 ByteArray（Little-Endian）。
 */
fun serializeTensor(t: Tensor): ByteArray {
    val shapeLen = t.shape.size
    val dataLen  = t.data.size

    // 计算总字节数：shapeLen(4) + shape*4 + requiresGrad(1) + data*8
    val totalBytes = 4 + shapeLen * 4 + 1 + dataLen * 8
    val buf = ByteBuffer
        .allocate(totalBytes)
        .order(ByteOrder.LITTLE_ENDIAN)

    // 1️⃣ shape 信息
    buf.putInt(shapeLen)
    for (dim in t.shape) buf.putInt(dim)

    // 2️⃣ requiresGrad
    buf.put(if (t.requiresGrad) 1 else 0)

    // 3️⃣ 数据
    for (d in t.data) buf.putDouble(d)

    return buf.array()
}

/**
 * 从 ByteArray 解析出 Tensor（Little-Endian）。
 *
 * @throws IllegalArgumentException 如果字节长度与 shape 不匹配
 */
fun deserializeTensor(bytes: ByteArray): Tensor {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

    // 1️⃣ 取 shape
    val shapeLen = buf.int
    val shape = IntArray(shapeLen) { buf.int }

    // 2️⃣ 取 requiresGrad
    val requiresGrad = buf.get().toInt() != 0

    // 3️⃣ 取数据
    val expectedDataLen = shape.reduce(Int::times)
    val data = DoubleArray(expectedDataLen) { buf.double }

    require(!buf.hasRemaining()) { "Extra bytes detected in serialized tensor" }

    return Tensor(data, shape, requiresGrad)
}

fun rand(vararg shape: Int): Tensor {
    val totalSize = shape.fold(1) { acc, dim -> acc * dim }
    val data = DoubleArray(totalSize) { Random.nextDouble() }
    return Tensor(data, shape, true)
}

fun Tensor.copyFrom(src: Tensor) {
    require(data.size == src.data.size && shape.contentEquals(src.shape))
    for (i in data.indices) data[i] = src.data[i]
}

fun haveSameShape(t1: Tensor, t2: Tensor): Boolean {
    // Handle scalar cases
    val shape1 = normalizeShape(t1.shape)
    val shape2 = normalizeShape(t2.shape)

    // Compare normalized shapes
    return shape1.contentEquals(shape2)
}

// Helper function to normalize shapes (e.g., [] and [1] both represent scalars)
private fun normalizeShape(shape: IntArray): IntArray {
    return when {
        // Empty shape (scalar) becomes [1]
        shape.isEmpty() -> intArrayOf(1)
        // Shape with single 1 (e.g., [1], [1,1]) gets reduced
        shape.all { it == 1 } -> IntArray(1) { 1 }
        else -> shape
    }
}

fun Tensor.takeRows(rows: IntArray): Tensor {
    val rowSize = this.shape[1]
    val newData = DoubleArray(rows.size * rowSize) { i ->
        val row = rows[i / rowSize]
        val col = i % rowSize
        this.data[row * rowSize + col]
    }
    return Tensor(newData, intArrayOf(rows.size, rowSize))
}

fun Tensor.isScaler(): Boolean {
    val shape = normalizeShape(this.shape)
     return shape.size == 1 || shape.size == 2 && shape[0] == 1
}

fun Tensor.isVector(): Boolean {
    val shape = normalizeShape(this.shape)
    return (shape.size == 1) || (shape.size == 2 && (shape[0] == 1 || shape[1] == 1))
}


operator fun Double.times(tensor: Tensor): Tensor {
    return tensor * this
}


fun fromIntArray(ints: IntArray, shape: IntArray): Tensor {
    require(ints.size == shape.reduce { a, b -> a * b }) {
        "Data size (${ints.size}) does not match shape ${shape.contentToString()}"
    }
    val data = DoubleArray(ints.size) { i -> ints[i].toDouble() }
    return Tensor(data, shape, requiresGrad = false)
}