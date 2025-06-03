package xyz.ifilk.utils

import xyz.ifilk.nn.DModule
import xyz.ifilk.data.DataLoader
import xyz.ifilk.data.SimpleTensorDataLoader
import xyz.ifilk.tensor.Tensor

fun clipGradients(model: DModule, maxNorm: Double) {
    for (param in model.parameters) {
        param.grad?.coerceIn(-maxNorm, maxNorm)
    }
}

fun createDataLoader(
    x: Tensor,
    y: Tensor,
    batchSize: Int = 32,
    shuffle: Boolean = true
): DataLoader {
    return SimpleTensorDataLoader(x, y, batchSize, shuffle)
}

fun Tensor.oneHot(numClasses: Int): Tensor {
    require(this.shape.size == 1) { "Only 1D tensors can be one-hot encoded" }
    val n = this.shape[0]
    val out = DoubleArray(n * numClasses) { 0.0 }

    for (i in 0 until n) {
        val classIndex = this.data[i].toInt()
        require(classIndex in 0 until numClasses) { "Index $classIndex out of bounds for numClasses=$numClasses" }
        out[i * numClasses + classIndex] = 1.0
    }

    return Tensor(out, intArrayOf(n, numClasses), requiresGrad = false)
}

