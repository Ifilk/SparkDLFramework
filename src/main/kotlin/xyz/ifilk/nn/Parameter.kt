package xyz.ifilk.nn

import xyz.ifilk.tensor.Tensor
import java.io.Serializable

class Parameter(
    val name: String,
    override var data: DoubleArray,
    override var shape: IntArray,
    override var requiresGrad: Boolean
): Tensor(data, shape, requiresGrad), Serializable