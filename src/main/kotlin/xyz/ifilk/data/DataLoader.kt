package xyz.ifilk.data

import xyz.ifilk.tensor.Tensor

interface DataLoader : Iterable<Pair<Tensor, Tensor>> {
    val numBatches: Int
}
