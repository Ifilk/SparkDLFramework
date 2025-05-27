package xyz.ifilk.optim

interface Optimizer {
    fun step()
    fun zeroGrad()
}
