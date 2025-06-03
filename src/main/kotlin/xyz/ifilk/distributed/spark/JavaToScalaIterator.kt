package xyz.ifilk.distributed.spark

import scala.collection.Iterator

class JavaToScalaIterator<T>(private val javaIt: kotlin.collections.Iterator<T>) : Iterator<T> {
    override fun hasNext(): Boolean = javaIt.hasNext()
    override fun next(): T = javaIt.next()
}
