package xyz.ifilk.autograd

import xyz.ifilk.tensor.Tensor
import java.util.*

class AutogradEngine {
    companion object  {
        /** Reverse‑mode autodiff using DFS topological ordering. */
        fun backward(tensor: Tensor) {
            // 1. Build a root‑first topological ordering using DFS with visited set
            val topo = ArrayList<Tensor>()
            val visited = HashSet<Tensor>()
            val stack = Stack<Tensor>()
            stack.push(tensor)
            while (stack.isNotEmpty()) {
                val node = stack.pop()
                if (!visited.add(node)) continue
                topo.add(node)
                if (node.creator != null)
                    for (input in node.creator!!.inputs)
                        stack.push(input)
            }
            // 2. Propagate gradients along the topological order (root → leaves)
            for (node in topo) {
                if (node.creator == null || node.grad == null) continue
                val grads = node.creator!!.backward(node.grad!!)
                for (i in grads.indices) {
                    val inp = node.creator!!.inputs[i]
                    if (inp.grad != null) {
                        // in-place operation
                        inp.grad!!._add(grads[i]!!)
                    } else inp.grad = grads[i]
                }
            }
        }
    }
}