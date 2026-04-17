# Load path: global → smem → rmem vs. global → rmem?

## Question

For a simple elementwise op (think elementwise add), how do I load a vector
from global memory into smem, then into rmem, do the op, and store the
result back to global?

And: is there a way to go straight from global to rmem and skip smem
entirely? When is that the right choice?

## Plan

Write both variants, check correctness, benchmark, and look at what the
compiler emitted.
