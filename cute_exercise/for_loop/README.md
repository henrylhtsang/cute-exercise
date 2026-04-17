# How bad is a `for` loop in CuTe DSL?

## Question

Say I'm writing an elementwise op in CuTe DSL — think elementwise add. The
simplest thing is to loop over elements with a plain Python `for` loop. Is
that a problem?

Some angles:

- Is there a vectorized add / scale primitive I should reach for instead?
- Or is the thing I really care about just vectorized loads and stores, and
  the `for` over the register tile doesn't actually matter?
- Does it change anything if the loop bound is a compile-time constant vs. a
  runtime value?

## Plan

Write a couple of elementwise-add variants, check correctness, benchmark, and
peek at the generated PTX/SASS to see what actually came out.
