"""Shape-dispatched FP16 vector add with inline PTX variants.

Operation:

    out = a + b

for contiguous ``(N, N)`` ``torch.float16`` tensors.  The exercise is mostly
about controlling the exact instruction shape for a memory-bound kernel on
B200/GB200-class hardware: compile-time threads, persistent CTA count,
per-thread vector width, cache policy, 128-bit vs 256-bit loads, and half2 math.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from cute_exercise.base import CuteDSLKernel


B200_SM_COUNT = 148
B200_SHAPES = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class VectorAddConfig:
    name: str
    variant: str
    threads: int
    cta_per_sm: int
    elems_per_thread: int
    vector_words: int
    cache_policy: str = "default"
    store_policy: str = "default"
    math_op: str = "add"
    unroll: int = 1
    schedule: str = "persistent"
    tiles_per_cta: int = 1
    assume_full_tiles: bool = False
    assumed_align: int = 16
    op_order: str = "interleaved"

    @property
    def elems_per_cta(self) -> int:
        return self.threads * self.elems_per_thread


CONFIGS = (
    VectorAddConfig("dsl_vectorized_t512", "dsl", 512, 2, 8, 4),
    VectorAddConfig(
        "ptx_h2_scalar_one_tile_fma_unroll4",
        "ptx",
        128,
        4,
        8,
        1,
        math_op="fma",
        unroll=4,
        schedule="one_tile",
    ),
    VectorAddConfig(
        "ptx_h2_scalar_one_tile_fma_unroll4_a128",
        "ptx",
        128,
        4,
        8,
        1,
        math_op="fma",
        unroll=4,
        schedule="one_tile",
        assumed_align=128,
    ),
    VectorAddConfig(
        "ptx_h2_scalar_one_tile_fma_unroll4_loads_first",
        "ptx",
        128,
        4,
        8,
        1,
        math_op="fma",
        unroll=4,
        schedule="one_tile",
        assumed_align=128,
        op_order="loads_first",
    ),
    VectorAddConfig(
        "ptx_h2_scalar_one_tile_fma_unroll4_bundle",
        "ptx",
        128,
        4,
        8,
        1,
        store_policy="wt",
        math_op="fma",
        unroll=4,
        schedule="one_tile",
        assumed_align=128,
        op_order="bundled_scalar",
    ),
    VectorAddConfig("ptx_h2_v4_fma_mix", "ptx", 256, 4, 8, 4, math_op="mix"),
    VectorAddConfig("ptx_h2_v4_fma_mix_unroll2", "ptx", 256, 4, 16, 4, math_op="mix", unroll=2),
    VectorAddConfig(
        "ptx_h2_v4_cs_t512_one_tile",
        "ptx",
        512,
        4,
        8,
        4,
        cache_policy="cs",
        schedule="one_tile",
    ),
    VectorAddConfig(
        "ptx_h2_v8_noalloc_one_tile",
        "ptx",
        128,
        4,
        32,
        8,
        cache_policy="noalloc",
        math_op="fma",
        unroll=2,
        schedule="one_tile",
    ),
    VectorAddConfig(
        "ptx_h2_v8_noalloc_tiles4",
        "ptx",
        128,
        4,
        32,
        8,
        cache_policy="noalloc",
        math_op="fma",
        unroll=2,
        schedule="fixed_tiles",
        tiles_per_cta=4,
    ),
    VectorAddConfig(
        "ptx_h2_v8_noalloc_persistent_x2_full",
        "ptx",
        128,
        2,
        32,
        8,
        cache_policy="noalloc",
        math_op="fma",
        unroll=2,
        schedule="persistent",
        assume_full_tiles=True,
    ),
    VectorAddConfig(
        "ptx_h2_v8_noalloc_persistent_x3_full",
        "ptx",
        128,
        3,
        32,
        8,
        cache_policy="noalloc",
        math_op="fma",
        unroll=2,
        schedule="persistent",
        assume_full_tiles=True,
    ),
    VectorAddConfig(
        "ptx_h2_v8_noalloc_persistent_x4_full",
        "ptx",
        128,
        4,
        32,
        8,
        cache_policy="noalloc",
        math_op="fma",
        unroll=2,
        schedule="persistent",
        assume_full_tiles=True,
    ),
)

CONFIG_BY_NAME = {cfg.name: cfg for cfg in CONFIGS}

# Measured on NVIDIA GB200 with CUDA graph replay.  ``bench.py --emit-dispatch``
# prints a replacement table for a different driver/toolchain.
DEFAULT_DISPATCH = {
    1024: "ptx_h2_v4_fma_mix",
    2048: "dsl_vectorized_t512",
    4096: "ptx_h2_v8_noalloc_one_tile",
    8192: "ptx_h2_v4_cs_t512_one_tile",
    16384: "ptx_h2_scalar_one_tile_fma_unroll4",
}


JIT_AUTOTUNE_CANDIDATES = {
    1024: (
        "ptx_h2_v4_fma_mix",
        "ptx_h2_v4_cs_t512_one_tile",
        "dsl_vectorized_t512",
        "ptx_h2_v8_noalloc_one_tile",
        "ptx_h2_scalar_one_tile_fma_unroll4",
    ),
    2048: (
        "dsl_vectorized_t512",
        "ptx_h2_v4_fma_mix",
        "ptx_h2_v4_fma_mix_unroll2",
        "ptx_h2_v8_noalloc_tiles4",
    ),
    4096: (
        "ptx_h2_v8_noalloc_one_tile",
        "ptx_h2_v8_noalloc_tiles4",
        "dsl_vectorized_t512",
        "ptx_h2_v4_fma_mix",
    ),
    8192: (
        "ptx_h2_v4_cs_t512_one_tile",
        "ptx_h2_scalar_one_tile_fma_unroll4",
        "ptx_h2_v8_noalloc_one_tile",
        "ptx_h2_v8_noalloc_tiles4",
    ),
    16384: (
        "ptx_h2_scalar_one_tile_fma_unroll4",
        "ptx_h2_scalar_one_tile_fma_unroll4_a128",
        "ptx_h2_scalar_one_tile_fma_unroll4_loads_first",
        "ptx_h2_scalar_one_tile_fma_unroll4_bundle",
        "ptx_h2_v4_cs_t512_one_tile",
    ),
}


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def add_f16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fma_one_f16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 one; mov.b32 one, 0x3c003c00; fma.rn.f16x2 $0, $1, one, $2; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _half2_add(a: Int32, b: Int32, math_op: cutlass.Constexpr[str], lane: cutlass.Constexpr[int]) -> Int32:
    if cutlass.const_expr(math_op == "fma"):
        return fma_one_f16x2(a, b)
    elif cutlass.const_expr(math_op == "mix" and lane % 2 == 1):
        return fma_one_f16x2(a, b)
    else:
        return add_f16x2(a, b)


@dsl_user_op
def ld_global_u32(ptr: cute.Pointer, cache_policy: str, *, loc=None, ip=None) -> Int32:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if cache_policy == "cs":
        asm = "ld.global.cs.u32 $0, [$1];"
    elif cache_policy == "noalloc":
        asm = "ld.global.L1::no_allocate.u32 $0, [$1];"
    elif cache_policy == "ca":
        asm = "ld.global.ca.u32 $0, [$1];"
    elif cache_policy == "cg":
        asm = "ld.global.cg.u32 $0, [$1];"
    else:
        asm = "ld.global.u32 $0, [$1];"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr_i64],
            asm,
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def st_global_u32(ptr: cute.Pointer, val: Int32, store_policy: str, *, loc=None, ip=None) -> None:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if store_policy == "cs":
        asm = "st.global.cs.u32 [$0], $1;"
    elif store_policy == "noalloc":
        asm = "st.global.L1::no_allocate.u32 [$0], $1;"
    elif store_policy == "wt":
        asm = "st.global.wt.u32 [$0], $1;"
    elif store_policy == "wb":
        asm = "st.global.wb.u32 [$0], $1;"
    elif store_policy == "cg":
        asm = "st.global.cg.u32 [$0], $1;"
    else:
        asm = "st.global.u32 [$0], $1;"
    llvm.inline_asm(
        None,
        [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        asm,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_v4_u32(ptr: cute.Pointer, cache_policy: str, *, loc=None, ip=None):
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if cache_policy == "cs":
        asm = "ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];"
    elif cache_policy == "noalloc":
        asm = "ld.global.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];"
    elif cache_policy == "ca":
        asm = "ld.global.ca.v4.u32 {$0, $1, $2, $3}, [$4];"
    elif cache_policy == "cg":
        asm = "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];"
    else:
        asm = "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [ptr_i64],
        asm,
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_global_v4_u32(ptr: cute.Pointer, vals, store_policy: str, *, loc=None, ip=None) -> None:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if store_policy == "cs":
        asm = "st.global.cs.v4.u32 [$0], {$1, $2, $3, $4};"
    elif store_policy == "noalloc":
        asm = "st.global.L1::no_allocate.v4.u32 [$0], {$1, $2, $3, $4};"
    elif store_policy == "wt":
        asm = "st.global.wt.v4.u32 [$0], {$1, $2, $3, $4};"
    elif store_policy == "wb":
        asm = "st.global.wb.v4.u32 [$0], {$1, $2, $3, $4};"
    elif store_policy == "cg":
        asm = "st.global.cg.v4.u32 [$0], {$1, $2, $3, $4};"
    else:
        asm = "st.global.v4.u32 [$0], {$1, $2, $3, $4};"
    llvm.inline_asm(
        None,
        [
            ptr_i64,
            Int32(vals[0]).ir_value(loc=loc, ip=ip),
            Int32(vals[1]).ir_value(loc=loc, ip=ip),
            Int32(vals[2]).ir_value(loc=loc, ip=ip),
            Int32(vals[3]).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_v8_u32(ptr: cute.Pointer, cache_policy: str, *, loc=None, ip=None):
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if cache_policy == "noalloc":
        asm = "ld.global.L1::no_allocate.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];"
    elif cache_policy == "cs":
        asm = "ld.global.cs.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];"
    elif cache_policy == "ca":
        asm = "ld.global.ca.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];"
    elif cache_policy == "cg":
        asm = "ld.global.cg.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];"
    else:
        asm = "ld.global.v8.u32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [ptr_i64],
        asm,
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [4], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [5], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [6], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), out, [7], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_global_v8_u32(ptr: cute.Pointer, vals, store_policy: str, *, loc=None, ip=None) -> None:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    if store_policy == "noalloc":
        asm = "st.global.L1::no_allocate.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    elif store_policy == "cs":
        asm = "st.global.cs.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    elif store_policy == "wt":
        asm = "st.global.wt.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    elif store_policy == "wb":
        asm = "st.global.wb.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    elif store_policy == "cg":
        asm = "st.global.cg.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    else:
        asm = "st.global.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
    llvm.inline_asm(
        None,
        [
            ptr_i64,
            Int32(vals[0]).ir_value(loc=loc, ip=ip),
            Int32(vals[1]).ir_value(loc=loc, ip=ip),
            Int32(vals[2]).ir_value(loc=loc, ip=ip),
            Int32(vals[3]).ir_value(loc=loc, ip=ip),
            Int32(vals[4]).ir_value(loc=loc, ip=ip),
            Int32(vals[5]).ir_value(loc=loc, ip=ip),
            Int32(vals[6]).ir_value(loc=loc, ip=ip),
            Int32(vals[7]).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "l,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scalar4_bundled_u32(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    cache_policy: str,
    store_policy: str,
    math_op: str,
    *,
    loc=None,
    ip=None,
) -> None:
    a_i64 = a_ptr.toint(loc=loc, ip=ip).ir_value()
    b_i64 = b_ptr.toint(loc=loc, ip=ip).ir_value()
    c_i64 = c_ptr.toint(loc=loc, ip=ip).ir_value()

    if cache_policy == "cs":
        ld = "ld.global.cs.u32"
    elif cache_policy == "noalloc":
        ld = "ld.global.L1::no_allocate.u32"
    elif cache_policy == "ca":
        ld = "ld.global.ca.u32"
    elif cache_policy == "cg":
        ld = "ld.global.cg.u32"
    else:
        ld = "ld.global.u32"

    if store_policy == "cs":
        st = "st.global.cs.u32"
    elif store_policy == "noalloc":
        st = "st.global.L1::no_allocate.u32"
    elif store_policy == "wt":
        st = "st.global.wt.u32"
    elif store_policy == "wb":
        st = "st.global.wb.u32"
    elif store_policy == "cg":
        st = "st.global.cg.u32"
    else:
        st = "st.global.u32"

    def math_line(dst: str, lhs: str, rhs: str, lane: int) -> str:
        if math_op == "fma" or (math_op == "mix" and lane % 2 == 1):
            return f"fma.rn.f16x2 {dst}, {lhs}, one, {rhs};"
        return f"add.rn.f16x2 {dst}, {lhs}, {rhs};"

    asm = (
        "{\n"
        "  .reg .b32 a0, a1, a2, a3;\n"
        "  .reg .b32 b0, b1, b2, b3;\n"
        "  .reg .b32 c0, c1, c2, c3;\n"
        "  .reg .b32 one;\n"
        "  mov.b32 one, 0x3c003c00;\n"
        f"  {ld} a0, [$0+0];\n"
        f"  {ld} b0, [$1+0];\n"
        f"  {ld} a1, [$0+4];\n"
        f"  {ld} b1, [$1+4];\n"
        f"  {ld} a2, [$0+8];\n"
        f"  {ld} b2, [$1+8];\n"
        f"  {ld} a3, [$0+12];\n"
        f"  {ld} b3, [$1+12];\n"
        f"  {math_line('c0', 'a0', 'b0', 0)}\n"
        f"  {math_line('c1', 'a1', 'b1', 1)}\n"
        f"  {math_line('c2', 'a2', 'b2', 2)}\n"
        f"  {math_line('c3', 'a3', 'b3', 3)}\n"
        f"  {st} [$2+0], c0;\n"
        f"  {st} [$2+4], c1;\n"
        f"  {st} [$2+8], c2;\n"
        f"  {st} [$2+12], c3;\n"
        "}"
    )
    llvm.inline_asm(
        None,
        [a_i64, b_i64, c_i64],
        asm,
        "l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _ptx_add_fragment(
    a32: cute.Tensor,
    b32: cute.Tensor,
    c32: cute.Tensor,
    vector_words: cutlass.Constexpr[int],
    cache_policy: cutlass.Constexpr[str],
    store_policy: cutlass.Constexpr[str],
    math_op: cutlass.Constexpr[str],
    unroll: cutlass.Constexpr[int],
    op_order: cutlass.Constexpr[str],
) -> None:
    if cutlass.const_expr(vector_words == 8):
        if cutlass.const_expr(unroll == 2):
            for i in cutlass.range_constexpr(0, cute.size(a32), 16):
                av0 = ld_global_v8_u32(elem_pointer(a32, i), cache_policy)
                bv0 = ld_global_v8_u32(elem_pointer(b32, i), cache_policy)
                cv0 = (
                    _half2_add(av0[0], bv0[0], math_op, 0),
                    _half2_add(av0[1], bv0[1], math_op, 1),
                    _half2_add(av0[2], bv0[2], math_op, 2),
                    _half2_add(av0[3], bv0[3], math_op, 3),
                    _half2_add(av0[4], bv0[4], math_op, 4),
                    _half2_add(av0[5], bv0[5], math_op, 5),
                    _half2_add(av0[6], bv0[6], math_op, 6),
                    _half2_add(av0[7], bv0[7], math_op, 7),
                )
                av1 = ld_global_v8_u32(elem_pointer(a32, i + 8), cache_policy)
                bv1 = ld_global_v8_u32(elem_pointer(b32, i + 8), cache_policy)
                cv1 = (
                    _half2_add(av1[0], bv1[0], math_op, 8),
                    _half2_add(av1[1], bv1[1], math_op, 9),
                    _half2_add(av1[2], bv1[2], math_op, 10),
                    _half2_add(av1[3], bv1[3], math_op, 11),
                    _half2_add(av1[4], bv1[4], math_op, 12),
                    _half2_add(av1[5], bv1[5], math_op, 13),
                    _half2_add(av1[6], bv1[6], math_op, 14),
                    _half2_add(av1[7], bv1[7], math_op, 15),
                )
                st_global_v8_u32(elem_pointer(c32, i), cv0, store_policy)
                st_global_v8_u32(elem_pointer(c32, i + 8), cv1, store_policy)
        else:
            for i in cutlass.range_constexpr(0, cute.size(a32), 8):
                av = ld_global_v8_u32(elem_pointer(a32, i), cache_policy)
                bv = ld_global_v8_u32(elem_pointer(b32, i), cache_policy)
                cv = (
                    _half2_add(av[0], bv[0], math_op, 0),
                    _half2_add(av[1], bv[1], math_op, 1),
                    _half2_add(av[2], bv[2], math_op, 2),
                    _half2_add(av[3], bv[3], math_op, 3),
                    _half2_add(av[4], bv[4], math_op, 4),
                    _half2_add(av[5], bv[5], math_op, 5),
                    _half2_add(av[6], bv[6], math_op, 6),
                    _half2_add(av[7], bv[7], math_op, 7),
                )
                st_global_v8_u32(elem_pointer(c32, i), cv, store_policy)
    elif cutlass.const_expr(vector_words == 4):
        if cutlass.const_expr(unroll == 2):
            for i in cutlass.range_constexpr(0, cute.size(a32), 8):
                av0 = ld_global_v4_u32(elem_pointer(a32, i), cache_policy)
                bv0 = ld_global_v4_u32(elem_pointer(b32, i), cache_policy)
                cv0 = (
                    _half2_add(av0[0], bv0[0], math_op, 0),
                    _half2_add(av0[1], bv0[1], math_op, 1),
                    _half2_add(av0[2], bv0[2], math_op, 2),
                    _half2_add(av0[3], bv0[3], math_op, 3),
                )
                av1 = ld_global_v4_u32(elem_pointer(a32, i + 4), cache_policy)
                bv1 = ld_global_v4_u32(elem_pointer(b32, i + 4), cache_policy)
                cv1 = (
                    _half2_add(av1[0], bv1[0], math_op, 4),
                    _half2_add(av1[1], bv1[1], math_op, 5),
                    _half2_add(av1[2], bv1[2], math_op, 6),
                    _half2_add(av1[3], bv1[3], math_op, 7),
                )
                st_global_v4_u32(elem_pointer(c32, i), cv0, store_policy)
                st_global_v4_u32(elem_pointer(c32, i + 4), cv1, store_policy)
        else:
            for i in cutlass.range_constexpr(0, cute.size(a32), 4):
                av = ld_global_v4_u32(elem_pointer(a32, i), cache_policy)
                bv = ld_global_v4_u32(elem_pointer(b32, i), cache_policy)
                cv = (
                    _half2_add(av[0], bv[0], math_op, 0),
                    _half2_add(av[1], bv[1], math_op, 1),
                    _half2_add(av[2], bv[2], math_op, 2),
                    _half2_add(av[3], bv[3], math_op, 3),
                )
                st_global_v4_u32(elem_pointer(c32, i), cv, store_policy)
    else:
        if cutlass.const_expr(unroll == 4):
            for i in cutlass.range_constexpr(0, cute.size(a32), 4):
                if cutlass.const_expr(op_order == "bundled_scalar"):
                    scalar4_bundled_u32(
                        elem_pointer(a32, i),
                        elem_pointer(b32, i),
                        elem_pointer(c32, i),
                        cache_policy,
                        store_policy,
                        math_op,
                    )
                elif cutlass.const_expr(op_order == "loads_first"):
                    av0 = ld_global_u32(elem_pointer(a32, i + 0), cache_policy)
                    bv0 = ld_global_u32(elem_pointer(b32, i + 0), cache_policy)
                    av1 = ld_global_u32(elem_pointer(a32, i + 1), cache_policy)
                    bv1 = ld_global_u32(elem_pointer(b32, i + 1), cache_policy)
                    av2 = ld_global_u32(elem_pointer(a32, i + 2), cache_policy)
                    bv2 = ld_global_u32(elem_pointer(b32, i + 2), cache_policy)
                    av3 = ld_global_u32(elem_pointer(a32, i + 3), cache_policy)
                    bv3 = ld_global_u32(elem_pointer(b32, i + 3), cache_policy)
                    cv0 = _half2_add(av0, bv0, math_op, 0)
                    cv1 = _half2_add(av1, bv1, math_op, 1)
                    cv2 = _half2_add(av2, bv2, math_op, 2)
                    cv3 = _half2_add(av3, bv3, math_op, 3)
                    st_global_u32(elem_pointer(c32, i + 0), cv0, store_policy)
                    st_global_u32(elem_pointer(c32, i + 1), cv1, store_policy)
                    st_global_u32(elem_pointer(c32, i + 2), cv2, store_policy)
                    st_global_u32(elem_pointer(c32, i + 3), cv3, store_policy)
                else:
                    av0 = ld_global_u32(elem_pointer(a32, i + 0), cache_policy)
                    bv0 = ld_global_u32(elem_pointer(b32, i + 0), cache_policy)
                    cv0 = _half2_add(av0, bv0, math_op, 0)
                    av1 = ld_global_u32(elem_pointer(a32, i + 1), cache_policy)
                    bv1 = ld_global_u32(elem_pointer(b32, i + 1), cache_policy)
                    cv1 = _half2_add(av1, bv1, math_op, 1)
                    av2 = ld_global_u32(elem_pointer(a32, i + 2), cache_policy)
                    bv2 = ld_global_u32(elem_pointer(b32, i + 2), cache_policy)
                    cv2 = _half2_add(av2, bv2, math_op, 2)
                    av3 = ld_global_u32(elem_pointer(a32, i + 3), cache_policy)
                    bv3 = ld_global_u32(elem_pointer(b32, i + 3), cache_policy)
                    cv3 = _half2_add(av3, bv3, math_op, 3)
                    st_global_u32(elem_pointer(c32, i + 0), cv0, store_policy)
                    st_global_u32(elem_pointer(c32, i + 1), cv1, store_policy)
                    st_global_u32(elem_pointer(c32, i + 2), cv2, store_policy)
                    st_global_u32(elem_pointer(c32, i + 3), cv3, store_policy)
        else:
            for i in cutlass.range_constexpr(cute.size(a32)):
                av = ld_global_u32(elem_pointer(a32, i), cache_policy)
                bv = ld_global_u32(elem_pointer(b32, i), cache_policy)
                st_global_u32(
                    elem_pointer(c32, i), _half2_add(av, bv, math_op, 0), store_policy
                )


class VectorAdd(CuteDSLKernel):
    def __init__(
        self,
        config: VectorAddConfig,
        num_ctas: int,
        tile_count: int,
        assume_full_tiles: int,
    ):
        assert config.elems_per_thread % 2 == 0
        assert config.vector_words in (1, 4, 8)
        assert config.cache_policy in ("default", "cs", "noalloc", "ca", "cg")
        assert config.store_policy in ("default", "cs", "noalloc", "wt", "wb", "cg")
        assert (config.elems_per_thread // 2) % config.vector_words == 0
        assert config.unroll in (1, 2, 4)
        assert ((config.elems_per_thread // 2) // config.vector_words) % config.unroll == 0
        assert config.schedule in ("persistent", "one_tile", "fixed_tiles")
        assert config.tiles_per_cta in (1, 2, 4, 8)
        assert not config.assume_full_tiles or config.schedule == "persistent"
        assert config.assumed_align in (16, 32, 64, 128)
        assert config.op_order in ("interleaved", "loads_first", "bundled_scalar")
        assert config.op_order == "interleaved" or (
            config.vector_words == 1 and config.unroll == 4
        )
        self.config = config
        self.num_ctas = num_ctas
        self.tile_count = tile_count
        self.assume_full_tiles = assume_full_tiles

    @cute.jit
    def _process_tile(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout,
        tile: Int32,
        tidx: Int32,
    ):
        blkA = gA[(None, tile)]
        blkB = gB[(None, tile)]
        blkC = gC[(None, tile)]

        tidfrgA = cute.composition(blkA, tv_layout)
        tidfrgB = cute.composition(blkB, tv_layout)
        tidfrgC = cute.composition(blkC, tv_layout)

        thrA = tidfrgA[(tidx, None)]
        thrB = tidfrgB[(tidx, None)]
        thrC = tidfrgC[(tidx, None)]

        if cutlass.const_expr(self.config.variant == "dsl"):
            thrC[None] = thrA.load() + thrB.load()
        else:
            a32 = cute.recast_tensor(thrA, Int32)
            b32 = cute.recast_tensor(thrB, Int32)
            c32 = cute.recast_tensor(thrC, Int32)
            _ptx_add_fragment(
                a32,
                b32,
                c32,
                self.config.vector_words,
                self.config.cache_policy,
                self.config.store_policy,
                self.config.math_op,
                self.config.unroll,
                self.config.op_order,
            )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tiler = cute.make_layout((self.config.elems_per_cta,))
        gA = cute.zipped_divide(mA, tiler)
        gB = cute.zipped_divide(mB, tiler)
        gC = cute.zipped_divide(mC, tiler)
        tv_layout = cute.make_layout(
            (self.config.threads, self.config.elems_per_thread),
            stride=(self.config.elems_per_thread, 1),
        )
        self.kernel(gA, gB, gC, tv_layout).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.config.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.config.schedule == "one_tile"):
            self._process_tile(gA, gB, gC, tv_layout, bidx, tidx)
        elif cutlass.const_expr(self.config.schedule == "fixed_tiles"):
            base_tile = bidx * self.config.tiles_per_cta
            for i in cutlass.range_constexpr(self.config.tiles_per_cta):
                tile = base_tile + i
                if tile < self.tile_count:
                    self._process_tile(gA, gB, gC, tv_layout, tile, tidx)
        else:
            if cutlass.const_expr(self.assume_full_tiles == 1):
                self._process_tile(gA, gB, gC, tv_layout, bidx, tidx)
                tile = bidx + self.num_ctas
                while tile < self.tile_count:
                    self._process_tile(gA, gB, gC, tv_layout, tile, tidx)
                    tile += self.num_ctas
            else:
                tile = bidx
                while tile < self.tile_count:
                    self._process_tile(gA, gB, gC, tv_layout, tile, tidx)
                    tile += self.num_ctas


_compile_cache: dict = {}
_jit_autotune_cache: dict = {}


def _device_sm_count() -> int:
    if not torch.cuda.is_available():
        return B200_SM_COUNT
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _device_tuning_key() -> tuple:
    if not torch.cuda.is_available():
        return ("no_cuda", B200_SM_COUNT)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.name, props.multi_processor_count, torch.version.cuda)


def _tile_count(numel: int, cfg: VectorAddConfig) -> int:
    return (numel + cfg.elems_per_cta - 1) // cfg.elems_per_cta


def _num_ctas(tile_count: int, cfg: VectorAddConfig) -> int:
    if cfg.schedule == "one_tile":
        return tile_count
    if cfg.schedule == "fixed_tiles":
        return (tile_count + cfg.tiles_per_cta - 1) // cfg.tiles_per_cta
    return _device_sm_count() * cfg.cta_per_sm


def _validate_launch_config(tile_count: int, num_ctas: int, cfg: VectorAddConfig) -> None:
    if cfg.assume_full_tiles and tile_count < num_ctas:
        raise ValueError(
            "assume_full_tiles requires tile_count >= num_ctas; "
            f"got tile_count={tile_count}, num_ctas={num_ctas}, config={cfg.name}"
        )


def dispatch_config(size: int) -> VectorAddConfig:
    if size not in DEFAULT_DISPATCH:
        raise ValueError(f"unsupported size {size}; expected one of {B200_SHAPES}")
    return CONFIG_BY_NAME[DEFAULT_DISPATCH[size]]


def clear_jit_autotune_cache() -> None:
    _jit_autotune_cache.clear()


def jit_autotune_config(
    size: int,
    *,
    score_fn=None,
    device_key: tuple | None = None,
    force: bool = False,
) -> VectorAddConfig:
    if size not in JIT_AUTOTUNE_CANDIDATES:
        raise ValueError(f"unsupported size {size}; expected one of {B200_SHAPES}")
    if device_key is None:
        device_key = _device_tuning_key()
    key = (device_key, size)
    if not force and key in _jit_autotune_cache:
        return _jit_autotune_cache[key]
    if score_fn is None:
        raise ValueError("score_fn is required for a new JIT autotune decision")

    best: tuple[float, VectorAddConfig] | None = None
    for name in JIT_AUTOTUNE_CANDIDATES[size]:
        cfg = CONFIG_BY_NAME[name]
        try:
            us = float(score_fn(cfg))
        except ValueError:
            continue
        if best is None or us < best[0]:
            best = (us, cfg)
    if best is None:
        raise ValueError(f"no valid JIT autotune candidates for size {size}")

    _jit_autotune_cache[key] = best[1]
    return best[1]


def _cuda_graph_time_us(fn, warmup: int = 3, iters: int = 20, runs: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    medians = []
    for _ in range(runs):
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        medians.append(statistics.median(start.elapsed_time(end) * 1000 for start, end in events))
    return statistics.median(medians)


def _run_vector_add(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    cfg: VectorAddConfig,
) -> torch.Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    out_flat = out.reshape(-1)
    tile_count = _tile_count(a_flat.numel(), cfg)
    num_ctas = _num_ctas(tile_count, cfg)
    assume_full_tiles = 1 if cfg.assume_full_tiles else 0
    _validate_launch_config(tile_count, num_ctas, cfg)
    if (
        a_flat.data_ptr() % cfg.assumed_align
        or b_flat.data_ptr() % cfg.assumed_align
        or out_flat.data_ptr() % cfg.assumed_align
    ):
        raise ValueError(
            f"{cfg.name} assumes {cfg.assumed_align}-byte aligned tensors; "
            f"got a={a_flat.data_ptr()} b={b_flat.data_ptr()} out={out_flat.data_ptr()}"
        )

    a_ = from_dlpack(a_flat, assumed_align=cfg.assumed_align)
    b_ = from_dlpack(b_flat, assumed_align=cfg.assumed_align)
    c_ = from_dlpack(out_flat, assumed_align=cfg.assumed_align)
    stream = cutlass_torch.current_stream()

    key = (a.shape[0], cfg, num_ctas, tile_count, assume_full_tiles)
    if key not in _compile_cache:
        op = VectorAdd(cfg, num_ctas, tile_count, assume_full_tiles)
        _compile_cache[key] = cute.compile(op, a_, b_, c_, stream)
    _compile_cache[key](a_, b_, c_, stream)

    return out


def _jit_autotune_config_for_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> VectorAddConfig:
    size = a.shape[0]

    def score(cfg: VectorAddConfig) -> float:
        return _cuda_graph_time_us(lambda cfg=cfg: _run_vector_add(a, b, out, cfg))

    return jit_autotune_config(size, score_fn=score)


def vector_add_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    config_name: str | None = None,
    config: VectorAddConfig | None = None,
    autotune: bool = False,
) -> torch.Tensor:
    assert a.shape == b.shape
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    assert a.shape[0] in B200_SHAPES
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()

    if out is None:
        out = torch.empty_like(a)
    else:
        assert out.shape == a.shape
        assert out.dtype == torch.float16
        assert out.is_cuda and out.is_contiguous()

    if config is not None:
        cfg = config
    elif autotune or config_name == "jit_autotune":
        cfg = _jit_autotune_config_for_tensors(a, b, out)
    elif config_name is not None:
        cfg = CONFIG_BY_NAME[config_name]
    else:
        cfg = dispatch_config(a.shape[0])

    return _run_vector_add(a, b, out, cfg)


__all__ = [
    "B200_SHAPES",
    "CONFIGS",
    "CONFIG_BY_NAME",
    "DEFAULT_DISPATCH",
    "JIT_AUTOTUNE_CANDIDATES",
    "VectorAddConfig",
    "VectorAdd",
    "clear_jit_autotune_cache",
    "dispatch_config",
    "jit_autotune_config",
    "vector_add_interface",
]
