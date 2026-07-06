import inspect

import pytest
import torch

from cute_exercise.ex29_best_vector_add.kernel import (
    B200_SHAPES,
    CONFIGS,
    CONFIG_BY_NAME,
    DEFAULT_DISPATCH,
    JIT_AUTOTUNE_CANDIDATES,
    VectorAdd,
    VectorAddConfig,
    clear_jit_autotune_cache,
    dispatch_config,
    jit_autotune_config,
    _num_ctas,
    _run_vector_add,
    _tile_count,
    _validate_launch_config,
    vector_add_interface,
)
from cute_exercise.ex29_best_vector_add.autotune import (
    AutotuneMeasurement,
    analyze_top_fraction,
    generate_autotune_configs,
)


def test_dispatch_covers_requested_b200_shapes():
    assert B200_SHAPES == (1024, 2048, 4096, 8192, 16384)
    assert len(CONFIGS) >= 8

    names = {cfg.name for cfg in CONFIGS}
    assert "ptx_h2_v4_cs_t512_one_tile" in names
    assert "ptx_h2_scalar_one_tile_fma_unroll4" in names
    assert "ptx_h2_v8_noalloc_one_tile" in names
    assert "ptx_h2_v8_noalloc_tiles4" in names
    assert "ptx_h2_v8_noalloc_persistent_x2_full" in names
    assert "ptx_h2_v8_noalloc_persistent_x3_full" in names
    assert "ptx_h2_v8_noalloc_persistent_x4_full" in names

    for size in B200_SHAPES:
        cfg = dispatch_config(size)
        assert cfg in CONFIGS
        assert cfg.threads in (128, 256, 512)
        assert cfg.cta_per_sm in (1, 2, 4, 8)
        assert cfg.elems_per_thread in (8, 16, 32)
        assert cfg.vector_words in (1, 4, 8)
        assert cfg.unroll in (1, 2, 4)
        assert cfg.schedule in ("persistent", "one_tile", "fixed_tiles")
        assert cfg.tiles_per_cta in (1, 2, 4, 8)


def test_jit_autotune_candidates_are_small_and_cover_every_shape():
    assert set(JIT_AUTOTUNE_CANDIDATES) == set(B200_SHAPES)
    for size, names in JIT_AUTOTUNE_CANDIDATES.items():
        assert 2 <= len(names) <= 6
        assert DEFAULT_DISPATCH[size] in names
        for name in names:
            assert name in CONFIG_BY_NAME


def test_jit_autotune_selects_and_caches_fastest_candidate():
    clear_jit_autotune_cache()
    calls = []
    scores = {
        "ptx_h2_v8_noalloc_one_tile": 3.0,
        "ptx_h2_v8_noalloc_tiles4": 1.0,
        "dsl_vectorized_t512": 2.0,
    }

    def score(cfg):
        calls.append(cfg.name)
        return scores.get(cfg.name, 10.0)

    cfg = jit_autotune_config(4096, score_fn=score, device_key=("B200", 148))
    assert cfg.name == "ptx_h2_v8_noalloc_tiles4"
    assert set(calls) == set(JIT_AUTOTUNE_CANDIDATES[4096])

    calls.clear()
    cached = jit_autotune_config(4096, score_fn=score, device_key=("B200", 148))
    assert cached.name == "ptx_h2_v8_noalloc_tiles4"
    assert calls == []

    cfg_other_device = jit_autotune_config(4096, score_fn=score, device_key=("GB200", 152))
    assert cfg_other_device.name == "ptx_h2_v8_noalloc_tiles4"
    assert set(calls) == set(JIT_AUTOTUNE_CANDIDATES[4096])


def test_tile_count_is_host_computed_from_compile_time_config():
    cfg = CONFIG_BY_NAME["ptx_h2_v8_noalloc_persistent_x4_full"]
    assert _tile_count(1024 * 1024, cfg) == 256
    assert _tile_count(16384 * 16384, cfg) == 65536


def test_persistent_stride_uses_compile_time_num_ctas_not_grid_dim_read():
    init_source = inspect.getsource(VectorAdd.__init__)
    source = inspect.getsource(VectorAdd.kernel)
    assert "num_ctas: int" in init_source
    assert "block_dim" not in source
    assert "grid_dim" not in source
    assert "tile += self.num_ctas" in source


def test_top_fraction_analysis_keeps_options_that_survive_every_size():
    fast_a = CONFIG_BY_NAME["ptx_h2_v8_noalloc_one_tile"]
    fast_b = CONFIG_BY_NAME["ptx_h2_v8_noalloc_persistent_x3_full"]
    slow = CONFIG_BY_NAME["ptx_h2_scalar_one_tile_fma_unroll4"]
    measurements = {
        8192: [
            AutotuneMeasurement(60.0, fast_a),
            AutotuneMeasurement(61.0, fast_b),
            AutotuneMeasurement(90.0, slow),
        ],
        16384: [
            AutotuneMeasurement(220.0, fast_b),
            AutotuneMeasurement(221.0, fast_a),
            AutotuneMeasurement(330.0, slow),
        ],
    }

    analysis = analyze_top_fraction(measurements, top_fraction=0.50)

    assert analysis["variant"].keep == {"ptx"}
    assert analysis["threads"].keep == {128}
    assert analysis["schedule"].keep == {"one_tile", "persistent"}
    assert analysis["vector_words"].drop == {1}


def test_assume_full_tiles_fast_path_is_explicit_and_guarded():
    cfg = VectorAddConfig(
        "test_full_tiles",
        "ptx",
        128,
        4,
        32,
        8,
        cache_policy="noalloc",
        assume_full_tiles=True,
    )
    init_source = inspect.getsource(VectorAdd.__init__)
    call_source = inspect.getsource(VectorAdd.__call__)
    kernel_source = inspect.getsource(VectorAdd.kernel)
    launch_source = inspect.getsource(_run_vector_add)
    assert "assume_full_tiles: int" in init_source
    assert "tile_count: int" in init_source
    assert "num_ctas: int" in init_source
    assert "assume_full_tiles:" not in call_source
    assert "num_ctas:" not in call_source
    assert "self.config.assume_full_tiles" not in kernel_source
    assert "cutlass.const_expr(self.assume_full_tiles == 1)" in kernel_source
    assert "tile = bidx + self.num_ctas" in kernel_source
    assert "cfg.assume_full_tiles" in launch_source

    with pytest.raises(ValueError, match="assume_full_tiles"):
        _validate_launch_config(tile_count=1, num_ctas=2, cfg=cfg)

    _validate_launch_config(tile_count=2, num_ctas=2, cfg=cfg)


def test_autotune_grid_covers_scheduling_strategies_without_duplicates():
    configs = generate_autotune_configs()
    keys = {
        (
            cfg.variant,
            cfg.threads,
            cfg.cta_per_sm,
            cfg.elems_per_thread,
            cfg.vector_words,
            cfg.cache_policy,
            getattr(cfg, "store_policy", None),
            getattr(cfg, "assumed_align", None),
            getattr(cfg, "op_order", None),
            cfg.math_op,
            cfg.unroll,
            cfg.schedule,
            cfg.tiles_per_cta,
            cfg.assume_full_tiles,
        )
        for cfg in configs
    }

    assert len(configs) == len(keys)
    assert {cfg.schedule for cfg in configs} == {"persistent", "one_tile", "fixed_tiles"}
    assert any(cfg.schedule == "persistent" and cfg.cta_per_sm == 6 for cfg in configs)
    assert any(
        cfg.schedule == "persistent" and cfg.cta_per_sm in (2, 3, 4) and cfg.assume_full_tiles
        for cfg in configs
    )
    assert any(cfg.schedule == "one_tile" and cfg.threads == 128 for cfg in configs)
    assert any(cfg.schedule == "fixed_tiles" and cfg.tiles_per_cta == 8 for cfg in configs)
    assert any(cfg.variant == "dsl" for cfg in configs)
    assert any(cfg.vector_words == 8 and cfg.unroll == 2 for cfg in configs)


def test_autotune_grid_covers_scalar_ordering_alignment_and_store_policy():
    configs = generate_autotune_configs()

    assert any(
        cfg.variant == "ptx"
        and cfg.vector_words == 1
        and cfg.unroll == 4
        and cfg.op_order == "loads_first"
        for cfg in configs
    )
    assert any(
        cfg.variant == "ptx"
        and cfg.vector_words == 1
        and cfg.unroll == 4
        and cfg.op_order == "bundled_scalar"
        for cfg in configs
    )
    assert any(cfg.cache_policy != cfg.store_policy for cfg in configs)
    assert {16, 32, 64, 128}.issubset({cfg.assumed_align for cfg in configs})


def test_configured_alignment_is_used_for_dlpack_and_cache_key():
    source = inspect.getsource(_run_vector_add)

    assert "cfg.assumed_align" in source
    assert "assumed_align=cfg.assumed_align" in source
    assert "a_flat.data_ptr() % cfg.assumed_align" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [1024, 2048])
@pytest.mark.parametrize(
    "config_name",
    [
        "dsl_vectorized_t512",
        "ptx_h2_scalar_one_tile_fma_unroll4",
        "ptx_h2_v4_fma_mix",
        "ptx_h2_v4_cs_t512_one_tile",
    ],
)
def test_vector_add_matches_torch_for_small_b200_shapes(size, config_name):
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    out = vector_add_interface(a, b, config_name=config_name)

    torch.testing.assert_close(out, a + b, rtol=0, atol=0)
