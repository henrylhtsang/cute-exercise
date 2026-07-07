from cute_exercise.ex30_gb200_tpc_groupings.analysis import (
    ClusterObservation,
    cluster_tpc_signatures,
    infer_topology,
)


def test_infer_topology_from_stable_cluster_rows():
    observations = [
        ClusterObservation(cluster_size=2, iteration=0, cluster_id=0, smids=(4, 9)),
        ClusterObservation(cluster_size=2, iteration=1, cluster_id=0, smids=(9, 4)),
        ClusterObservation(cluster_size=2, iteration=0, cluster_id=1, smids=(1, 7)),
        ClusterObservation(cluster_size=2, iteration=1, cluster_id=1, smids=(7, 1)),
        ClusterObservation(
            cluster_size=4, iteration=0, cluster_id=0, smids=(4, 9, 1, 7)
        ),
        ClusterObservation(
            cluster_size=4, iteration=1, cluster_id=0, smids=(7, 1, 9, 4)
        ),
    ]

    topology = infer_topology(observations)

    assert topology.tpc_groups == ((1, 7), (4, 9))
    assert topology.cluster_groups[2] == ((1, 7), (4, 9))
    assert topology.cluster_groups[4] == ((1, 4, 7, 9),)
    assert topology.cluster_group_observation_counts[2] == {
        (1, 7): 2,
        (4, 9): 2,
    }
    assert topology.cluster_group_observation_counts[4] == {
        (1, 4, 7, 9): 2,
    }
    assert topology.smid_to_tpc == {1: 0, 7: 0, 4: 1, 9: 1}


def test_infer_topology_marks_conflicting_cluster2_pairs_ambiguous():
    observations = [
        ClusterObservation(cluster_size=2, iteration=0, cluster_id=0, smids=(0, 1)),
        ClusterObservation(cluster_size=2, iteration=1, cluster_id=0, smids=(0, 2)),
        ClusterObservation(cluster_size=2, iteration=2, cluster_id=0, smids=(0, 1)),
    ]

    topology = infer_topology(observations)

    assert topology.tpc_groups == ((0, 1),)
    assert topology.ambiguous_groups[2] == ((0, 2),)
    assert "conflicting cluster_size=2 groups were observed" in topology.notes


def test_cluster_observation_requires_matching_size():
    try:
        ClusterObservation(cluster_size=4, iteration=0, cluster_id=0, smids=(0, 1))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("ClusterObservation accepted a malformed group")


def test_cluster_tpc_signatures_groups_similar_latency_vectors():
    # Six TPCs, two SMs per TPC. The synthetic signatures form physical groups
    # of sizes 3, 2, and 1.
    smid_to_tpc = {smid: smid // 2 for smid in range(12)}
    templates = {
        0: (1.0, 1.1, 0.9, 1.0),
        1: (1.1, 1.0, 1.0, 0.9),
        2: (0.9, 1.2, 1.0, 1.0),
        3: (9.0, 9.1, 8.9, 9.0),
        4: (8.9, 9.0, 9.2, 9.1),
        5: (20.0, 19.8, 20.2, 20.1),
    }
    signatures = {}
    for smid, tpc in smid_to_tpc.items():
        epsilon = 0.01 if smid % 2 else 0.0
        signatures[smid] = tuple(value + epsilon for value in templates[tpc])

    result = cluster_tpc_signatures(signatures, smid_to_tpc, target_groups=3)

    assert result.group_counts == (3, 2, 1)
    assert result.groups == ((0, 1, 2), (3, 4), (5,))
    assert len(result.merge_distances) == 5
