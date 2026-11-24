from make_splits import make_splits

NAMES = [f"potholes{i}.png" for i in range(665)]


def test_splits_are_disjoint_and_cover_everything():
    splits = make_splits(NAMES, seed=0)
    all_names = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_names) == sorted(NAMES)
    assert len(set(all_names)) == len(NAMES)


def test_split_sizes_follow_70_10_20():
    splits = make_splits(NAMES, seed=0)
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (466, 66, 133)


def test_same_seed_gives_same_split_regardless_of_input_order():
    assert make_splits(NAMES, seed=0) == make_splits(list(reversed(NAMES)), seed=0)


def test_different_seed_gives_different_split():
    assert make_splits(NAMES, seed=0) != make_splits(NAMES, seed=1)
