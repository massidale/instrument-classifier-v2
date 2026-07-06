from scripts_ablation_shim import ABLATION_COMBOS, combo_name, format_ablation_table


def test_combos_grow_monotonically():
    actives = [sorted(k for k, on in c.items() if on) for c in ABLATION_COMBOS]
    assert actives[0] == ["mel"]
    assert actives[-1] == ["chroma", "cqt", "mel", "wave"]
    for smaller, larger in zip(actives, actives[1:]):
        assert set(smaller) < set(larger)


def test_format_ablation_table():
    rows = [
        {"name": "mel", "val_micro_f1": 0.61,
         "test": {"micro_f1": 0.55, "macro_f1": 0.48}},
        {"name": "mel+cqt", "val_micro_f1": 0.64, "test": None},
    ]
    table = format_ablation_table(rows)
    assert "| mel " in table and "| mel+cqt " in table
    assert "0.5500" in table and "n/a" in table
