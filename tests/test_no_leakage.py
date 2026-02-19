from pathlib import Path

from src.data.leakage_check import run_leakage_check
from src.data.split_indication import split_indication


def test_split_has_no_train_leakage(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = tmp_path / "splits"

    split_meta = split_indication(
        data_dir=data_dir,
        output_dir=output_dir,
        seed=0,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    assert split_meta["pair_counts"]["train"] > 0
    assert split_meta["pair_counts"]["val"] > 0
    assert split_meta["pair_counts"]["test"] > 0

    report = run_leakage_check(output_dir)

    assert split_meta["quadruple_counts"]["train"] == report["quadruples"]["train"]
    assert split_meta["quadruple_counts"]["val"] == report["quadruples"]["val"]
    assert split_meta["quadruple_counts"]["test"] == report["quadruples"]["test"]

    assert report["pairs"]["overlap_train_val"] == 0
    assert report["pairs"]["overlap_train_test"] == 0
    assert report["pairs"]["overlap_val_test"] == 0
    assert report["quadruples"]["pair_membership_missing_train"] == 0
    assert report["quadruples"]["pair_membership_missing_val"] == 0
    assert report["quadruples"]["pair_membership_missing_test"] == 0
    assert report["quadruples"]["pair_overlap_train_val"] == 0
    assert report["quadruples"]["pair_overlap_train_test"] == 0
    assert report["quadruples"]["full_overlap_train_val"] == 0
    assert report["quadruples"]["full_overlap_train_test"] == 0
    assert report["quadruples"]["full_overlap_val_test"] == 0
