"""S2 CLI wiring: arg validation, --dry-run, exit codes. Spec 3, 4, 6."""

from __future__ import annotations

import json

import pytest

from placement_sampler.cli import main

# A deliberately feasible sector for the wiring tests. The real provisional
# sector (-35..88 deg) does NOT fit 90 points at 2 cm -- that abort path is
# covered by test_infeasible_config_exits_2.
MANUAL = [
    "--r-inner", "20", "--r-outer", "30",
    "--theta-min", "-90", "--theta-max", "90",
    "--d-min", "2.0", "--seed", "20260831", "--label", "camp_A", "--date", "20260831",
]


def _run(args, tmp_path, extra=None):
    argv = [*args, "--out-dir", str(tmp_path)]
    if extra:
        argv += extra
    return main(argv)


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_missing_d_min_is_rejected(tmp_path):
    args = [a for a in MANUAL if a not in ("--d-min", "2.0")]
    with pytest.raises(SystemExit) as exc:
        _run(args, tmp_path)
    assert exc.value.code == 2


def test_from_summary_together_with_manual_bounds_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(MANUAL, tmp_path, extra=["--from-summary", "whatever.json"])
    assert exc.value.code == 2


def test_from_summary_without_margin_is_rejected(tmp_path):
    summary = tmp_path / "s.json"
    summary.write_text("{}", encoding="utf-8")
    args = [
        "--from-summary", str(summary),
        "--d-min", "2.0", "--seed", "1", "--label", "c", "--date", "20260831",
    ]
    with pytest.raises(SystemExit) as exc:
        _run(args, tmp_path)
    assert exc.value.code == 2


def test_dry_run_writes_nothing_and_reports_feasibility(tmp_path, capsys):
    rc = _run(MANUAL, tmp_path, extra=["--dry-run"])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []
    assert "N_hex" in capsys.readouterr().out


def test_infeasible_config_exits_2(tmp_path):
    args = [
        "--r-inner", "20", "--r-outer", "21", "--theta-min", "0", "--theta-max", "5",
        "--d-min", "5.0", "--seed", "1", "--label", "c", "--date", "20260831",
        "--n-train", "40", "--n-eval-open", "0", "--n-eval-close", "0",
    ]
    assert _run(args, tmp_path) == 2


def test_full_manual_run_creates_three_csvs_plus_meta(tmp_path):
    rc = _run(MANUAL, tmp_path)
    assert rc == 0
    got = sorted(p.name for p in tmp_path.iterdir())
    assert got == [
        "camp_A_20260831_eval-close.csv",
        "camp_A_20260831_eval-open.csv",
        "camp_A_20260831_meta.json",
        "camp_A_20260831_train.csv",
    ]
    meta = json.loads((tmp_path / "camp_A_20260831_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 20260831
    assert meta["d_min_cm"] == 2.0
    assert meta["sampling_source"] == "manual"
    # r^2 should be uniform over U(r_in^2, r_out^2) -- the anti-bug check (spec 6).
    assert meta["per_list"]["train"]["r2_uniform_ks_p"] > 0.01


def test_run_prints_the_r_squared_ks_pvalue(tmp_path, capsys):
    _run(MANUAL, tmp_path)
    assert "KS p" in capsys.readouterr().out


def test_second_run_without_force_refuses_and_exits_1(tmp_path):
    assert _run(MANUAL, tmp_path) == 0
    assert _run(MANUAL, tmp_path) == 1


def test_force_lets_the_second_run_succeed(tmp_path):
    assert _run(MANUAL, tmp_path) == 0
    assert _run(MANUAL, tmp_path, extra=["--force"]) == 0


def test_reruns_produce_byte_identical_csvs(tmp_path):
    _run(MANUAL, tmp_path)
    first = (tmp_path / "camp_A_20260831_train.csv").read_bytes()
    _run(MANUAL, tmp_path, extra=["--force"])
    assert (tmp_path / "camp_A_20260831_train.csv").read_bytes() == first
