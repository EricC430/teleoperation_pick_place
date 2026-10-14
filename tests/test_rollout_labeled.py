"""Checks for scripts/rollout_labeled.py that need no robot: keys, the episode_meta CSV, the reset-phase wait."""

import csv
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rollout_labeled", _REPO / "scripts" / "rollout_labeled.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)


def _rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_keys():
    e = rl.new_events()
    assert rl.apply_key("right", e) and e["verdict"] == "success" and e["exit_early"]
    assert rl.apply_key("up", e) and e["verdict"] == "fail"  # last press wins
    assert rl.apply_key("left", e) and e["rerecord_episode"]
    assert rl.apply_key("esc", e) and e["stop_recording"]
    assert not rl.apply_key("down", rl.new_events())


def test_sheet_prefill_verdict_prune(tmp_path):
    path = tmp_path / "rollout_x.csv"
    prefill = {"object_name": "paper_cup", "object_orientation": "stand", "env_light": "L", "env_bg": ""}
    sheet = rl.MetaSheet(path, first_index=0, count=36, prefill=prefill, placement_prefix="c")
    rows = _rows(path)
    assert [r["episode_index"] for r in rows] == [str(i) for i in range(36)]
    assert rows[0]["placement_id"] == "c1" and rows[35]["placement_id"] == "c36"
    assert rows[0]["object_name"] == "paper_cup" and rows[0]["env_bg"] == "" and rows[0]["outcome"] == ""

    sheet.set_verdict(0, "success")
    sheet.set_verdict(1, "fail")
    assert [r["outcome"] for r in _rows(path)[:2]] == ["success", ""]

    assert sheet.prune(saved_count=2) == list(range(2, 36))
    assert [r["episode_index"] for r in _rows(path)] == ["0", "1"]


def test_sheet_keeps_existing_rows(tmp_path):
    path = tmp_path / "rollout_x.csv"
    rl.MetaSheet(path, 0, 2, {"object_name": "a"}, "c").set_verdict(0, "success")
    rl.MetaSheet(path, 1, 2, {"object_name": "b"}, "c")  # resume from episode 1
    rows = _rows(path)
    assert [(r["object_name"], r["outcome"]) for r in rows] == [("a", "success"), ("a", ""), ("b", "")]


def test_meta_flags_are_stripped_from_lerobot_args():
    meta, rest = rl.parse_meta_args(
        ["--config_path", "x.yaml", "--object-name", "paper_cup", "--policy.path=r", "--env-light", "燈"]
    )
    assert meta.object_name == "paper_cup" and meta.env_light == "燈" and rl.default_placement_prefix(meta) == "c"
    assert rest == ["--config_path", "x.yaml", "--policy.path=r"]


def _template(tmp_path, n=3):
    path = tmp_path / "tpl.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("episode_index,object_name,object_orientation,placement_id,env_light,env_bg\n")
        for i in range(n):
            f.write(f"{i},alcan,{'label' if i % 2 else 'bare'},p{i},大燈,木質桌面\n")
    return path


def test_template_rows_copied_and_flags_override(tmp_path):
    meta, _ = rl.parse_meta_args(["--meta-template", str(_template(tmp_path)), "--env-bg", "黑布"])
    assert rl.default_placement_prefix(meta) is None  # template keeps its own placement_id
    out = tmp_path / "out.csv"
    rl.MetaSheet(out, 0, 2, {k: getattr(meta, k) for k in rl.PREFILL}, None, template=meta.template_data)
    rows = _rows(out)
    assert [(r["object_orientation"], r["placement_id"], r["env_light"], r["env_bg"]) for r in rows] == [
        ("bare", "p0", "大燈", "黑布"), ("label", "p1", "大燈", "黑布")]  # 3rd template row unused


def test_template_too_short_is_an_error(tmp_path):
    import pytest

    with pytest.raises(ValueError, match=r"\[3, 4\]"):
        rl.MetaSheet(tmp_path / "out.csv", 0, 5, {}, None, template=rl.load_template(str(_template(tmp_path))))


def test_template_name_resolves_to_templates_dir():
    assert rl.template_path("paper_cup_A1") == rl.TEMPLATE_DIR / "paper_cup_A1.csv"
    assert rl.load_template("paper_cup_A1")[0][35]["placement_id"] == "c36"


def _strategy():
    s = rl.make_strategy_class()(SimpleNamespace(reset_to_initial_position=True))
    ctx = SimpleNamespace(runtime=SimpleNamespace(shutdown_event=threading.Event()))
    return s, ctx


def _press_later(s, key, delay):
    threading.Timer(delay, rl.apply_key, args=(key, s._events)).start()


def test_verdict_in_policy_phase_saves_without_reset_wait():
    e = rl.new_events()
    assert not rl.saves_without_reset_wait(e)  # ran out its time: wait for a verdict
    rl.apply_key("right", e)
    assert rl.saves_without_reset_wait(e)  # one press is enough
    rl.apply_key("left", e)
    assert not rl.saves_without_reset_wait(e)  # Left (even after Right) = redo, never saved


def test_wait_without_verdict_outlasts_timeout_until_a_key():
    s, ctx = _strategy()
    _press_later(s, "up", 0.5)
    t0 = time.perf_counter()
    s._wait_for_verdict(ctx, reset_time_s=0.1, is_last=False)
    assert time.perf_counter() - t0 >= 0.5 and s._events["verdict"] == "fail"


def test_wait_key_ends_reset_early():
    s, ctx = _strategy()
    _press_later(s, "right", 0.1)
    t0 = time.perf_counter()
    s._wait_for_verdict(ctx, reset_time_s=30, is_last=False)
    assert time.perf_counter() - t0 < 5 and s._events["verdict"] == "success"


def test_wait_last_episode_with_verdict_returns_at_once():
    s, ctx = _strategy()
    s._events["verdict"] = "fail"
    t0 = time.perf_counter()
    s._wait_for_verdict(ctx, reset_time_s=30, is_last=True)
    assert time.perf_counter() - t0 < 1


def test_wait_redo_and_stop():
    s, ctx = _strategy()
    _press_later(s, "left", 0.1)
    s._wait_for_verdict(ctx, reset_time_s=30, is_last=False)
    assert s._events["rerecord_episode"]

    s, ctx = _strategy()
    _press_later(s, "esc", 0.1)
    s._wait_for_verdict(ctx, reset_time_s=30, is_last=False)
    assert s._events["stop_recording"]


# ---- run(): one Right/Up per episode, whether it ended early or ran out its time ----

class _FakeDataset:
    repo_id = "x/rollout_test"

    def __init__(self, save_s=0.0):
        self.num_episodes = 0
        self.save_s = save_s
        self.cleared = 0

    def save_episode(self):
        time.sleep(self.save_s)
        self.num_episodes += 1

    def clear_episode_buffer(self):
        self.cleared += 1


def _run(tmp_path, monkeypatch, policy_keys, n=2, save_s=0.0, wait_keys=()):
    """policy_keys[i]: keys pressed during episode attempt i's policy phase ([] = it ran out its time);
    wait_keys: (delay_s, key) pressed from a timer, for the reset phase."""
    import contextlib as _cl

    import lerobot.datasets

    monkeypatch.setattr(lerobot.datasets, "VideoEncodingManager", lambda ds: _cl.nullcontext())
    s = rl.make_strategy_class()(SimpleNamespace(reset_to_initial_position=True))
    said = []
    s._speaker = SimpleNamespace(say=lambda text, wait=False: said.append(text))
    ds = _FakeDataset(save_s)
    s._sheet = rl.MetaSheet(tmp_path / "m.csv", 0, n, {}, "c")
    s._engine = SimpleNamespace(reset=lambda: None, resume=lambda: None)
    s._interpolator = SimpleNamespace(reset=lambda: None)
    s._return_to_initial_position = lambda hw, duration_s=3.0: None
    s._go_home = lambda ctx: None
    attempts = iter(policy_keys)

    def policy_loop(**kw):
        for k in next(attempts):
            s._on_key(k)

    s._policy_loop = policy_loop
    cfg = SimpleNamespace(
        fps=15, task="t", play_sounds=False,
        dataset=SimpleNamespace(num_episodes=n, reset_time_s=30, episode_time_s=1, single_task="t"),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=threading.Event()),
        hardware=SimpleNamespace(robot_wrapper=None),
        data=SimpleNamespace(dataset=ds, dataset_features={}),
    )
    for delay, key in wait_keys:
        threading.Timer(delay, s._on_key, args=(key,)).start()
    t0 = time.perf_counter()
    s.run(ctx)
    return time.perf_counter() - t0, ds, said, [r["outcome"] for r in _rows(tmp_path / "m.csv")]


def test_run_early_verdict_saves_without_second_key(tmp_path, monkeypatch):
    dt, ds, said, outcomes = _run(tmp_path, monkeypatch, [["right"], ["up"]])
    assert dt < 5  # reset_time_s=30 never waited
    assert ds.num_episodes == 2 and outcomes == ["success", ""]
    assert said[:2] == ["Recording episode 0", "success"]  # verdict spoken at the press


def test_run_timed_out_episode_waits_for_one_key(tmp_path, monkeypatch):
    dt, ds, said, outcomes = _run(tmp_path, monkeypatch, [[]], n=1, wait_keys=[(0.3, "right")])
    assert 0.3 <= dt < 5 and ds.num_episodes == 1 and outcomes == ["success"]


def test_run_left_redoes_and_keys_during_save_are_ignored(tmp_path, monkeypatch, capsys):
    # attempt 0: Left in the policy phase -> discarded, reset phase ends on the next key
    # attempt 1: Right, then Left and Up while it is saving -> still saved as success
    dt, ds, said, outcomes = _run(
        tmp_path, monkeypatch, [["left"], ["right"]], n=1, save_s=0.8,
        wait_keys=[(0.2, "left"), (0.5, "left"), (0.6, "up")],
    )
    assert ds.cleared >= 1 and ds.num_episodes == 1 and outcomes == ["success"]
    out = capsys.readouterr().out
    assert "left ignored" in out and "up ignored" in out
    assert "fail" not in said
