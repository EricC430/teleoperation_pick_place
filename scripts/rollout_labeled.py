#!/usr/bin/env python
"""lerobot-rollout (strategy.type=episodic) with the success/fail verdict typed in at the robot.

Same arguments as `lerobot-rollout`, plus the episode_meta flags below:

    uv run python scripts/rollout_labeled.py --config_path configs/rollout_omx_b1_uvc60_eval.yaml \
        --policy.path=<repo> --policy.n_action_steps=30 --dataset.repo_id=<rollout_...> \
        --meta-template paper_cup_A1

At start it writes episode_meta/<stamped dataset name>.csv with one row per planned episode, copied
from the template's row with the same episode_index (episode_meta/templates/<name>.csv: edit it in
Excel, one row per episode; extra rows are ignored, a missing row stops the run before the first
episode starts). Flags given on the command line override the template for every row:
    --object-name / --object-orientation / --env-light / --env-bg   that column
    --placement-prefix c                                            placement_id = c<index+1>
Without a template, rows get only the flags (placement_id defaults to c<index+1>).

Keys (any time during the episode or the reset phase that follows it):
    Right  success  -> saved, outcome=success
    Up     fail     -> saved, outcome left blank (blank outcome = failed; fill no_grasp/... later)
    Left   redo     -> discarded, same episode index again
    Esc    stop     -> the unsaved episode is discarded; the arm returns to its start pose, then exit
The last Right/Up pressed before the save wins. An episode is never saved without a verdict: when the
reset time runs out and none was given, it keeps waiting for a key. Rows for episodes that were never
saved are removed when the session ends, so a blank outcome always means a saved failure.

Differences from lerobot's EpisodicStrategy: one voice line per transition (a new line cuts off the
previous one instead of playing over it); Esc and crashes discard the in-progress episode instead of
saving it; the arm returns to its start pose before the dataset is finalized. Teleop is not supported.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from annotate_episodes import DEFAULT_SCHEMA, KEY, csv_path_for, load_schema, read_csv, write_csv  # noqa: E402

logger = logging.getLogger(__name__)

ENTRYPOINT = "lerobot-rollout-labeled"  # how config_records/ names this command
TEMPLATE_DIR = SCRIPTS.parent / "episode_meta" / "templates"
PREFILL = ("object_name", "object_orientation", "env_light", "env_bg")
VERDICT_KEYS = {"right": "success", "up": "fail"}
OUTCOME = {"success": "success", "fail": ""}  # blank outcome = failed (user decision 2026-10-08)


def template_path(name: str) -> Path:
    """`paper_cup_A1` -> episode_meta/templates/paper_cup_A1.csv; anything with a suffix or a slash is a path."""
    p = Path(name)
    if p.suffix or len(p.parts) > 1:
        return p
    return TEMPLATE_DIR / f"{name}.csv"


def load_template(name: str) -> tuple[dict, list]:
    path = template_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"episode_meta template not found: {path}")
    return read_csv(str(path))


class MetaSheet:
    """The episode_meta CSV for one rollout session: prefilled at start, one outcome per saved episode.

    Each new row is the template row with the same episode_index (if a template is given), then the
    non-empty `prefill` values on top, then placement_id = <placement_prefix><index+1> if a prefix is given.
    """

    def __init__(self, path, first_index: int, count: int, prefill: dict, placement_prefix: str | None,
                 template: tuple[dict, list] | None = None, schema=DEFAULT_SCHEMA):
        self.path = str(path)
        _, self.fields = load_schema(schema)
        self.rows, self.columns = read_csv(self.path)  # existing rows (e.g. --resume) are kept as they are
        t_rows, t_columns = template or ({}, [])
        self.columns = self.columns or t_columns
        wanted = range(first_index, first_index + count)
        if template is not None:
            missing = [i for i in wanted if i not in t_rows and i not in self.rows]
            if missing:
                raise ValueError(f"template has no row for episode(s) {missing}: it needs episode_index "
                                 f"{first_index}..{first_index + count - 1}")
        for idx in wanted:
            if idx in self.rows:
                continue
            row = {**t_rows.get(idx, {}), KEY: str(idx), **{k: v for k, v in prefill.items() if v}}
            if placement_prefix:
                row["placement_id"] = f"{placement_prefix}{idx + 1}"
            self.rows[idx] = row
        self.write()

    def write(self) -> None:
        write_csv(self.path, self.rows, self.fields, self.columns)

    def set_verdict(self, idx: int, verdict: str) -> None:
        self.rows.setdefault(idx, {KEY: str(idx)})["outcome"] = OUTCOME[verdict]
        self.write()

    def prune(self, saved_count: int) -> list[int]:
        """Drop rows for episodes that were never saved, so a blank outcome only ever means 'failed'."""
        dropped = sorted(i for i in self.rows if i >= saved_count)
        for i in dropped:
            del self.rows[i]
        self.write()
        return dropped


def apply_key(name: str, events: dict) -> bool:
    """Update the shared events dict for one key press. Returns True if the key is one of ours."""
    if name in VERDICT_KEYS:
        events["verdict"] = VERDICT_KEYS[name]
    elif name == "left":
        events["rerecord_episode"] = True
    elif name == "esc":
        events["stop_recording"] = True
    else:
        return False
    events["exit_early"] = True
    return True


def new_events() -> dict:
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False, "verdict": None}


class Speaker:
    """One voice line at a time: starting a new line stops the one still playing."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._proc = None

    def say(self, text: str, wait: bool = False) -> None:
        logger.info(text)
        if not self.enabled:
            return
        if platform.system() != "Windows":
            from lerobot.utils.utils import say

            say(text, blocking=wait)
            return
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
        try:
            self._proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
            if wait:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=10)
        except FileNotFoundError as e:
            logger.warning("Text-to-speech failed: %s", e)


def make_strategy_class():
    from lerobot.datasets import VideoEncodingManager
    from lerobot.rollout.strategies.core import safe_push_to_hub
    from lerobot.rollout.strategies.episodic import EpisodicStrategy
    from lerobot.utils.keyboard_input import create_key_listener

    class LabeledEpisodicStrategy(EpisodicStrategy):
        meta_args: argparse.Namespace  # set by main()

        def __init__(self, config) -> None:
            super().__init__(config)
            self._events = new_events()
            self._speaker = Speaker(False)
            self._pending = ""  # verdict line carried into the next voice line
            self._homed = False
            self._sheet = None

        def setup(self, ctx) -> None:
            if ctx.hardware.teleop is not None:
                raise ValueError("rollout_labeled.py does not support a teleop; use lerobot-rollout")
            self._init_engine(ctx)
            self._speaker = Speaker(ctx.runtime.cfg.play_sounds)
            dataset = ctx.data.dataset
            self._sheet = MetaSheet(
                csv_path_for(dataset.repo_id, self.meta_args.csv),
                first_index=dataset.num_episodes,
                count=ctx.runtime.cfg.dataset.num_episodes,
                prefill={k: getattr(self.meta_args, k) for k in PREFILL},
                placement_prefix=default_placement_prefix(self.meta_args),
                template=self.meta_args.template_data,
            )
            logger.info("episode_meta: %s", self._sheet.path)

            def dispatch(name: str) -> None:
                if apply_key(name, self._events):
                    print(f"[key] {name} -> {self._describe()}", flush=True)

            self._listener = create_key_listener(
                dispatch, controls_help="Right = success, Up = fail, Left = redo, Esc = stop"
            )

        def _describe(self) -> str:
            e = self._events
            if e["stop_recording"]:
                return "STOP (this episode is discarded)"
            if e["rerecord_episode"]:
                return "REDO"
            return {"success": "success", "fail": "FAIL", None: "no verdict yet"}[e["verdict"]]

        def run(self, ctx) -> None:
            cfg = ctx.runtime.cfg
            dataset = ctx.data.dataset
            events = self._events
            num_episodes = cfg.dataset.num_episodes
            reset_time_s = cfg.dataset.reset_time_s
            single_task = cfg.dataset.single_task or cfg.task
            redo = False

            with VideoEncodingManager(dataset):
                try:
                    recorded = 0
                    while recorded < num_episodes and not events["stop_recording"]:
                        if ctx.runtime.shutdown_event.is_set():
                            break
                        idx = dataset.num_episodes
                        events.update(verdict=None, exit_early=False, rerecord_episode=False)

                        self._engine.reset()
                        self._interpolator.reset()
                        self._engine.resume()

                        line = f"Re-record episode {idx}" if redo else f"Recording episode {idx}"
                        self._speaker.say(self._pending + line)
                        self._pending = ""
                        self._policy_loop(
                            ctx=ctx,
                            robot=ctx.hardware.robot_wrapper,
                            events=events,
                            features=ctx.data.dataset_features,
                            fps=cfg.fps,
                            control_time_s=cfg.dataset.episode_time_s,
                            dataset=dataset,
                            single_task=single_task,
                        )
                        if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                            break

                        if self.config.reset_to_initial_position:
                            self._return_to_initial_position(hw=ctx.hardware, duration_s=1)
                        is_last = recorded == num_episodes - 1
                        if not is_last or events["rerecord_episode"]:
                            self._speaker.say("Reset the environment")
                        self._wait_for_verdict(ctx, reset_time_s, is_last)
                        if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                            break

                        if events["rerecord_episode"]:
                            dataset.clear_episode_buffer()
                            redo = True
                            continue

                        verdict = events["verdict"]
                        dataset.save_episode()
                        self._sheet.set_verdict(idx, verdict)
                        print(f"[saved] episode {idx}: {'success' if verdict == 'success' else 'FAIL'}", flush=True)
                        self._pending = f"Saved, {verdict}. "
                        recorded += 1
                        redo = False
                finally:
                    # Esc / Ctrl-C / crash: the in-progress episode has no verdict -> discard it, never save it.
                    with contextlib.suppress(Exception):
                        dataset.clear_episode_buffer()
                    # Home before the with-block finalizes the dataset (that can take a while).
                    self._go_home(ctx)

        def _wait_for_verdict(self, ctx, reset_time_s: float, is_last: bool) -> None:
            """Reset phase. Ends on a key, or on timeout once there is a verdict (or a redo). Never on timeout alone."""
            events = self._events
            events["exit_early"] = False  # a key pressed during the policy phase must not also end this phase
            start = time.perf_counter()
            prompted = False
            while not events["stop_recording"] and not ctx.runtime.shutdown_event.is_set():
                decided = events["rerecord_episode"] or events["verdict"] is not None
                if events["exit_early"]:
                    events["exit_early"] = False
                    if decided:
                        return
                timed_out = time.perf_counter() - start >= reset_time_s
                # last episode: nothing to re-place, so only a missing verdict (or a redo) keeps us here
                if (is_last and not events["rerecord_episode"]) or timed_out:
                    if decided:
                        return
                    if not prompted:
                        print("[wait] no verdict yet: Right = success, Up = fail, Left = redo", flush=True)
                        self._speaker.say("Right for success, up for fail")
                        prompted = True
                time.sleep(0.05)

        def _go_home(self, ctx) -> None:
            hw = ctx.hardware
            if self._homed or not ctx.runtime.cfg.return_to_initial_position or not hw.initial_position:
                return
            if hw.robot_wrapper.inner.is_connected:
                logger.info("Returning robot to initial position...")
                self._return_to_initial_position(hw)
                self._homed = True

        def teardown(self, ctx) -> None:
            cfg = ctx.runtime.cfg
            self._go_home(ctx)  # run() normally did this already; covers a failure inside setup()
            self._speaker.say(self._pending + "Stop recording", wait=True)
            if self._listener is not None:
                self._listener.stop()
            dataset = ctx.data.dataset
            if dataset is not None:
                dataset.finalize()
                if self._sheet is not None:
                    dropped = self._sheet.prune(dataset.num_episodes)
                    if dropped:
                        logger.info("episode_meta: removed rows never recorded: %s", dropped)
                    logger.info("episode_meta: %s", self._sheet.path)
                if cfg.dataset.push_to_hub and safe_push_to_hub(
                    dataset, tags=cfg.dataset.tags, private=cfg.dataset.private
                ):
                    self._speaker.say("Dataset uploaded to hub", wait=True)
            self._teardown_hardware(ctx.hardware, return_to_initial_position=False)
            self._speaker.say("Exiting")

    return LabeledEpisodicStrategy


def parse_meta_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--object-name", dest="object_name", default="")
    p.add_argument("--object-orientation", dest="object_orientation", default="")
    p.add_argument("--env-light", dest="env_light", default="")
    p.add_argument("--env-bg", dest="env_bg", default="")
    p.add_argument("--placement-prefix", dest="placement_prefix", default=None,
                   help="placement_id = <prefix><episode_index + 1>. Default: 'c' without a template, "
                        "the template's own placement_id with one; empty string = leave as is")
    p.add_argument("--meta-template", dest="template", default=None,
                   help="name under episode_meta/templates/ (without .csv) or a path to a template CSV")
    p.add_argument("--meta-csv", dest="csv", default=None,
                   help="CSV path; default episode_meta/<stamped dataset name>.csv")
    meta, rest = p.parse_known_args(argv)
    meta.template_data = load_template(meta.template) if meta.template else None
    return meta, rest


def default_placement_prefix(meta: argparse.Namespace) -> str | None:
    if meta.placement_prefix is not None:
        return meta.placement_prefix
    return None if meta.template else "c"


def main() -> None:
    meta, rest = parse_meta_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]  # lerobot's parser must not see our flags

    try:
        from lerobot_robot_config_record.recorder import record_from_argv

        record_from_argv(argv=[ENTRYPOINT, *rest])
    except Exception as e:  # noqa: BLE001 - never block the robot command
        print(f"[config-record] WARNING: could not record the config: {e!r}", file=sys.stderr, flush=True)

    import lerobot.scripts.lerobot_rollout as lerobot_rollout

    strategy_cls = make_strategy_class()
    strategy_cls.meta_args = meta

    def create_strategy(config):
        if config.type != "episodic":
            raise ValueError(f"rollout_labeled.py needs --strategy.type=episodic, got {config.type!r}")
        return strategy_cls(config)

    lerobot_rollout.create_strategy = create_strategy
    lerobot_rollout.main()


if __name__ == "__main__":
    main()
