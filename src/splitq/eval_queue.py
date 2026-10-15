"""Explicit multi-GPU evaluation queue with restart, drain and cancellation support."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .checkpoint import read_manifest, write_json

FULL_COUNTS = {
    "scienceqa": {"scienceqa": 4241},
    "ocrbench": {"ocrbench": 1000},
    "mmmu_val": {"mmmu_val": 900},
    "textvqa_val": {"textvqa_val": 5000},
    "vizwiz_vqa_val": {"vizwiz_vqa_val": 4319},
    "seedbench": {"seedbench": 17990},
}
TERMINAL = {"complete", "failed", "cancelled"}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def process_alive(pid, output):
    """Verify the command line before treating a reused PID as a tracked evaluation."""
    if not pid:
        return False
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return b"splitq.evaluate" in arguments and os.fsencode(str(output)) in arguments
    except FileNotFoundError:
        return False


def choose_output(base):
    base = Path(base)
    if not base.exists() or not any(base.iterdir()):
        return base
    index = 1
    while True:
        candidate = base.with_name(base.name + f"_retry{index}")
        if not candidate.exists():
            return candidate
        index += 1


def completed_result(job, output, limit=None):
    directory = Path(output)
    run = read_json(directory / "run.json")
    metrics = read_json(directory / "metrics.json")
    if not run or not metrics:
        return None
    if (
        Path(run["checkpoint"]).resolve() != Path(job["checkpoint"]).resolve()
        or run["tasks"] != [job["task"]]
        or run.get("limit") != limit
    ):
        raise ValueError(f"Result does not match the request: {directory}")
    if "checkpoint_sha256" in run:
        manifest = read_json(Path(job["checkpoint"]) / "manifest.json")
        if not manifest or manifest["checkpoint"]["sha256"] != run["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed; cannot reuse results: {directory}")
    counts = metrics.get("n-samples", {})
    expected = FULL_COUNTS[job["task"]]
    if any(
        counts.get(key, {}).get("effective") != (min(size, limit) if limit else size)
        for key, size in expected.items()
    ):
        raise ValueError(f"Incomplete result sample count: {directory}：{counts}")
    return {
        "results": metrics.get("results"),
        "groups": metrics.get("groups"),
        "n_samples": counts,
        "output": str(directory),
    }


def checkpoint_ready(path):
    manifest = read_json(Path(path) / "manifest.json")
    if not manifest or not manifest.get("complete"):
        return False
    # Validate the format and checksum before launching a subprocess.
    read_manifest(path)
    return True


def free_gpus(allowed, memory_limit):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        int(index)
        for line in output.splitlines()
        for index, memory in [line.split(",")]
        if int(index) in allowed and int(memory) < memory_limit
    ]


def environment(gpu):
    # Inherit network/cache settings; only select the GPU.
    result = os.environ.copy()
    result["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return result


def run_queue(directory, spec):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(directory / "state.json")
        if old and old["spec"] != spec:
            raise ValueError(
                "Queue specification differs from saved state; use a new --queue-dir"
            )
        jobs = [
            {
                "id": f"{index}_{task}",
                "checkpoint": checkpoint,
                "task": task,
                "output": str(
                    directory / "results" / f"{index}_{Path(checkpoint).name}_{task}"
                ),
            }
            for task in spec["tasks"]
            for index, checkpoint in enumerate(spec["checkpoints"])
        ]
        states = (
            old["jobs"]
            if old
            else {job["id"]: {"status": "pending", "attempts": 0} for job in jobs}
        )
        children = {}
        ready = set()
        last_message = 0

        def persist():
            write_json(
                directory / "state.json",
                {
                    "spec": spec,
                    "jobs": states,
                    "updated": time.time(),
                    "finished": all(
                        record["status"] in TERMINAL for record in states.values()
                    ),
                    "complete": all(
                        record["status"] == "complete" for record in states.values()
                    ),
                },
            )

        persist()
        while True:
            control = read_json(directory / "control.json") or {}
            stopping = control.get("drain") or control.get("cancel")
            for job in jobs:
                record = states[job["id"]]
                output = record.get("output", job["output"])
                if record["status"] in {"pending", "running"}:
                    result = completed_result(job, output, spec["limit"])
                    if result is not None:
                        record.update(status="complete", result=result)
                        child = children.get(job["id"])
                        if child:
                            child.poll()
                        continue
                if stopping and record["status"] == "pending":
                    record.update(status="cancelled")
                if record["status"] != "running":
                    continue
                if control.get("cancel") and process_alive(record["pid"], output):
                    os.kill(record["pid"], signal.SIGTERM)
                child = children.get(job["id"])
                alive = (
                    child.poll() is None
                    if child
                    else process_alive(record["pid"], output)
                )
                if not alive:
                    record.update(
                        status=(
                            "cancelled"
                            if control.get("cancel")
                            else (
                                "pending"
                                if record["attempts"] < spec["max_attempts"]
                                and not stopping
                                else "failed"
                            )
                        ),
                        exit_code=child.returncode if child else None,
                    )
            occupied = {
                record["gpu"]
                for record in states.values()
                if record["status"] == "running"
            }
            available = (
                [] if stopping else free_gpus(spec["gpus"], spec["memory_limit"])
            )
            for gpu in available:
                if gpu in occupied:
                    continue
                job = next(
                    (
                        job
                        for job in jobs
                        if states[job["id"]]["status"] == "pending"
                        and (
                            job["checkpoint"] in ready
                            or checkpoint_ready(job["checkpoint"])
                        )
                    ),
                    None,
                )
                if job is None:
                    break
                ready.add(job["checkpoint"])
                record = states[job["id"]]
                output = choose_output(record.get("output", job["output"]))
                record["attempts"] += 1
                log_path = (
                    directory
                    / f"{job['id']}_attempt{record['attempts']}_{time.time_ns()}.log"
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "splitq.evaluate",
                    "--checkpoint",
                    job["checkpoint"],
                    "--model-path",
                    spec["model_path"],
                    "--tasks",
                    job["task"],
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda:0",
                ]
                if spec["cache_root"]:
                    command += ["--cache-root", spec["cache_root"]]
                if spec["limit"]:
                    command += ["--limit", str(spec["limit"])]
                with log_path.open("x") as log:
                    child = subprocess.Popen(
                        command,
                        env=environment(gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                children[job["id"]] = child
                record.update(
                    status="running",
                    pid=child.pid,
                    gpu=gpu,
                    output=str(output),
                    log=str(log_path),
                )
                print("START", job["id"], "GPU", gpu, "PID", child.pid, flush=True)
                persist()
            persist()
            if all(record["status"] in TERMINAL for record in states.values()):
                return int(
                    any(record["status"] == "failed" for record in states.values())
                )
            if time.time() - last_message >= spec["log_seconds"]:
                print(
                    "STATUS",
                    {key: value["status"] for key, value in states.items()},
                    flush=True,
                )
                last_message = time.time()
            time.sleep(spec["poll_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status", "drain", "cancel"))
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--tasks", help="Tasks with complete local scoring: " + ",".join(FULL_COUNTS)
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cache-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=2000,
        help="Maximum memory use for an available GPU, in MiB",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--log-seconds", type=int, default=600)
    args = parser.parse_args()
    directory = Path(args.queue_dir).resolve()
    if args.mode == "status":
        print(
            json.dumps(
                read_json(directory / "state.json"), ensure_ascii=False, indent=2
            )
        )
        return
    if args.mode in {"drain", "cancel"}:
        if not (directory / "state.json").is_file():
            parser.error("Queue has not been created")
        write_json(directory / "control.json", {args.mode: True})
        return
    if not args.checkpoint or not args.model_path or not args.tasks:
        parser.error("run requires --checkpoint, --model-path and --tasks")
    tasks = args.tasks.split(",")
    if len(set(tasks)) != len(tasks) or any(task not in FULL_COUNTS for task in tasks):
        parser.error(
            "Tasks must be unique and locally scorable; use evaluate separately for MMMU test"
        )
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must contain comma-separated nonnegative integers")
    if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
        parser.error("--gpus must be unique and nonnegative")
    if (
        args.max_attempts < 1
        or args.memory_limit < 1
        or args.poll_seconds < 10
        or args.log_seconds < args.poll_seconds
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Invalid retry, polling or limit configuration")
    spec = {
        "checkpoints": [str(Path(path).resolve()) for path in args.checkpoint],
        "model_path": str(Path(args.model_path).resolve()),
        "tasks": tasks,
        "gpus": gpus,
        "cache_root": (
            str(Path(args.cache_root).expanduser().resolve())
            if args.cache_root
            else None
        ),
        "limit": args.limit,
        "max_attempts": args.max_attempts,
        "memory_limit": args.memory_limit,
        "poll_seconds": args.poll_seconds,
        "log_seconds": args.log_seconds,
    }
    if len(set(spec["checkpoints"])) != len(spec["checkpoints"]):
        parser.error("Checkpoints must be unique")
    raise SystemExit(run_queue(directory, spec))


if __name__ == "__main__":
    main()
