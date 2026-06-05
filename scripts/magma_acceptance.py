"""One-command MAGMA acceptance check.

This script is the daily confidence gate for the production memory layer. It
combines health, recall quality, product-level behavior, and L1 distillation
reachability into one red/yellow/green report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configured_api_base() -> str:
    env_base = os.environ.get("MAGMA_API_BASE")
    if env_base:
        return env_base.rstrip("/")
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'"apiBaseUrl"\s*:\s*"(http://127\.0\.0\.1:\d+)"', text)
        if match:
            return match.group(1).rstrip("/")
    except OSError:
        pass
    return "http://127.0.0.1:8902"


def run_python(args: list[str], timeout: int) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_ms": round((time.time() - started) * 1000),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_json_output(result: dict[str, Any]) -> Any:
    text = result.get("stdout") or ""
    if not text:
        return None
    return json.loads(text)


def api_get_json(api_base: str, path: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(f"{api_base.rstrip('/')}{path}", timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get((status or "").lower(), 2)


def merge_status(current: str, candidate: str) -> str:
    return candidate if status_rank(candidate) > status_rank(current) else current


def add_failure(report: dict[str, Any], check: str, message: str) -> None:
    report["failures"].append({"check": check, "message": message})


def add_warning(report: dict[str, Any], check: str, message: str) -> None:
    report["warnings"].append({"check": check, "message": message})


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    api_base = args.api_base.rstrip("/")
    report: dict[str, Any] = {
        "overall": "green",
        "api_base": api_base,
        "checks": {},
        "failures": [],
        "warnings": [],
    }

    # Health endpoint is checked directly so a broken doctor script is visible.
    try:
        health = api_get_json(api_base, "/api/v1/health")
        ok = health.get("status") == "ok"
        report["checks"]["api_health"] = {"ok": ok, "response": health}
        if not ok:
            add_failure(report, "api_health", f"Unexpected health response: {health}")
            report["overall"] = "red"
    except Exception as exc:
        report["checks"]["api_health"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        add_failure(report, "api_health", f"{type(exc).__name__}: {exc}")
        report["overall"] = "red"

    doctor_run = run_python(["scripts/magma_doctor.py", "--json"], timeout=60)
    doctor_check: dict[str, Any] = {"command": "scripts/magma_doctor.py --json", **doctor_run}
    if doctor_run["ok"]:
        try:
            doctor = parse_json_output(doctor_run)
            doctor_check["summary"] = {
                "overall": doctor.get("overall"),
                "failures": doctor.get("failures", []),
                "warnings": doctor.get("warnings", []),
            }
            doctor_status = (doctor.get("overall") or "red").lower()
            report["overall"] = merge_status(report["overall"], doctor_status)
            if doctor_status == "red":
                add_failure(report, "doctor", "doctor reported red")
            elif doctor_status == "yellow":
                add_warning(report, "doctor", "doctor reported yellow")
        except Exception as exc:
            doctor_check["json_error"] = f"{type(exc).__name__}: {exc}"
            add_failure(report, "doctor", "doctor output was not valid JSON")
            report["overall"] = "red"
    else:
        add_failure(report, "doctor", doctor_run.get("stderr") or "doctor failed")
        report["overall"] = "red"
    report["checks"]["doctor"] = doctor_check

    compile_files = [
        "magma/api/server.py",
        "magma/api/mcp_server.py",
        "magma/search.py",
        "magma/short_command.py",
        "magma/capture_policy.py",
        "magma/l1_distiller.py",
    ]
    compile_run = run_python(["-m", "py_compile", *compile_files], timeout=60)
    report["checks"]["py_compile"] = {"command": f"python -m py_compile {' '.join(compile_files)}", **compile_run}
    if not compile_run["ok"]:
        add_failure(report, "py_compile", compile_run.get("stderr") or "py_compile failed")
        report["overall"] = "red"

    previous_api_base = os.environ.get("MAGMA_API_BASE")
    os.environ["MAGMA_API_BASE"] = api_base
    try:
        recall_run = run_python(["scripts/magma_recall_eval.py", "--json"], timeout=180)
    finally:
        if previous_api_base is None:
            os.environ.pop("MAGMA_API_BASE", None)
        else:
            os.environ["MAGMA_API_BASE"] = previous_api_base
    recall_check: dict[str, Any] = {"command": "scripts/magma_recall_eval.py --json", **recall_run}
    if recall_run["ok"]:
        try:
            recall = parse_json_output(recall_run)
            recall_total = recall.get("total") or recall.get("max_total")
            recall_passed = recall.get("passed")
            if recall_passed is None and recall.get("pct") == 100.0:
                recall_passed = recall_total
            recall_check["summary"] = {
                "passed": recall_passed,
                "total": recall_total,
                "pct": recall.get("pct"),
            }
            if recall.get("pct", 0) < args.min_recall_pct:
                add_failure(report, "recall_eval", f"recall pct {recall.get('pct')} < {args.min_recall_pct}")
                report["overall"] = "red"
        except Exception as exc:
            recall_check["json_error"] = f"{type(exc).__name__}: {exc}"
            add_failure(report, "recall_eval", "recall eval output was not valid JSON")
            report["overall"] = "red"
    else:
        add_failure(report, "recall_eval", recall_run.get("stderr") or "recall eval failed")
        report["overall"] = "red"
    report["checks"]["recall_eval"] = recall_check

    if not args.skip_benchmark:
        benchmark_run = run_python(["scripts/magma_product_benchmark.py", "--json", "--api-base", api_base], timeout=420)
        benchmark_check: dict[str, Any] = {"command": "scripts/magma_product_benchmark.py --json", **benchmark_run}
        if benchmark_run["ok"]:
            try:
                benchmark = parse_json_output(benchmark_run)
                benchmark_check["summary"] = {
                    "passed": benchmark.get("passed"),
                    "total": benchmark.get("total"),
                    "pct": benchmark.get("pct"),
                }
                if benchmark.get("pct", 0) < args.min_benchmark_pct:
                    add_failure(report, "product_benchmark", f"benchmark pct {benchmark.get('pct')} < {args.min_benchmark_pct}")
                    report["overall"] = "red"
            except Exception as exc:
                benchmark_check["json_error"] = f"{type(exc).__name__}: {exc}"
                add_failure(report, "product_benchmark", "benchmark output was not valid JSON")
                report["overall"] = "red"
        else:
            add_failure(report, "product_benchmark", benchmark_run.get("stderr") or "product benchmark failed")
            report["overall"] = "red"
        report["checks"]["product_benchmark"] = benchmark_check

    l1_run = run_python(
        ["scripts/magma_l1_runner.py", "--json", "--hours", str(args.l1_hours), "--limit", str(args.l1_limit), "--api-base", api_base],
        timeout=180,
    )
    l1_check: dict[str, Any] = {"command": "scripts/magma_l1_runner.py --json", **l1_run}
    if l1_run["ok"]:
        try:
            l1 = parse_json_output(l1_run)
            l1_check["summary"] = {
                "scanned": l1.get("scanned"),
                "candidate_count": l1.get("candidate_count"),
                "written_count": l1.get("written_count"),
                "dry_run": l1.get("status") == "dry_run",
                "by_kind": l1.get("by_kind"),
            }
        except Exception as exc:
            l1_check["json_error"] = f"{type(exc).__name__}: {exc}"
            add_warning(report, "l1_distill", "L1 dry-run output was not valid JSON")
            report["overall"] = merge_status(report["overall"], "yellow")
    else:
        add_warning(report, "l1_distill", l1_run.get("stderr") or "L1 dry-run failed")
        report["overall"] = merge_status(report["overall"], "yellow")
    report["checks"]["l1_distill_dry_run"] = l1_check

    return report


def print_human(report: dict[str, Any]) -> None:
    print(f"MAGMA acceptance: {report['overall'].upper()} ({report['api_base']})")
    for name, check in report["checks"].items():
        marker = "OK" if check.get("ok", True) else "FAIL"
        summary = check.get("summary")
        if summary:
            print(f"- {name}: {marker} {json.dumps(summary, ensure_ascii=False)}")
        else:
            print(f"- {name}: {marker}")
    if report["warnings"]:
        print("Warnings:")
        for item in report["warnings"]:
            print(f"- {item['check']}: {item['message']}")
    if report["failures"]:
        print("Failures:")
        for item in report["failures"]:
            print(f"- {item['check']}: {item['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MAGMA production acceptance gate.")
    parser.add_argument("--api-base", default=configured_api_base())
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the slower product benchmark.")
    parser.add_argument("--min-recall-pct", type=float, default=100.0)
    parser.add_argument("--min-benchmark-pct", type=float, default=100.0)
    parser.add_argument("--l1-hours", type=int, default=24)
    parser.add_argument("--l1-limit", type=int, default=200)
    args = parser.parse_args()

    report = run_acceptance(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0 if report["overall"] != "red" else 1


if __name__ == "__main__":
    raise SystemExit(main())
