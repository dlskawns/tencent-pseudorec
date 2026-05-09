"""Local pre-submission validator — CLAUDE.md §14 G1–G6.

Run before every submission:

    EVAL_DATA_PATH=/abs/path/to/test_dir \
    EVAL_RESULT_PATH=/abs/path/to/out_dir \
    .venv-arm64/bin/python submission/local_validate.py

Outputs PASS/FAIL per gate + a single non-zero exit code on any failure.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
INFER = HERE / "infer.py"


FORBIDDEN_IMPORTS = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "http", "socket"}


def _read_infer_source() -> str:
    return INFER.read_text(encoding="utf-8")


def gate_g1_signature() -> tuple[bool, str]:
    src = _read_infer_source()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"infer.py syntax error: {e}"
    has_main = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True
            args = node.args
            n_args = (
                len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
                + (1 if args.vararg else 0) + (1 if args.kwarg else 0)
            )
            if n_args != 0:
                return False, f"main() must take 0 args, found {n_args}"
    if not has_main:
        return False, "missing top-level def main()"
    if "argparse" in src:
        return False, "argparse import detected (forbidden by §13.3)"
    return True, "ok"


def gate_g2_env_only_paths() -> tuple[bool, str]:
    src = _read_infer_source()
    bad = []
    for m in re.finditer(r'["\'](/[^"\']+)["\']', src):
        path_lit = m.group(1)
        if path_lit.startswith("/tmp") or path_lit in {"/", "//"}:
            continue
        if path_lit.startswith(("/usr", "/lib", "/etc", "/opt", "/Users", "/home", "/var", "/data", "/mnt")):
            bad.append(path_lit)
    if bad:
        return False, f"absolute path literals found: {bad[:3]}"
    return True, "ok"


def gate_g3_g4_run_and_check() -> tuple[bool, str]:
    eval_data = os.environ.get("EVAL_DATA_PATH")
    eval_out = os.environ.get("EVAL_RESULT_PATH")
    if not eval_data or not eval_out:
        return False, "EVAL_DATA_PATH / EVAL_RESULT_PATH env not set"

    files = sorted(Path(eval_data).rglob("*.parquet")) if Path(eval_data).is_dir() else [Path(eval_data)]
    if not files:
        return False, f"no .parquet files under {eval_data}"

    expected_uids = set()
    for f in files:
        col = pq.ParquetFile(str(f)).read(columns=["user_id"])
        expected_uids.update(str(v) for v in col.column("user_id").to_pylist())

    Path(eval_out).mkdir(parents=True, exist_ok=True)
    out_file = Path(eval_out) / "predictions.json"
    if out_file.exists():
        out_file.unlink()

    proc = subprocess.run(
        [sys.executable, str(INFER)],
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return False, f"infer.py exit={proc.returncode} stderr={proc.stderr[-500:]}"

    if not out_file.exists():
        return False, f"predictions.json not written at {out_file}"

    with out_file.open("r") as fh:
        obj = json.load(fh)
    if "predictions" not in obj:
        return False, "top-level 'predictions' key missing"
    preds = obj["predictions"]
    if not isinstance(preds, dict):
        return False, "predictions is not a dict"

    actual_uids = set(preds.keys())
    missing = expected_uids - actual_uids
    extra = actual_uids - expected_uids
    if missing or extra:
        return False, f"user_id coverage mismatch: missing={len(missing)} extra={len(extra)}"

    bad_values = []
    for k, v in preds.items():
        if not isinstance(v, (int, float)):
            bad_values.append((k, type(v).__name__))
            continue
        if not (0.0 <= float(v) <= 1.0):
            bad_values.append((k, v))
    if bad_values:
        return False, f"value out-of-range or wrong type: {bad_values[:3]}"

    other = [str(p) for p in Path(eval_out).rglob("*") if p != out_file and p.is_file()]
    other = [p for p in other if not p.endswith(".tmp")]
    if other:
        return False, f"unexpected output files outside predictions.json: {other[:3]}"

    return True, f"ok (n_users={len(preds)})"


def gate_g5_determinism() -> tuple[bool, str]:
    eval_out = os.environ.get("EVAL_RESULT_PATH")
    if not eval_out:
        return False, "EVAL_RESULT_PATH unset"
    out_file = Path(eval_out) / "predictions.json"
    if not out_file.exists():
        return False, "predictions.json missing (run G3 first)"
    first = out_file.read_bytes()

    with tempfile.TemporaryDirectory() as td:
        os.environ["EVAL_RESULT_PATH"] = td
        proc = subprocess.run(
            [sys.executable, str(INFER)],
            env={**os.environ, "PYTHONHASHSEED": "0"},
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            os.environ["EVAL_RESULT_PATH"] = eval_out
            return False, f"second-run failed: {proc.stderr[-500:]}"
        second = (Path(td) / "predictions.json").read_bytes()

    os.environ["EVAL_RESULT_PATH"] = eval_out
    if first != second:
        return False, "predictions.json bytes differ across two runs"
    return True, "ok (bit-identical across two runs)"


def gate_g6_no_internet() -> tuple[bool, str]:
    src = _read_infer_source()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                bad.append(node.module or "")
    if bad:
        return False, f"forbidden network imports: {bad}"
    return True, "ok"


def main() -> int:
    gates = [
        ("G1 signature", gate_g1_signature),
        ("G2 env-only paths", gate_g2_env_only_paths),
        ("G3+G4 run+coverage", gate_g3_g4_run_and_check),
        ("G5 determinism", gate_g5_determinism),
        ("G6 no-internet", gate_g6_no_internet),
    ]
    fails = 0
    for name, fn in gates:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {e!r}"
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}: {msg}")
        if not ok:
            fails += 1
    print(f"\nResult: {len(gates) - fails}/{len(gates)} gates passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
