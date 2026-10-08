#!/usr/bin/env python3
"""Delete run folders that are already safely archived locally -- on the server and in the local backup.

Nothing is deleted by the *plan* commands. A folder is only listed as safe when
  * a local copy exists (IaCGOD/benchmark_runs/<dir> or IaCGOD/runs/**/<run_id>) with exactly the same file names and sizes
    (add --strict to also compare sha256 of every file), and
  * for runs/: the run is finished (final_report.json present) and has not been modified for --min-age minutes, and
  * for benchmark_runs/: the folder belongs to a finished experiment (see FINISHED below / --include-unfulfilled).
The delete commands re-read the plan file, re-validate every path, and ask for a typed confirmation.

  python3 scripts/cleanup_synced.py backup-plan            # local backup  (../backup/{runs,benchmark_runs})
  python3 scripts/cleanup_synced.py backup-delete
  python3 scripts/cleanup_synced.py server-plan            # asks for the SSH password once (connection is shared)
  python3 scripts/cleanup_synced.py server-delete
Plans are written to IaCGOD/.cleanup_plan/*.txt (one relative path per line) -- read them before deleting.
"""
import argparse, hashlib, re, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT.parent / "backup"
PLAN = ROOT / ".cleanup_plan"
REMOTE = "tianyi@100.117.64.38"
REMOTE_ROOT = "/home/tianyi/iac-research/iac-god"
RUN_ID = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9._+ -]+(/[A-Za-z0-9._+ -]+)*$")


def finished_dirs(include_unfulfilled):
    """benchmark_runs folders of finished experiments (fulfilled in final_results/SUMMARY.csv) plus folders of no experiment (legacy)."""
    import importlib.util, csv
    spec = importlib.util.spec_from_file_location("fr", ROOT / "scripts" / "finalize_results.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    summ = {r["experiment"]: r["fulfilled"] == "True" for r in csv.DictReader(open(ROOT / "final_results" / "SUMMARY.csv"))}
    keep, done = set(), set()
    for e in m.EXPERIMENTS:
        (done if (summ.get(e["name"]) or include_unfulfilled) else keep).update(e["dirs"])
    return done, keep            # keep = folders of experiments that are not fulfilled yet (may still be resumed)


def local_files(d, strict=False):
    out = {}
    for p in d.rglob("*"):
        if p.is_file() and p.name != ".DS_Store":
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest() if strict else p.stat().st_size
    return out


def covers(loc, srv):
    return all(f in loc and loc[f] == v for f, v in srv.items())


def run_ssh(cmd, local_dir=None):
    """Run a shell command on the server (or in a local directory for testing with --remote local:/dir)."""
    if local_dir:
        return subprocess.run(["bash", "-c", cmd], cwd=local_dir, capture_output=True, text=True, check=True).stdout
    ctl = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/ssh_cleanup_%r@%h:%p", "-o", "ControlPersist=20m"]
    return subprocess.run(["ssh", *ctl, REMOTE, f"cd {REMOTE_ROOT} && {cmd}"], capture_output=True, text=True, check=True).stdout


def manifest(text, strict):
    """'path<TAB>value' lines -> {top: {relpath: value}} (top = first path component)"""
    out = defaultdict(dict)
    for ln in text.splitlines():
        if "\t" not in ln: continue
        p, v = ln.rsplit("\t", 1)
        top, _, rest = p.partition("/")
        out[top][rest] = v if strict else int(v)
    return out


def list_manifest(sub, strict, ld):
    """{top: {relpath: size|sha256}} of <server>/<sub> (ld = local directory standing in for the server, for testing)."""
    if ld:
        out = defaultdict(dict)
        for p in (ld / sub).rglob("*"):
            if p.is_file() and p.name != ".DS_Store":
                rel = str(p.relative_to(ld / sub)); top, _, rest = rel.partition("/")
                out[top][rest] = hashlib.sha256(p.read_bytes()).hexdigest() if strict else p.stat().st_size
        return out
    if strict:
        cmd = f"cd {sub} && find . -type f ! -name .DS_Store -exec sha256sum {{}} + | sed 's#^\\([0-9a-f]*\\)  \\./\\(.*\\)$#\\2\\t\\1#'"
    else:
        cmd = f"cd {sub} && find . -type f ! -name .DS_Store -printf '%P\\t%s\\n'"
    return manifest(run_ssh(cmd), strict)


def list_finished(min_age, ld):
    """run folders (flat under runs/) that have a final_report.json and were not modified for min_age minutes"""
    if ld:
        now = time.time()
        return {p.parent.name for p in (ld / "runs").rglob("final_report.json") if now - p.stat().st_mtime > min_age * 60}
    out = run_ssh(f"cd runs && find . -mindepth 2 -maxdepth 2 -name final_report.json -mmin +{min_age} -printf '%h\\n'")
    return {x[2:] for x in out.split()}


# ---------------------------------------------------------------------------------------------- backup (local)
def backup_plan(args):
    PLAN.mkdir(exist_ok=True)
    idx = defaultdict(list)                                    # run_id -> local dirs (any depth)
    for p in (ROOT / "runs").rglob("*"):
        if p.is_dir() and RUN_ID.match(p.name): idx[p.name].append(p)
    safe_runs, keep_runs = [], []
    for d in sorted((BACKUP / "runs").rglob("*")):
        if not (d.is_dir() and RUN_ID.match(d.name)): continue
        a = local_files(d, args.strict)
        ok = any(local_files(x, args.strict) == a for x in idx.get(d.name, []))
        (safe_runs if ok else keep_runs).append(str(d.relative_to(BACKUP)))
    safe_b, keep_b = [], []
    for d in sorted((BACKUP / "benchmark_runs").iterdir()):
        if not d.is_dir(): continue
        l = ROOT / "benchmark_runs" / d.name
        ok = l.is_dir() and local_files(d, args.strict) == local_files(l, args.strict)
        (safe_b if ok else keep_b).append(str(d.relative_to(BACKUP)))
    (PLAN / "backup_delete.txt").write_text("\n".join(safe_runs + safe_b) + "\n")
    (PLAN / "backup_keep.txt").write_text("\n".join(keep_runs + keep_b) + "\n")
    print(f"backup: {len(safe_runs)} run folders + {len(safe_b)} benchmark_runs folders identical to the local copy -> {PLAN/'backup_delete.txt'}")
    print(f"backup: {len(keep_runs)} run folders + {len(keep_b)} benchmark_runs folders NOT in the local copy (new or different) are kept -> {PLAN/'backup_keep.txt'}")
    print("  keep: sync them from the server / merge them into IaCGOD first, then re-run backup-plan")


def backup_delete(args):
    todo = [l for l in (PLAN / "backup_delete.txt").read_text().splitlines() if l]
    bad = [t for t in todo if not SAFE_PATH.match(t) or ".." in t or not (t.startswith("runs/") or t.startswith("benchmark_runs/"))]
    assert not bad, bad[:5]
    print(f"about to delete {len(todo)} folders under {BACKUP} (cannot be undone)")
    if input("type DELETE to continue: ") != "DELETE": sys.exit("aborted")
    for t in todo: shutil.rmtree(BACKUP / t, ignore_errors=True)
    # remove now-empty organiser folders
    for d in sorted((BACKUP / "runs").rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()): d.rmdir()
    print("done")


# ---------------------------------------------------------------------------------------------- server
def server_plan(args):
    PLAN.mkdir(exist_ok=True)
    ld = Path(args.remote[6:]) if args.remote.startswith("local:") else None
    done, keep = finished_dirs(args.include_unfulfilled)
    # ---- benchmark_runs
    print("[server] listing benchmark_runs ...")
    srv = list_manifest("benchmark_runs", args.strict, ld)
    safe_b, why = [], []
    for top, files in sorted(srv.items()):
        if top in keep: why.append(f"keep  {top}: experiment not fulfilled yet (use --include-unfulfilled to override)"); continue
        l = ROOT / "benchmark_runs" / top
        if not l.is_dir(): why.append(f"keep  {top}: no local copy"); continue
        loc = local_files(l, args.strict)
        miss = [f for f in files if f not in loc or loc[f] != files[f]]
        if miss: why.append(f"keep  {top}: {len(miss)} file(s) missing or different locally, e.g. {miss[0]}"); continue
        safe_b.append(f"benchmark_runs/{top}"); why.append(f"SAFE  {top}")
    # ---- runs (finished, idle, identical to a local copy)
    print("[server] listing runs ...")
    finished = list_finished(args.min_age, ld)
    srv_runs = list_manifest("runs", args.strict, ld)
    idx = defaultdict(list)
    for p in (ROOT / "runs").rglob("*"):
        if p.is_dir() and RUN_ID.match(p.name): idx[p.name].append(p)
    safe_r, skipped = [], defaultdict(int)
    for rid, files in sorted(srv_runs.items()):
        if not RUN_ID.match(rid): skipped["not a run folder"] += 1; continue
        if rid not in finished: skipped["unfinished or modified in the last %d min" % args.min_age] += 1; continue
        if not idx.get(rid): skipped["no local copy"] += 1; continue
        if not any(covers(local_files(x, args.strict), files) for x in idx[rid]):
            skipped["local copy differs / incomplete"] += 1; continue
        safe_r.append(f"runs/{rid}")
    (PLAN / "server_delete.txt").write_text("\n".join(safe_b + safe_r) + "\n")
    (PLAN / "server_benchmark_runs_report.txt").write_text("\n".join(why) + "\n")
    print("\n".join(why))
    print(f"\n[server] {len(safe_b)} benchmark_runs folders and {len(safe_r)} runs folders can be deleted -> {PLAN/'server_delete.txt'}")
    for k, v in skipped.items(): print(f"[server] runs kept: {v} {k}")


def server_delete(args):
    todo = [l for l in (PLAN / "server_delete.txt").read_text().splitlines() if l]
    bad = [t for t in todo if not SAFE_PATH.match(t) or ".." in t or not (t.startswith("runs/") or t.startswith("benchmark_runs/"))]
    assert not bad, bad[:5]
    ld = Path(args.remote[6:]) if args.remote.startswith("local:") else None
    print(f"about to delete {len(todo)} folders under {REMOTE}:{REMOTE_ROOT} (cannot be undone). First 5: {todo[:5]}")
    if input("type DELETE to continue: ") != "DELETE": sys.exit("aborted")
    script = "while IFS= read -r p; do rm -rf -- \"$p\"; done"
    if ld:
        subprocess.run(["bash", "-c", script], input="\n".join(todo) + "\n", text=True, cwd=ld, check=True)
    else:
        ctl = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/ssh_cleanup_%r@%h:%p", "-o", "ControlPersist=20m"]
        subprocess.run(["ssh", *ctl, REMOTE, f"cd {REMOTE_ROOT} && {script}"], input="\n".join(todo) + "\n", text=True, check=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backup-plan", "backup-delete", "server-plan", "server-delete"])
    ap.add_argument("--strict", action="store_true", help="compare sha256 of every file instead of name+size (slower)")
    ap.add_argument("--min-age", type=int, default=120, help="server runs: minutes without modification before a finished run counts as idle")
    ap.add_argument("--include-unfulfilled", action="store_true", help="also delete benchmark_runs folders of experiments that are not fulfilled yet")
    ap.add_argument("--remote", default="ssh", help="'ssh' (default) or 'local:/dir' to test against a local directory standing in for the server")
    a = ap.parse_args()
    {"backup-plan": backup_plan, "backup-delete": backup_delete, "server-plan": server_plan, "server-delete": server_delete}[a.cmd](a)
