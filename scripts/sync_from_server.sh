#!/usr/bin/env bash
# Sync benchmark_runs/ and runs/ from the experiments server without clobbering local work.
#   scripts/sync_from_server.sh            # dry run (default): only lists what would change
#   scripts/sync_from_server.sh --go       # really transfer
#   scripts/sync_from_server.sh --go --finalize   # transfer, then run scripts/finalize_results.py --apply
#     (merge attempts, fulfilment check, group run folders -> final_results/; see that script's docstring)
# Run from anywhere; paths are resolved relative to this script. Asks for the SSH password itself.
set -euo pipefail

REMOTE=tianyi@100.117.64.38
REMOTE_ROOT=/home/tianyi/iac-research/iac-god
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$LOCAL_ROOT/.sync_backup_$STAMP"      # local files that a transfer would overwrite are moved here
EXCLUDES="$(mktemp -t runs_exclude.XXXXXX)"

DRY=(-n)
FINALIZE=0
for a in "$@"; do
  [[ "$a" == "--go" ]] && DRY=()
  [[ "$a" == "--finalize" ]] && FINALIZE=1
done

# 1) runs/: every run_id that is already COMPLETE locally (has final_report.json) at ANY depth,
#    including the organiser subfolders made by move_run_folders_from_csv, is excluded.
#    Without this, rsync re-downloads archived runs as flat duplicates under runs/.
find "$LOCAL_ROOT/runs" -name final_report.json -print0 \
  | xargs -0 -n1 dirname | xargs -n1 basename | sort -u | sed 's#^#/#; s#$#/#' > "$EXCLUDES"
echo "[sync] $(wc -l < "$EXCLUDES") complete local run folders excluded from the runs/ transfer"
# runs that exist locally WITHOUT final_report.json (interrupted at the last sync) are not excluded, so they get completed.

COMMON=(-avzP --itemize-changes --update --backup "--backup-dir=$BACKUP_DIR" -e "ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh_cleanup_%r@%h:%p -o ControlPersist=20m")

echo "[sync] benchmark_runs/  (checksum compare; newer local files are kept, older ones are backed up then replaced)"
rsync "${COMMON[@]}" -c ${DRY[@]+"${DRY[@]}"} \
  --exclude='.DS_Store' \
  "$REMOTE:$REMOTE_ROOT/benchmark_runs/" "$LOCAL_ROOT/benchmark_runs/"

echo "[sync] runs/"
rsync "${COMMON[@]}" ${DRY[@]+"${DRY[@]}"} --exclude-from="$EXCLUDES" --exclude='.DS_Store' \
  "$REMOTE:$REMOTE_ROOT/runs/" "$LOCAL_ROOT/runs/"

rm -f "$EXCLUDES"
[[ ${#DRY[@]} -gt 0 ]] && echo "[sync] DRY RUN only. Re-run with --go to transfer." || echo "[sync] done. Overwritten local files (if any) are in $BACKUP_DIR"

if [[ ${#DRY[@]} -eq 0 && $FINALIZE -eq 1 ]]; then
  python3 "$LOCAL_ROOT/scripts/finalize_results.py" --apply
fi
