#!/usr/bin/env bash
# vcfbot daily refresh — runs at 04:00 via cron on apollo.
#
# Behavior: invoke `vcfbot update` inside the running container. That command
# does a conditional fetch (no-op if Broadcom hasn't republished) and, only on
# real upstream change, wipes chroma and re-indexes. A changelog entry is
# appended per actual update so the web UI can show recent changes.
#
# Idempotent. Safe to run on demand:
#   sudo /home/anthony/dev/vcfbot/scripts/daily-update.sh
#
set -uo pipefail

CONTAINER="vcfbot"
LOG_DIR=/backup/logs
LOG_RETENTION_DAYS=30
TS_FILE=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/vcfbot-update-$TS_FILE.log"
ERR="$LOG_DIR/vcfbot-update.errors.log"   # tail-friendly cross-run failures

mkdir -p "$LOG_DIR"
touch "$ERR"

ts() { date -Iseconds; }

{
  echo "===== $(ts) vcfbot daily-update start ====="
  if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
    echo "$(ts) ERROR: container '$CONTAINER' is not running" >&2
    exit 2
  fi
  docker exec "$CONTAINER" python -m vcfbot update
  rc=$?
  echo "$(ts) finished rc=$rc"
  exit "$rc"
} >> "$LOG" 2>&1

rc=$?
if [ "$rc" -ne 0 ]; then
  echo "$(ts) vcfbot daily-update FAILED rc=$rc — see $LOG" >> "$ERR"
fi

# Prune per-run logs older than retention window. Matches the homelab
# convention (ufo-files / backup-storage.sh use the same pattern).
find "$LOG_DIR" -name 'vcfbot-update-*.log' -type f -mtime +"$LOG_RETENTION_DAYS" -delete 2>/dev/null || true

exit "$rc"
