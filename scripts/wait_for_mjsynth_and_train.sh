#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 || "$1" != <-> ]]; then
  print -u2 "Использование: $0 TAR_PID"
  exit 2
fi

TAR_PID="$1"
PROJECT_ROOT="${0:A:h:h}"

print "Ожидание завершения распаковки MJSynth (PID $TAR_PID)..."
while kill -0 "$TAR_PID" 2>/dev/null; do
  sleep 30
done

print "Распаковка завершена; подготовка manifests и запуск обучения."
exec "$PROJECT_ROOT/scripts/train_mjsynth_local.sh"
