#!/bin/zsh
set -uo pipefail

if [[ $# -ne 1 || "$1" != <-> ]]; then
  print -u2 "Использование: $0 CURRENT_TRAINING_PID"
  exit 2
fi

CURRENT_PID="$1"
PROJECT_ROOT="${0:A:h:h}"

print "$(date '+%Y-%m-%d %H:%M:%S') watchdog ожидает PID $CURRENT_PID"
while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 10
done

print "$(date '+%Y-%m-%d %H:%M:%S') PID $CURRENT_PID завершился; включён автоперезапуск"
while true; do
  "$PROJECT_ROOT/scripts/train_mjsynth_local.sh"
  status=$?
  if [[ $status -eq 0 ]]; then
    print "$(date '+%Y-%m-%d %H:%M:%S') обучение завершено успешно"
    exit 0
  fi
  print -u2 "$(date '+%Y-%m-%d %H:%M:%S') обучение завершилось с кодом $status; повтор через 15 секунд"
  sleep 15
done
