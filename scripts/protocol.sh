#!/usr/bin/env bash
# The full filming protocol: clips in, a fast pair first, then the untouched repeat, calibration and the real pair,
# one heavy step at a time. For just two videos, `whatchanged all BEFORE AFTER` is enough.
#
#   scripts/protocol.sh A=IMG_0102.MOV A2=IMG_0103.MOV B=IMG_0107.MOV     ingest (names in ~/Downloads), then all
#   scripts/protocol.sh --times "pilot=09:05 A=09:20 A2=09:33 B=10:08"   same, clips matched by clock time
#   scripts/protocol.sh                                                   data/raw/{A,A2,B}.mov already there
#   scripts/protocol.sh --from full       resume at a stage (ingest, fast, full, report); finished work is reused
#   scripts/protocol.sh --only fast       just one stage
#   scripts/protocol.sh --dry-run         print the commands, run nothing
#
# Knobs (environment): SIDE_MM=150 FAST_S=90 FAST_FPS=2 FAST_ITERS=7000 FPS=2 ITERS=30000 REPEAT_ITERS=2000
#   FRAMES=auto (0.4 keyframes per second of the shortest of A, A2, B, from 24 to 64; the same for both runs, so the
#   frozen thresholds fit the real pair) GT= (an answer key to score the real pair against); RAW=data/raw OUT=out CONFIG=config.frozen.json;
#   FAST_LABELS=auto REPEAT_LABELS=auto (siglip: no Claude calls for that run, so Claude names only the real pair);
#   SKIP_CALIBRATE=1 runs A vs B on the default thresholds when the untouched pair can't be calibrated
# Every step is logged to $OUT/protocol/NN-step.log, with its time in $OUT/protocol/timings.tsv. A failed
# step prints STOP and nothing after it runs. Other arguments (e.g. --allow-hdr) pass through to ingest.py.

set -euo pipefail
cd "$(dirname "$0")/.."

SIDE_MM=${SIDE_MM:-150}; FAST_S=${FAST_S:-90}; FAST_FPS=${FAST_FPS:-2}; FAST_ITERS=${FAST_ITERS:-7000}
FPS=${FPS:-2}; ITERS=${ITERS:-30000}; REPEAT_ITERS=${REPEAT_ITERS:-2000}; FRAMES=${FRAMES:-auto}
GT=${GT:-}; SKIP_CALIBRATE=${SKIP_CALIBRATE:-}
FAST_LABELS=${FAST_LABELS:-auto}; REPEAT_LABELS=${REPEAT_LABELS:-auto}
RAW=${RAW:-data/raw}; OUT=${OUT:-out}; CONFIG=${CONFIG:-config.frozen.json}
PY=.venv/bin/python
WC=.venv/bin/whatchanged
LOGDIR=$OUT/protocol
STAGES="ingest fast full report"
# Heavy jobs: COLMAP, the CLI, splat training, MapAnything. Narrow on purpose: an http.server serving
# out/ also has "whatchanged" in its command line and can be left running.
HEAVY='(^|/)colmap |bin/whatchanged |-m whatchanged|msplat|mapanything|dense_depth'

FROM=ingest; ONLY=""; DRY=0; INGEST=()
while [ $# -gt 0 ]; do
  case $1 in
    --from) FROM=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) INGEST+=("$1"); shift ;;
  esac
done
for s in "$FROM" ${ONLY:+"$ONLY"}; do
  case " $STAGES " in *" $s "*) ;; *) echo "unknown stage $s (use: $STAGES)"; exit 2 ;; esac
done

mkdir -p "$LOGDIR"
N=$(ls "$LOGDIR" | grep -c '\.log$' || true)
TIMES=$LOGDIR/timings.tsv
[ -f "$TIMES" ] || printf 'step\tstarted\tseconds\texit\n' > "$TIMES"

say() { echo "[$(date +%H:%M:%S)] $*"; }

want() {  # should stage $1 run?
  if [ -n "$ONLY" ]; then [ "$ONLY" = "$1" ]; return; fi
  local s on=0
  for s in $STAGES; do
    [ "$s" = "$FROM" ] && on=1
    [ "$s" = "$1" ] && return $((1 - on))
  done
  return 1
}

heavy_pids() {  # other heavy jobs now running. A shell, grep or tail whose command line merely mentions one
  # (a terminal's `zsh -c "... whatchanged run ..."`, a log watcher) is not one: the job it starts is listed itself.
  local p exe
  for p in $(pgrep -f "$HEAVY" || true); do
    [ "$p" = "$$" ] && continue
    exe=$(ps -o command= -p "$p" 2>/dev/null) || continue
    exe=${exe%% *}
    case "${exe##*/}" in
      ""|sh|bash|zsh|dash|-sh|-bash|-zsh|grep|egrep|tail|less|caffeinate) ;;
      *) echo "$p" ;;
    esac
  done
}

wait_idle() {  # never two heavy jobs at once: wait for any other COLMAP, splat or MapAnything job to finish
  [ $DRY = 1 ] && return 0
  local n=0 pids
  while pids=$(heavy_pids); [ -n "$pids" ]; do
    if [ $((n % 8)) = 0 ]; then
      say "waiting: another heavy job is running"
      ps -o pid=,etime=,command= -p "$(echo $pids | tr ' ' ',')" | cut -c1-160 || true
    fi
    n=$((n + 1)); sleep 15
  done
}

step() {  # step NAME CMD...: run, log, time; STOP on failure
  local name=$1; shift
  local log; log=$LOGDIR/$(printf %02d "$N")-$name.log; N=$((N + 1))
  say ">>> $name: $*"
  [ $DRY = 1 ] && return 0
  local t0 rc; t0=$(date +%s)
  set +e
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  local s=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\n' "$name" "$(date -r "$t0" +%H:%M:%S)" "$s" "$rc" >> "$TIMES"
  if [ "$rc" != 0 ]; then
    echo; echo "STOP: $name failed (exit $rc) after $((s / 60))m$((s % 60))s. Log: $log"
    [ -n "${HINT:-}" ] && echo "      $HINT"
    exit "$rc"
  fi
  say "<<< $name done in $((s / 60))m$((s % 60))s"
}

heavy() { wait_idle; step "$@"; }

fresh() {  # fresh DIR STAMP CLIP...: refuse to reuse a folder built from older clips
  local dir=$1 stamp=$1/$2; shift 2
  [ -e "$stamp" ] || return 0
  local c
  for c in "$@"; do
    if [ "$c" -nt "$stamp" ]; then
      echo "STOP: $dir was built before $c was ingested, so it would reuse the old frames and poses."
      echo "      Move it aside and run again:  mv $dir $dir.old-$(date +%H%M)"
      exit 3
    fi
  done
}

MPU_OK=""   # the marker scale goes in as --metres-per-unit, once both `run` and `room` take it
case "$("$WC" run --help 2>/dev/null || true) $("$WC" room --help 2>/dev/null || true)" in
  *--metres-per-unit*--metres-per-unit*) MPU_OK=1 ;;
esac

marker() {  # marker DIR: print the marker's metres per COLMAP unit, or nothing (never fatal)
  local dir=$1 log; log=$LOGDIR/$(printf %02d "$N")-marker-$(basename "$1").log; N=$((N + 1))
  if [ $DRY = 1 ]; then say ">>> marker: $PY scripts/marker_scale.py $dir --side-mm $SIDE_MM" >&2; return 0; fi
  if "$PY" scripts/marker_scale.py "$dir" --side-mm "$SIDE_MM" --json "$dir/marker_scale.json" > "$log" 2>&1; then
    say "marker: $(grep '^Scale:' "$log")" >&2
    sed -n 's/^METRES_PER_UNIT=//p' "$log" | tail -1
  else
    say "marker: no usable scale for $dir ($(tail -1 "$log")); the scale stays MapAnything's" >&2
  fi
}

scale_flag() {  # the marker scale as a CLI flag
  if [ -n "$1" ] && [ -n "$MPU_OK" ]; then echo "--metres-per-unit $1"; fi
}

# ---------------------------------------------------------------- preflight
say "protocol.sh: stages from '$FROM'${ONLY:+ (only $ONLY)}; logs in $LOGDIR"
for t in ffmpeg ffprobe colmap "$PY" "$WC"; do command -v "$t" >/dev/null || { echo "STOP: $t not found"; exit 2; }; done
free=$(df -g . | awk 'NR==2 {print $4}')
if [ "$free" -lt 15 ]; then say "warning: only ${free} GB free on disk; the full runs want about 15 GB"; fi
[ -n "$MPU_OK" ] || say "note: whatchanged has no --metres-per-unit, so the marker scale is logged but not used"
if command -v caffeinate >/dev/null && [ $DRY = 0 ]; then caffeinate -dims -w $$ & fi   # no sleeping mid-run
say "code: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' + uncommitted changes')"

# ---------------------------------------------------------------- ingest
if want ingest; then
  if [ ${#INGEST[@]} -gt 0 ]; then
    step ingest "$PY" scripts/ingest.py "${INGEST[@]}" --out "$RAW"
  elif [ -f "$RAW/A.mov" ] && [ -f "$RAW/B.mov" ]; then
    say "ingest: using $RAW/A.mov and B.mov$([ -f "$RAW/A2.mov" ] && echo ' (and A2.mov)') as they are"
  else
    echo "STOP: no clips. Pass them (A=IMG_....MOV A2=... B=...) or --times \"A=09:20 A2=09:33 B=10:08\""; exit 2
  fi
fi
for c in A B; do [ -f "$RAW/$c.mov" ] || [ $DRY = 1 ] || { echo "STOP: $RAW/$c.mov missing"; exit 2; }; done
A=$RAW/A.mov; A2=$RAW/A2.mov; B=$RAW/B.mov; A90=$RAW/A90.mov; B90=$RAW/B90.mov
# A2 (the untouched repeat) is optional: without it there is no zero-change check and no calibration, and
# A vs B runs on the default thresholds (the README says so).
HAS_A2=""; [ -f "$A2" ] && HAS_A2=1
[ -n "$HAS_A2" ] || say "no A2 clip: skipping the untouched-pair check; A vs B uses the default thresholds"
if [ "$FRAMES" = auto ]; then
  # On the trial room 24 keyframes over 42 s (0.6 a second) gave 0 false changes and 62 (1.5 a second) gave 6-10:
  # close frames add votes, not views. 64 over a 3-minute walk is 0.36 a second.
  shortest=$(for c in "$A" ${HAS_A2:+"$A2"} "$B"; do ffprobe -v error -show_entries format=duration -of csv=p=0 "$c" 2>/dev/null || true; done \
             | sort -n | head -1)
  FRAMES=$(awk -v d="${shortest:-0}" 'BEGIN { f = int(d * 0.4 + 0.5); if (f < 24) f = 24; if (f > 64) f = 64; print f }')
  say "keyframes per video: $FRAMES (shortest clip ${shortest:-?} s)"
fi

# ---------------------------------------------------------------- fast pair: the first ~90 s of A and of B
if want fast; then
  for c in A B; do
    if [ $DRY = 1 ] || [ ! -f "$RAW/${c}90.mov" ] || [ "$RAW/$c.mov" -nt "$RAW/${c}90.mov" ]; then
      step "cut-$c" ffmpeg -v error -nostdin -y -i "$RAW/$c.mov" -t "$FAST_S" -map 0:v:0 -c copy "$RAW/${c}90.mov"
    fi
  done
  fresh "$OUT/fast-room" work/frames.json "$A90" "$B90"
  fresh "$OUT/fast" cache/diag.json "$A90" "$B90"
  heavy fast-room "$WC" room "$A90" "$B90" --fps "$FAST_FPS" --iters "$FAST_ITERS" --out "$OUT/fast-room"
  MPU=$(marker "$OUT/fast-room")
  heavy fast-run "$WC" run "$A90" "$B90" --poses colmap --colmap-from "$OUT/fast-room" --out "$OUT/fast" \
    --labels "$FAST_LABELS" $(scale_flag "$MPU")
  heavy fast-viewer "$WC" room "$A90" "$B90" --out "$OUT/fast-room" --detections "$OUT/fast" $(scale_flag "$MPU")
  heavy fast-video "$WC" video "$OUT/fast-room" --detections "$OUT/fast" --out "$OUT/fast-room/whatchanged.mp4"
  say "FIRST RESULT: $OUT/fast/report.html and $OUT/fast-room/index.html"
fi

# ---------------------------------------------------------------- full runs
if want full; then
  fresh "$OUT/pair" work/frames.json "$A" "$B"
  if [ -n "$HAS_A2" ]; then
    fresh "$OUT/repeat-room" work/frames.json "$A" "$A2"
    fresh "$OUT/repeat" cache/diag.json "$A" "$A2"
  fi
  fresh "$OUT/real" cache/diag.json "$A" "$B"
  # A room rebuilt in place (new COLMAP model) needs nothing here: `room` retrains splats older than the
  # model, and `run` redoes depth fitted to an older model.
  # A+B: one COLMAP model and a splat each (the viewer).
  # A folder that already has work/frames.json keeps its frames whatever --fps says, so a retry at
  # another FPS needs the folder moved aside first.
  HINT="If too few frames posed, the videos did not join or the model is broken, retry at 2 fps in a fresh folder:
      mv $OUT/pair $OUT/pair.fps$FPS && FPS=2 scripts/protocol.sh --from full"
  heavy pair-room "$WC" room "$A" "$B" --fps "$FPS" --iters "$ITERS" --out "$OUT/pair"
  MPU_PAIR=$(marker "$OUT/pair")
  HINT=""
  CFG=(--config "$CONFIG")
  if [ -n "$HAS_A2" ]; then
    # A+A2: a second joint model (nothing registers A2 into A's model); its splats are only a by-product.
    HINT="If too few frames posed, the videos did not join or the model is broken, retry at 2 fps in a fresh folder:
      mv $OUT/repeat-room $OUT/repeat-room.fps$FPS && FPS=2 scripts/protocol.sh --from full"
    heavy repeat-room "$WC" room "$A" "$A2" --fps "$FPS" --iters "$REPEAT_ITERS" --out "$OUT/repeat-room"
    HINT=""
    MPU_REPEAT=$(marker "$OUT/repeat-room")
    heavy repeat-run "$WC" run "$A" "$A2" --poses colmap --colmap-from "$OUT/repeat-room" --frames "$FRAMES" \
      --labels "$REPEAT_LABELS" --out "$OUT/repeat" $(scale_flag "$MPU_REPEAT")
  fi
  # Thresholds from the untouched pair, frozen BEFORE the real pair is run.
  if [ -z "$HAS_A2" ]; then
    say "no A2: A vs B runs on the default thresholds"
    CFG=()
  elif [ -n "$SKIP_CALIBRATE" ]; then
    say "SKIP_CALIBRATE: A vs B runs on the default thresholds; say so in the README"
    CFG=()
  else
    HINT="The untouched pair still shows changes. Look at $OUT/repeat/report.html; to get the A vs B result anyway:
      SKIP_CALIBRATE=1 scripts/protocol.sh --from full"
    step calibrate "$WC" calibrate "$OUT/repeat" --write "$CONFIG"
    HINT=""
    [ $DRY = 1 ] || say "frozen: $CONFIG sha256 $(shasum -a 256 "$CONFIG" | cut -c1-16) at $(date +%H:%M:%S); commit it"
    step repeat-report "$WC" report "$OUT/repeat" --config "$CONFIG" --labels "$REPEAT_LABELS"
  fi
  heavy real-run "$WC" run "$A" "$B" --poses colmap --colmap-from "$OUT/pair" --frames "$FRAMES" \
    ${CFG[@]+"${CFG[@]}"} --out "$OUT/real" $(scale_flag "$MPU_PAIR")
  heavy pair-viewer "$WC" room "$A" "$B" --out "$OUT/pair" --detections "$OUT/real" $(scale_flag "$MPU_PAIR")
  heavy pair-video "$WC" video "$OUT/pair" --detections "$OUT/real" --out "$OUT/pair/whatchanged.mp4"
  if [ -n "$GT" ] && [ -f "$GT" ]; then
    step score "$WC" score "$OUT/real" "$GT"
  else
    say "score skipped: pass GT=path/to/ground_truth.yaml to score the real pair"
  fi
fi

# ---------------------------------------------------------------- report
if want report && [ $DRY = 0 ]; then
  echo; say "Results"
  for f in "$OUT/fast/report.html" "$OUT/fast-room/index.html" "$OUT/repeat/report.html" "$OUT/real/report.html" \
           "$OUT/real/changes.json" "$OUT/pair/index.html" "$CONFIG" "$OUT/real/scorecard.md"; do
    if [ -e "$f" ]; then echo "  $f"; fi
  done
  for d in "$OUT/fast" "$OUT/repeat" "$OUT/real"; do
    if [ -f "$d/changes.json" ]; then
      "$PY" -c "import json, sys; c = json.load(open(sys.argv[1]))['changes']; print(f'  {sys.argv[1]}: {len(c)} changes')
for x in c: print('    -', x['type'], x['sentence'])" "$d/changes.json"
    fi
  done
  echo "  (the untouched pair, $OUT/repeat, must show 0 changes)"
  echo "  3D viewer: python3 -m http.server -d $OUT/pair 8000   then open http://localhost:8000"
  echo; echo "Step times (seconds):"; column -t -s $'\t' "$TIMES" | tail -n 25
fi
