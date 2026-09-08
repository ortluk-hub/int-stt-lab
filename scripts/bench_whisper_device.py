#!/usr/bin/env python3
"""
On-device FP Whisper CPU baseline on a fixed Common Voice test slice.

Prepares N wavs from the test manifest, pushes them to the phone, runs
whisper-cli per file, and reports per-run + aggregate WER/CER/RTF.

Usage: ./venv/bin/python scripts/bench_whisper_device.py [--model ggml-base.bin] [--n 64]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import editdistance
import librosa
import numpy as np
import soundfile as sf

CLIPS = Path("/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en")
DEVICE_DIR = "/data/local/tmp/whisper/bench"
ROOT = Path(__file__).parent.parent
LOCAL_WAV = ROOT / "bench" / "cv_test_wavs"
LOCAL_REF = ROOT / "bench" / "cv_test_refs.json"


def adb(*args, capture=True):
    return subprocess.run(["adb", *args], capture_output=capture, text=True, check=True)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def mix_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / max(len(noise), 1))))
    noise = noise[: len(clean)]
    p_sig = float(np.mean(clean ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    return clean + noise * float(np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10))))


MUSAN_NOISE = ROOT / "data" / "musan" / "noise"


def prepare(n: int, max_dur: float, snr_db: float | None = None):
    wav_dir = LOCAL_WAV if snr_db is None else ROOT / "bench" / f"cv_test_wavs_snr{snr_db:g}"
    wav_dir.mkdir(parents=True, exist_ok=True)
    noise_files = sorted(MUSAN_NOISE.rglob("*.wav")) if snr_db is not None else []
    assert snr_db is None or noise_files, f"no MUSAN noise wavs under {MUSAN_NOISE}"
    refs = []
    with open(ROOT / "data/manifests/test.jsonl") as f:
        lines = f.readlines()
    for line in lines[: n * 3]:  # oversample, some may be missing
        if len(refs) >= n:
            break
        e = json.loads(line)
        if not (1.0 <= e.get("duration", 0) <= max_dur):
            continue
        src = CLIPS / e["audio_path"]
        wav = wav_dir / (Path(e["audio_path"]).stem + ".wav")
        if not wav.exists():
            if not src.exists():
                continue
            y, _ = librosa.load(src, sr=16000, mono=True)
            if snr_db is not None:
                nz, _ = librosa.load(noise_files[len(refs) % len(noise_files)], sr=16000, mono=True)
                y = np.clip(mix_snr(y, nz, snr_db), -1.0, 1.0)
            sf.write(wav, y, 16000)
        refs.append({"wav": wav.name, "transcript": norm(e["transcript"]),
                     "duration": e["duration"]})
    return refs


def push(refs, wav_dir: Path):
    adb("shell", f"mkdir -p {DEVICE_DIR}")
    for r in refs:
        adb("push", str(wav_dir / r["wav"]), f"{DEVICE_DIR}/{r['wav']}")


def run_device(model: str, threads: int, refs: list) -> list:
    results = []
    for r in refs:
        t0 = time.time()
        out = adb("shell", f"cd /data/local/tmp/whisper && LD_LIBRARY_PATH=. ./whisper-cli "
                           f"-m {model} -f bench/{r['wav']} -t {threads} -nt")
        wall = time.time() - t0
        hyp_lines = [l.strip() for l in out.stdout.splitlines()
                     if l.strip() and not l.startswith(("whisper_", "main:", "system_info", "[", "\\"))]
        hyp = norm(" ".join(hyp_lines))
        m = re.search(r"total time =\s+([\d.]+) ms", out.stderr)
        results.append({"wav": r["wav"], "hyp": hyp, "ref": r["transcript"],
                        "duration": r["duration"], "wall_s": wall,
                        "infer_ms": float(m.group(1)) if m else None})
        print(f"  {r['wav']}: infer {results[-1]['infer_ms']} ms / wall {wall:.1f}s "
              f"| {hyp[:60]!r}", flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ggml-base.bin")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-dur", type=float, default=15.0)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--snr", type=float, default=None, help="mix MUSAN noise at this SNR dB")
    a = p.parse_args()
    if a.skip_prepare:
        refs = json.load(open(LOCAL_REF))
        wav_dir = LOCAL_WAV
    else:
        refs = prepare(a.n, a.max_dur, a.snr)
        wav_dir = LOCAL_WAV if a.snr is None else ROOT / "bench" / f"cv_test_wavs_snr{a.snr:g}"
    label = f"{a.model} snr{a.snr:g}" if a.snr is not None else a.model
    print(f"slice: {len(refs)} utterances, {sum(r['duration'] for r in refs):.0f}s audio | {label}")
    if not a.skip_prepare:
        push(refs, wav_dir)
    results = run_device(a.model, a.threads, refs)
    w_err = w_tot = c_err = c_tot = 0
    audio_s = infer_ms = 0.0
    for r in results:
        hw, rw = r["hyp"].split(), r["ref"].split()
        w_err += editdistance.eval(hw, rw); w_tot += len(rw)
        c_err += editdistance.eval(r["hyp"], r["ref"]); c_tot += max(len(r["ref"]), 1)
        audio_s += r["duration"]
        if r["infer_ms"]:
            infer_ms += r["infer_ms"]
    wer, cer = w_err / max(w_tot, 1), c_err / max(c_tot, 1)
    rtf = (infer_ms / 1000) / audio_s if infer_ms else None
    summary = {"model": a.model,
               "snr": a.snr if a.snr is not None else "clean",
               "n": len(results), "wer": round(wer, 4), "cer": round(cer, 4),
               "rtf_internal": round(rtf, 4) if rtf else None,
               "rtf_wall_avg": round(sum(r["wall_s"] for r in results) / audio_s, 4),
               "audio_s": round(audio_s, 1),
               "infer_s": round(infer_ms / 1000, 1) if infer_ms else None}
    print("\n=== BASELINE:", json.dumps(summary))
    out = ROOT / "bench" / (f"baseline_{Path(a.model).stem}_snr{a.snr:g}.json"
                            if a.snr is not None else f"baseline_{Path(a.model).stem}.json")
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1)
    print(f"saved {out}")
