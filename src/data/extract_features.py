#!/usr/bin/env python3
"""
Extract log-mel filterbank features from audio files using librosa.
Saves as .npy files (float32) and computes global statistics for normalization.

Usage:
    python extract_features.py --manifest data/manifests/train.jsonl --clips-dir /path/to/clips --out-dir data/features/train --n-mels 80
    python extract_features.py --compute-stats-only --feature-dirs data/features/train data/features/dev data/features/test --out data/features/global_stats.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def compute_log_mel_librosa(audio_path: Path, n_fft: int = 400, hop_length: int = 160, 
                            n_mels: int = 80, f_min: float = 20.0, f_max: float = 7600.0, 
                            target_sr: int = 16000) -> tuple[np.ndarray, float]:
    """
    Compute log-mel filterbank features using librosa.
    Returns: (features (T, n_mels) float32, duration)
    """
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return compute_log_mel(waveform, sr, n_fft, hop_length, n_mels, f_min, f_max, target_sr)


def compute_log_mel(waveform: np.ndarray, sr: int, n_fft: int = 400, hop_length: int = 160,
                    n_mels: int = 80, f_min: float = 20.0, f_max: float = 7600.0,
                    target_sr: int = 16000) -> tuple[np.ndarray, float]:
    """Compute log-mel from an in-memory waveform (mono, already at sr)."""
    duration = len(waveform) / sr
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
        htk=True  # Use HTK formula for mel scale
    )  # (n_mels, T)
    
    # Log compression
    log_mel = np.log(mel_spec + 1e-10)
    
    # Transpose to (T, n_mels)
    log_mel = log_mel.T.astype(np.float32)
    
    return log_mel, duration


def process_manifest(manifest_path: Path, clips_dir: Path, out_dir: Path, 
                     n_mels: int = 80, n_fft: int = 400, hop_length: int = 160,
                     max_samples: int | None = None):
    """Process all entries in a manifest and save features as .npy files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    failed = 0
    total_duration = 0.0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and count >= max_samples:
                break
            
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                failed += 1
                continue
            
            audio_rel_path = entry.get('audio_path', '')
            audio_path = clips_dir / audio_rel_path
            
            if not audio_path.exists():
                failed += 1
                if failed <= 5:
                    print(f"  Missing audio: {audio_path}", file=sys.stderr)
                continue
            
            # Compute features using librosa
            try:
                features, duration = compute_log_mel_librosa(
                    audio_path, n_fft, hop_length, n_mels
                )
                total_duration += duration
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Failed to compute features for {audio_path}: {e}", file=sys.stderr)
                continue
            
            # Save as .npy (named by original audio filename stem)
            stem = audio_path.stem
            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, features)
            
            # Update manifest entry with actual duration
            entry['duration'] = duration
            entry['feature_path'] = str(out_path.relative_to(out_dir.parent))
            
            count += 1
            
            if count % 10000 == 0:
                print(f"  Processed {count} samples...")
    
    print(f"  Completed: {count} succeeded, {failed} failed, {total_duration/3600:.2f} hours audio")
    return count, failed


def compute_global_stats(feature_dirs: list[Path], out_path: Path, n_mels: int = 80, max_files: int = 50000):
    """Compute global mean and std across feature files (sample for speed)."""
    import random
    
    all_files = []
    for d in feature_dirs:
        all_files.extend(list(d.glob("*.npy")))
    
    if not all_files:
        raise ValueError("No feature files found")
    
    # Sample files for statistics
    if len(all_files) > max_files:
        sampled = random.sample(all_files, max_files)
    else:
        sampled = all_files
    
    print(f"Computing stats from {len(sampled)} files...")
    
    sum_feat = np.zeros(n_mels, dtype=np.float64)
    sum_sq_feat = np.zeros(n_mels, dtype=np.float64)
    total_frames = 0
    
    for i, fpath in enumerate(sampled):
        try:
            feat = np.load(fpath)  # (T, n_mels)
            sum_feat += feat.sum(axis=0)
            sum_sq_feat += (feat ** 2).sum(axis=0)
            total_frames += feat.shape[0]
        except Exception:
            continue
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(sampled)} files")
    
    mean = sum_feat / total_frames
    std = np.sqrt(sum_sq_feat / total_frames - mean ** 2)
    std = np.maximum(std, 1e-10)  # Avoid zero std
    
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_mels": n_mels,
        "total_frames": int(total_frames),
        "num_files_used": len(sampled)
    }
    
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats saved to {out_path}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range:  [{std.min():.4f}, {std.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Extract log-mel features from audio using librosa")
    parser.add_argument("--manifest", help="Manifest JSONL file")
    parser.add_argument("--clips-dir", help="Root directory containing audio clips")
    parser.add_argument("--out-dir", help="Output directory for .npy feature files")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins")
    parser.add_argument("--n-fft", type=int, default=400, help="FFT size (25ms at 16kHz)")
    parser.add_argument("--hop-length", type=int, default=160, help="Hop length (10ms at 16kHz)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    
    # Stats-only mode
    parser.add_argument("--compute-stats-only", action="store_true", help="Only compute global stats")
    parser.add_argument("--feature-dirs", nargs="+", help="Feature directories for stats computation")
    parser.add_argument("--out", help="Output stats JSON file")
    
    args = parser.parse_args()
    
    if args.compute_stats_only:
        if not args.feature_dirs or not args.out:
            parser.error("--feature-dirs and --out required for --compute-stats-only")
        compute_global_stats([Path(d) for d in args.feature_dirs], Path(args.out), args.n_mels)
    else:
        if not args.manifest or not args.clips_dir or not args.out_dir:
            parser.error("--manifest, --clips-dir, --out-dir required for feature extraction")
        process_manifest(
            Path(args.manifest), Path(args.clips_dir), Path(args.out_dir),
            args.n_mels, args.n_fft, args.hop_length, args.max_samples
        )


if __name__ == "__main__":
    main()