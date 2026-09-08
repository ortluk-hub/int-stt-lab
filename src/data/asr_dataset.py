#!/usr/bin/env python3
"""
ASR Dataset class for PyTorch DataLoader.
Loads precomputed log-mel features when available; otherwise computes them
on the fly from audio when clips_dir is provided. Normalizes, quantizes to
int8, and tokenizes transcripts.

Usage:
    from src.data.asr_dataset import ASRDataset
    from torch.utils.data import DataLoader
    
    dataset = ASRDataset(
        manifest_path="data/manifests/train.jsonl",
        feature_dir="data/features/train",
        tokenizer_path="data/tokenizer/tokenizer.json",
        stats_path="data/features/global_stats.json",
        clips_dir="/path/to/cv-corpus/en",  # optional: audio fallback root
    )
    loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate, num_workers=2)
"""

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer


def _load_log_mel_fn():
    return _load_ef_module().compute_log_mel_librosa


def _load_ef_module():
    spec = importlib.util.spec_from_file_location(
        "extract_features", Path(__file__).parent / "extract_features.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ASRDataset(torch.utils.data.Dataset):
    """ASR Dataset with on-the-fly feature loading, normalization, and int8 quantization."""
    
    def __init__(
        self,
        manifest_path: str | Path,
        feature_dir: str | Path,
        tokenizer_path: str | Path,
        stats_path: str | Path,
        clips_dir: str | Path | None = None,
        max_duration: float = 30.0,  # seconds
        min_duration: float = 0.5,   # seconds
        noise_dir: str | Path | None = None,   # e.g. data/musan/noise (free-sound wavs)
        noise_prob: float = 0.0,               # probability of mixing noise into a sample
        snr_db_range: tuple[float, float] = (0.0, 20.0),
    ):
        self.manifest_path = Path(manifest_path)
        self.feature_dir = Path(feature_dir)
        self.clips_dir = Path(clips_dir) if clips_dir else None
        self.noise_files = sorted(Path(noise_dir).rglob("*.wav")) if noise_dir else []
        self.noise_prob = noise_prob
        self.snr_db_range = snr_db_range
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Load global stats for normalization
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std = np.array(stats['std'], dtype=np.float32)
        self.n_mels = stats['n_mels']
        
        # Load manifest entries
        self.entries = []
        print(f"Loading manifest from {manifest_path}...")
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = entry.get('duration', 0.0)
                if min_duration <= duration <= max_duration:
                    self.entries.append(entry)
        
        print(f"  Loaded {len(self.entries)} entries (filtered by duration {min_duration}-{max_duration}s)")
        
        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.blank_id = self.tokenizer.token_to_id("[BLANK]")
        
        # Quantization parameters (will be computed per-sample)
        self.quant_scale = 127.0  # For int8 range [-127, 127]
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def _features_from_audio(self, audio_path: Path):
        """Load waveform, optionally mix MUSAN noise at a random SNR, compute log-mel."""
        ef = _load_ef_module()
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if self.noise_files and random.random() < self.noise_prob:
            noise, _ = librosa.load(random.choice(self.noise_files), sr=16000, mono=True)
            if len(noise) < len(y):
                reps = int(np.ceil(len(y) / max(len(noise), 1)))
                noise = np.tile(noise, reps)
            start = random.randint(0, len(noise) - len(y))
            noise = noise[start:start + len(y)]
            snr_db = random.uniform(*self.snr_db_range)
            p_sig = float(np.mean(y ** 2)) + 1e-12
            p_noise = float(np.mean(noise ** 2)) + 1e-12
            noise *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
            y = y + noise
        return ef.compute_log_mel(y, sr)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        
        # Load features
        feature_rel = entry.get('feature_path', '')
        if not feature_rel:
            # Fallback: construct from audio_path
            audio_rel = entry.get('audio_path', '')
            stem = Path(audio_rel).stem
            feature_rel = f"{self.feature_dir.name}/{stem}.npy"
        
        feature_path = self.feature_dir / feature_rel if not Path(feature_rel).is_absolute() else Path(feature_rel)
        
        if not feature_path.exists():
            # Try alternative naming
            stem = Path(entry.get('audio_path', '')).stem
            feature_path = self.feature_dir / f"{stem}.npy"
        
        needs_audio = (not feature_path.exists()
                       or (self.noise_files and random.random() < self.noise_prob))
        if needs_audio:
            if self.clips_dir is None:
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
            features, _ = self._features_from_audio(self.clips_dir / entry.get('audio_path', ''))
        else:
            features = np.load(feature_path)  # (T, n_mels) float32
        
        # Normalize
        features = (features - self.mean) / self.std  # (T, n_mels)
        
        # Quantize to int8 with exponent tracking
        # We use symmetric quantization: scale = max_abs / 127
        max_abs = np.max(np.abs(features))
        if max_abs > 0:
            scale = max_abs / self.quant_scale
            features_int8 = np.clip(np.round(features / scale), -128, 127).astype(np.int8)
            exp = int(np.ceil(np.log2(max_abs))) - 7  # Exponent for dequantization
        else:
            features_int8 = np.zeros_like(features, dtype=np.int8)
            scale = 1.0
            exp = 0
        
        # Tokenize transcript
        transcript = entry.get('transcript', '')
        encoded = self.tokenizer.encode(transcript)
        token_ids = encoded.ids  # Includes BOS/EOS
        
        return {
            'features': features_int8,           # (T, 80) int8
            'feature_exp': exp,                  # int exponent
            'feature_scale': scale,              # float scale
            'feature_len': features.shape[0],    # int
            'tokens': np.array(token_ids, dtype=np.int32),  # (U,) int32
            'token_len': len(token_ids),         # int
            'transcript': transcript,            # str (for debugging)
            'speaker_id': entry.get('speaker_id', ''),  # str
            'audio_path': entry.get('audio_path', ''),    # str
        }
    
    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader: pad sequences to max length in batch."""
        # Find max lengths
        max_feat_len = max(item['feature_len'] for item in batch)
        max_token_len = max(item['token_len'] for item in batch)
        batch_size = len(batch)
        
        # Pad features: (B, T_max, 80)
        features_padded = torch.full((batch_size, max_feat_len, self.n_mels), 0, dtype=torch.int8)
        feature_lens = torch.zeros(batch_size, dtype=torch.long)
        feature_exps = torch.zeros(batch_size, dtype=torch.long)
        feature_scales = torch.zeros(batch_size, dtype=torch.float32)
        
        # Pad tokens: (B, U_max)
        tokens_padded = torch.full((batch_size, max_token_len), self.pad_id, dtype=torch.long)
        token_lens = torch.zeros(batch_size, dtype=torch.long)
        
        # Metadata
        transcripts = []
        speaker_ids = []
        audio_paths = []
        
        for i, item in enumerate(batch):
            feat = item['features']  # (T, 80)
            T = item['feature_len']
            U = item['token_len']
            
            features_padded[i, :T] = torch.from_numpy(feat)
            feature_lens[i] = T
            feature_exps[i] = item['feature_exp']
            feature_scales[i] = float(item['feature_scale'])
            
            tokens = item['tokens']
            tokens_padded[i, :U] = torch.from_numpy(tokens)
            token_lens[i] = U
            
            transcripts.append(item['transcript'])
            speaker_ids.append(item['speaker_id'])
            audio_paths.append(item['audio_path'])
        
        return {
            'features': features_padded,           # (B, T_max, 80) int8
            'feature_lens': feature_lens,          # (B,) long
            'feature_exps': feature_exps,          # (B,) long
            'feature_scales': feature_scales,      # (B,) float
            'tokens': tokens_padded,               # (B, U_max) long
            'token_lens': token_lens,              # (B,) long
            'transcripts': transcripts,            # list[str]
            'speaker_ids': speaker_ids,            # list[str]
            'audio_paths': audio_paths,            # list[str]
        }
    
    @staticmethod
    def dequantize(features_int8: torch.Tensor, feature_exps: torch.Tensor, feature_scales: torch.Tensor) -> torch.Tensor:
        """Convert int8 features back to float for debugging/verification."""
        # features_int8: (B, T, 80) or (T, 80)
        # feature_exps: (B,) or scalar
        # feature_scales: (B,) or scalar
        return features_int8.float() * feature_scales.view(-1, 1, 1)


def create_dataloader(
    manifest_path: str | Path,
    feature_dir: str | Path,
    tokenizer_path: str | Path,
    stats_path: str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """Convenience function to create a DataLoader with the ASRDataset."""
    dataset = ASRDataset(manifest_path, feature_dir, tokenizer_path, stats_path, **dataset_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate,
        drop_last=shuffle,  # Drop last incomplete batch during training
    )
    return loader


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    dataset = ASRDataset(
        manifest_path="data/manifests/dev.jsonl",
        feature_dir="data/features/dev",
        tokenizer_path="data/tokenizer/tokenizer.json",
        stats_path="data/features/global_stats.json",  # Will fail if not exist
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single item
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    print(f"Features shape: {item['features'].shape}")
    print(f"Feature exp: {item['feature_exp']}")
    print(f"Feature scale: {item['feature_scale']:.4f}")
    print(f"Tokens: {item['tokens']}")
    print(f"Token len: {item['token_len']}")
    print(f"Transcript: {item['transcript'][:50]}...")