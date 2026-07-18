# Shubhangi, Upload Date: 2026-07-15
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np


@dataclass(frozen=True)
class AcousticFeatures:
    sample_rate: int
    duration_seconds: float
    rms_mean: float
    zero_crossing_rate: float
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    mfcc_mean: list[float]
    mel_spectrogram_preview: list[float]
    reverb_tail_ratio: float
    rir_decay_slope: float
    background_consistency: float
    breathing_cadence_score: float
    anomaly_curve: list[float]

    def to_public_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "rms_mean": round(self.rms_mean, 6),
            "zero_crossing_rate": round(self.zero_crossing_rate, 6),
            "spectral_centroid_mean": round(self.spectral_centroid_mean, 3),
            "spectral_rolloff_mean": round(self.spectral_rolloff_mean, 3),
            "mfcc_mean": [round(value, 4) for value in self.mfcc_mean],
            "mel_spectrogram_preview": [round(value, 4) for value in self.mel_spectrogram_preview],
            "reverb_tail_ratio": round(self.reverb_tail_ratio, 6),
            "rir_decay_slope": round(self.rir_decay_slope, 6),
            "background_consistency": round(self.background_consistency, 6),
            "breathing_cadence_score": round(self.breathing_cadence_score, 6),
        }


class LibrosaFeatureExtractor:
    def __init__(self, target_sample_rate: int = 16_000) -> None:
        self.target_sample_rate = target_sample_rate

    def extract(self, audio_bytes: bytes) -> AcousticFeatures:
        try:
            waveform, sample_rate = librosa.load(
                BytesIO(audio_bytes),
                sr=self.target_sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise ValueError("Could not decode audio file. Try WAV, MP3, FLAC, OGG, or M4A.") from exc

        waveform = librosa.util.normalize(np.asarray(waveform, dtype=np.float32))
        if waveform.size < sample_rate // 2:
            raise ValueError("Audio is too short for acoustic analysis. Upload at least 0.5 seconds.")

        duration = float(librosa.get_duration(y=waveform, sr=sample_rate))
        frame_length = 1024
        hop_length = 256

        frames = _frame_audio(waveform, frame_length=frame_length, hop_length=hop_length)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
        frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
        spectrum_sum = np.sum(spectrum, axis=1) + 1e-8
        centroid = np.sum(spectrum * frequencies, axis=1) / spectrum_sum
        rolloff = _spectral_rolloff(spectrum=spectrum, frequencies=frequencies)
        band_summary = _spectral_band_summary(spectrum=spectrum, band_count=16)
        mfcc_like = _compact_cepstral_summary(band_summary=band_summary, coefficient_count=13)

        reverb_tail_ratio = self._estimate_reverb_tail_ratio(rms)
        rir_decay_slope = self._estimate_decay_slope(rms)
        background_consistency = self._estimate_background_consistency(waveform, sample_rate)
        breathing_cadence_score = self._estimate_breathing_cadence(waveform, sample_rate)
        anomaly_curve = self._build_anomaly_curve(
            rms=rms,
            zcr=zcr,
            centroid=centroid,
            reverb_tail_ratio=reverb_tail_ratio,
            background_consistency=background_consistency,
        )

        return AcousticFeatures(
            sample_rate=sample_rate,
            duration_seconds=duration,
            rms_mean=float(np.mean(rms)),
            zero_crossing_rate=float(np.mean(zcr)),
            spectral_centroid_mean=float(np.mean(centroid)),
            spectral_rolloff_mean=float(np.mean(rolloff)),
            mfcc_mean=mfcc_like.astype(float).tolist(),
            mel_spectrogram_preview=band_summary.astype(float).tolist(),
            reverb_tail_ratio=float(reverb_tail_ratio),
            rir_decay_slope=float(rir_decay_slope),
            background_consistency=float(background_consistency),
            breathing_cadence_score=float(breathing_cadence_score),
            anomaly_curve=anomaly_curve,
        )

    @staticmethod
    def _estimate_reverb_tail_ratio(rms: np.ndarray) -> float:
        if rms.size < 8:
            return 0.0
        threshold = np.quantile(rms, 0.65)
        energetic = np.where(rms >= threshold)[0]
        if energetic.size == 0:
            return 0.0
        tail_start = min(int(energetic[-1] + 1), rms.size - 1)
        tail_energy = float(np.mean(rms[tail_start:])) if tail_start < rms.size else 0.0
        body_energy = float(np.mean(rms[: tail_start + 1])) + 1e-8
        return float(np.clip(tail_energy / body_energy, 0.0, 1.0))

    @staticmethod
    def _estimate_decay_slope(rms: np.ndarray) -> float:
        if rms.size < 12:
            return 0.0
        envelope = np.maximum(rms, 1e-8)
        tail = envelope[int(envelope.size * 0.55) :]
        x_axis = np.arange(tail.size)
        log_tail = np.log(tail)
        slope, _ = np.polyfit(x_axis, log_tail, deg=1)
        return float(np.clip(slope, -1.0, 1.0))

    @staticmethod
    def _estimate_background_consistency(waveform: np.ndarray, sample_rate: int) -> float:
        window = max(sample_rate // 2, 1)
        usable = waveform[: waveform.size - (waveform.size % window)]
        if usable.size < window * 2:
            return 0.5
        chunks = usable.reshape(-1, window)
        energies = np.mean(np.square(chunks), axis=1)
        variation = float(np.std(energies) / (np.mean(energies) + 1e-8))
        return float(np.clip(1.0 - variation, 0.0, 1.0))

    @staticmethod
    def _estimate_breathing_cadence(waveform: np.ndarray, sample_rate: int) -> float:
        band = np.append(waveform[0], waveform[1:] - (0.97 * waveform[:-1]))
        envelope = np.abs(librosa.util.normalize(band))
        hop = max(sample_rate // 10, 1)
        blocks = [np.mean(envelope[index : index + hop]) for index in range(0, envelope.size, hop)]
        if len(blocks) < 4:
            return 0.5
        block_array = np.asarray(blocks)
        low_energy_pulses = block_array < np.quantile(block_array, 0.35)
        transitions = np.count_nonzero(np.diff(low_energy_pulses.astype(int)))
        cadence = transitions / max(len(block_array), 1)
        return float(np.clip(cadence, 0.0, 1.0))

    @staticmethod
    def _build_anomaly_curve(
        rms: np.ndarray,
        zcr: np.ndarray,
        centroid: np.ndarray,
        reverb_tail_ratio: float,
        background_consistency: float,
    ) -> list[float]:
        min_size = min(rms.size, zcr.size, centroid.size)
        if min_size == 0:
            return []
        rms_norm = _normalize_series(rms[:min_size])
        zcr_norm = _normalize_series(zcr[:min_size])
        centroid_norm = _normalize_series(centroid[:min_size])
        global_pressure = (reverb_tail_ratio + (1.0 - background_consistency)) / 2.0
        curve = (0.4 * rms_norm) + (0.25 * zcr_norm) + (0.2 * centroid_norm) + (0.15 * global_pressure)
        return np.clip(curve, 0.0, 1.0).astype(float).tolist()


def _normalize_series(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _frame_audio(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if waveform.size < frame_length:
        padded = np.pad(waveform, (0, frame_length - waveform.size))
        return padded.reshape(1, frame_length)

    frame_count = 1 + ((waveform.size - frame_length) // hop_length)
    shape = (frame_count, frame_length)
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
    return np.asarray(frames, dtype=np.float32)


def _spectral_rolloff(spectrum: np.ndarray, frequencies: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    cumulative = np.cumsum(spectrum, axis=1)
    limits = cumulative[:, -1:] * threshold
    indices = np.argmax(cumulative >= limits, axis=1)
    return frequencies[indices]


def _spectral_band_summary(spectrum: np.ndarray, band_count: int) -> np.ndarray:
    bands = np.array_split(spectrum, band_count, axis=1)
    energies = np.asarray([np.mean(band) for band in bands], dtype=np.float32)
    energies = np.maximum(energies, 1e-8)
    return 20.0 * np.log10(energies / np.max(energies))


def _compact_cepstral_summary(band_summary: np.ndarray, coefficient_count: int) -> np.ndarray:
    centered = band_summary - np.mean(band_summary)
    cepstral = np.fft.rfft(centered).real
    if cepstral.size < coefficient_count:
        cepstral = np.pad(cepstral, (0, coefficient_count - cepstral.size))
    return cepstral[:coefficient_count]
