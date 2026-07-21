import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extract Librosa features from a single audio file.
    Returns a dictionary of numerical features.
    """

    y, sr = librosa.load(audio_path, sr=16000)

    features = {}

    # -----------------------------
    # Time-domain Features
    # -----------------------------
    features["rms"] = np.mean(librosa.feature.rms(y=y))

    features["zcr"] = np.mean(
        librosa.feature.zero_crossing_rate(y)
    )

    # -----------------------------
    # Spectral Features
    # -----------------------------
    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    features["spectral_flatness"] = np.mean(
        librosa.feature.spectral_flatness(y=y)
    )

    features["spectral_contrast"] = np.mean(
        librosa.feature.spectral_contrast(
            y=y,
            sr=sr
        )
    )

    # -----------------------------
    # Harmonic Features
    # -----------------------------
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features["chroma_mean"] = np.mean(chroma)

    # -----------------------------
    # MFCC Features
    # -----------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}"] = np.mean(mfcc[i])

    return features