# Shubhangi, Upload Date: 2026-07-15
from pathlib import Path
from wave import open as wave_open

import numpy as np


def main() -> None:
    sample_rate = 16_000
    duration_seconds = 4
    time = np.linspace(0, duration_seconds, sample_rate * duration_seconds, endpoint=False)
    voice_like = 0.32 * np.sin(2 * np.pi * 180 * time) + 0.12 * np.sin(2 * np.pi * 420 * time)
    room_tail = np.exp(-np.linspace(0, 5, time.size)) * np.sin(2 * np.pi * 55 * time)
    background = 0.015 * np.random.default_rng(7).normal(size=time.size)
    waveform = voice_like + 0.25 * room_tail + background
    waveform = waveform / np.max(np.abs(waveform))
    pcm = (waveform * 32767).astype(np.int16)

    output_dir = Path("data/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_suspect_room.wav"

    with wave_open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    print(f"Demo audio written to {output_path.resolve()}")


if __name__ == "__main__":
    main()

