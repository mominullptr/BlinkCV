# audio_alert.py — Non-Blocking Synthesized Gentle Zen Audio System
import io
import math
import struct
import threading
import time
import wave
import winsound
import config


def _build_wav(notes, sample_rate=44100):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        raw = bytearray()
        for freq, dur, vol, mode in notes:
            n_samples = int(sample_rate * dur)
            attack = int(0.025 * sample_rate)    # Soft 25ms attack (zero harsh click)
            decay = int(0.060 * sample_rate)     # Smooth exponential release

            for i in range(n_samples):
                if freq <= 0.0:
                    sample = 0
                else:
                    t = i / sample_rate
                    env = 1.0
                    if i < attack:
                        env = (i / max(1, attack)) ** 1.5
                    elif i > n_samples - decay:
                        env = ((n_samples - i) / max(1, decay)) ** 1.8

                    if mode == "zen":
                        # Soothing acoustic bell harmonic synthesis (Fundamental + 2nd + 3rd harmonic)
                        sig = (
                            0.70 * math.sin(2.0 * math.pi * freq * t)
                            + 0.22 * math.sin(4.0 * math.pi * freq * t)
                            + 0.08 * math.sin(6.0 * math.pi * freq * t)
                        )
                    else:
                        # Soft acoustic wooden droplet tap
                        sig = (
                            0.78 * math.sin(2.0 * math.pi * freq * t)
                            + 0.22 * math.sin(3.0 * math.pi * freq * t)
                        )

                    sample = int(sig * env * vol * 32767)
                    sample = max(-32768, min(32767, sample))
                raw.extend(struct.pack("<h", sample))

        wf.writeframes(bytes(raw))
    return buf.getvalue()


class AudioAlertManager:
    """
    Ultra-Gentle Harmonic Audio Alerts (Calm, Non-Irritating & Rate-Limited):
    1. Sound 1 (Blink Whisper): Soft D5 -> A5 Zen Chime (Vol: 0.28)
    2. Sound 2 (Posture Nudge): Gentle Low Double-Tap (Vol: 0.28)
    3. Sound 3 (20-20-20 Break): Calming 3-Note Zen Chord
    """

    def __init__(self):
        # Default muted to prioritize visual floating pill, user can toggle anytime
        self.muted = config.AUDIO_MUTED_BY_DEFAULT
        self.last_blink_sound = 0.0
        self.last_posture_sound = 0.0
        self._lock = threading.Lock()

        # Sound 1: Soft Zen Droplet (D5 587Hz -> A5 880Hz, soft gentle chime)
        self.blink_wav = _build_wav([
            (587.33, 0.12, 0.26, "zen"),
            (0.0,    0.03, 0.00, "zen"),
            (880.00, 0.22, 0.30, "zen"),
        ])

        # Sound 2: Posture Nudge (Gentle 360Hz -> 300Hz wooden tap)
        self.posture_wav = _build_wav([
            (360.00, 0.08, 0.26, "tap"),
            (0.0,    0.04, 0.00, "tap"),
            (300.00, 0.12, 0.28, "tap"),
        ])

        # Sound 3: 20-20-20 Break Zen Chord (C5 -> E5 -> G5)
        self.break_wav = _build_wav([
            (523.25, 0.14, 0.28, "zen"),
            (659.25, 0.14, 0.28, "zen"),
            (783.99, 0.26, 0.32, "zen"),
        ])

    def play_blink_alert(self, force=False):
        if self.muted and not force:
            return
        now = time.perf_counter()
        if force or (now - self.last_blink_sound) >= config.BLINK_SOUND_COOLDOWN:
            self.last_blink_sound = now
            threading.Thread(target=self._play_bytes, args=(self.blink_wav,), daemon=True).start()

    def play_proximity_or_posture_alert(self, force=False):
        if self.muted and not force:
            return
        now = time.perf_counter()
        if force or (now - self.last_posture_sound) >= config.POSTURE_SOUND_COOLDOWN:
            self.last_posture_sound = now
            threading.Thread(target=self._play_bytes, args=(self.posture_wav,), daemon=True).start()

    def play_break_chime(self):
        if self.muted:
            return
        threading.Thread(target=self._play_bytes, args=(self.break_wav,), daemon=True).start()

    def test_all_sounds(self):
        """Forces all 3 sounds to play sequentially so user can verify audio output anytime with [T]."""
        def _seq():
            self._play_bytes(self.blink_wav)
            time.sleep(0.30)
            self._play_bytes(self.posture_wav)
            time.sleep(0.30)
            self._play_bytes(self.break_wav)

        threading.Thread(target=_seq, daemon=True).start()

    def _play_bytes(self, wav_data):
        with self._lock:
            try:
                winsound.PlaySound(wav_data, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
            except Exception:
                pass
