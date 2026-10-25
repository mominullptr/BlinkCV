# ai_fatigue_predictor.py — Neural Cognitive Eye Strain & Fatigue Predictor
import math
import numpy as np


class CognitiveEyeStrainPredictor:
    """
    Lightweight 2-Layer Neural Network (MLP) for Real-Time Eye Strain & Fatigue Index:
    Input Features (8-D Vector):
    1. PERCLOS (% eyelid droop/closure in last 60s)
    2. Continuous stare duration (seconds without blink)
    3. Blinks per minute (BPM)
    4. Screen distance (cm)
    5. Slouch posture flag (0.0 or 1.0)
    6. Ambient room lux (luminance)
    7. Total yawns count
    8. Static gaze fixation duration (seconds without eye saccade)

    Output:
    - Eye Strain Index: 0% to 100%
    - Classification: OPTIMAL (0-35%), MILD_STRAIN (36-70%), HIGH_FATIGUE (71-100%)
    - Ergonomic Recommendations
    """

    def __init__(self):
        # Calibrated weights for clinical CVS (Computer Vision Syndrome) detection
        # Layer 1: 8 -> 16
        np.random.seed(42)
        self.w1 = np.array([
            # PERCLOS, StareSec,    BPM, DistCM, Slouch,   Lux, Yawns, Fixation
            [ 0.65,     0.55,    -0.45,  -0.40,   0.35,  -0.25,  0.40,   0.30],  # Neuron 1: Dry Eye Risk
            [ 0.80,     0.30,    -0.20,  -0.10,   0.20,  -0.10,  0.60,   0.15],  # Neuron 2: Drowsiness
            [ 0.20,     0.75,    -0.50,  -0.55,   0.45,  -0.35,  0.10,   0.50],  # Neuron 3: Ciliary Spasm (Staring)
            [ 0.10,     0.15,    -0.10,  -0.80,   0.70,  -0.15,  0.05,   0.10],  # Neuron 4: Posture / Proximity Strain
            [ 0.40,     0.40,    -0.30,  -0.30,   0.20,  -0.65,  0.20,   0.25],  # Neuron 5: Dark Room Contrast Strain
            [ 0.50,     0.60,    -0.40,  -0.35,   0.30,  -0.20,  0.30,   0.60],  # Neuron 6: Fixation Lock
            [ 0.70,     0.35,    -0.25,  -0.20,   0.25,  -0.15,  0.55,   0.20],  # Neuron 7: General Burnout
            [ 0.30,     0.70,    -0.60,  -0.45,   0.40,  -0.30,  0.15,   0.45],  # Neuron 8: Accommodation Fatigue
            [ 0.25,     0.20,    -0.15,  -0.70,   0.65,  -0.10,  0.10,   0.15],  # Neuron 9: Lean-In Strain
            [ 0.60,     0.50,    -0.35,  -0.30,   0.25,  -0.50,  0.35,   0.35],  # Neuron 10: Screen Glare
            [ 0.55,     0.45,    -0.30,  -0.25,   0.20,  -0.20,  0.45,   0.30],  # Neuron 11: Sleep Deprivation
            [ 0.35,     0.65,    -0.45,  -0.40,   0.35,  -0.30,  0.20,   0.55],  # Neuron 12: Visual Focus Load
            [ 0.75,     0.30,    -0.20,  -0.15,   0.15,  -0.10,  0.50,   0.10],  # Neuron 13: Eyelid Heaviness
            [ 0.20,     0.50,    -0.40,  -0.60,   0.50,  -0.25,  0.10,   0.40],  # Neuron 14: Head/Neck Tension
            [ 0.45,     0.40,    -0.30,  -0.25,   0.20,  -0.40,  0.25,   0.30],  # Neuron 15: Ambient Fatigue
            [ 0.50,     0.55,    -0.50,  -0.35,   0.30,  -0.25,  0.30,   0.45],  # Neuron 16: Composite Eye Strain
        ], dtype=np.float32)

        self.b1 = np.array([-0.25] * 16, dtype=np.float32)

        # Layer 2: 16 -> 1
        self.w2 = np.array([
            0.18, 0.16, 0.19, 0.12, 0.14, 0.15, 0.16, 0.18,
            0.11, 0.13, 0.15, 0.16, 0.14, 0.13, 0.12, 0.17
        ], dtype=np.float32).reshape(16, 1)

        self.b2 = np.array([-0.10], dtype=np.float32)

        self.smoothed_strain_score = 12.0

    def _normalize_features(
        self,
        perclos: float,
        stare_sec: float,
        bpm: float,
        dist_cm: float,
        is_slouching: bool,
        ambient_lux: float,
        yawns: int,
        fixation_sec: float,
    ) -> np.ndarray:
        n_perclos = min(1.0, max(0.0, perclos / 40.0))
        n_stare = min(1.0, max(0.0, stare_sec / 15.0))
        n_bpm = min(1.0, max(0.0, (22.0 - bpm) / 18.0)) if bpm < 22 else 0.0
        n_dist = min(1.0, max(0.0, (55.0 - dist_cm) / 25.0)) if dist_cm < 55 else 0.0
        n_slouch = 1.0 if is_slouching else 0.0
        n_lux = min(1.0, max(0.0, (50.0 - ambient_lux) / 50.0)) if ambient_lux < 50 else 0.0
        n_yawns = min(1.0, max(0.0, yawns / 5.0))
        n_fixation = min(1.0, max(0.0, fixation_sec / 20.0))

        return np.array([
            n_perclos, n_stare, n_bpm, n_dist, n_slouch, n_lux, n_yawns, n_fixation
        ], dtype=np.float32)

    def predict(
        self,
        perclos: float,
        stare_sec: float,
        bpm: float,
        dist_cm: float,
        is_slouching: bool,
        ambient_lux: float,
        yawns: int,
        fixation_sec: float,
    ) -> dict:
        x = self._normalize_features(
            perclos, stare_sec, bpm, dist_cm, is_slouching, ambient_lux, yawns, fixation_sec
        )

        h1 = np.maximum(0.0, np.dot(self.w1, x) + self.b1)
        raw_out = float(np.dot(h1, self.w2)[0] + self.b2[0])
        prob = 1.0 / (1.0 + math.exp(-raw_out))
        raw_strain = prob * 100.0

        raw_strain = max(5.0, min(98.0, raw_strain))
        self.smoothed_strain_score = 0.85 * self.smoothed_strain_score + 0.15 * raw_strain

        score = self.smoothed_strain_score

        if score < 35.0:
            level = "OPTIMAL"
            tag = "Fresh"
            color_hex = "#00e676"
            rec = "Eyes healthy and rested."
        elif score < 70.0:
            level = "MILD_STRAIN"
            tag = "Mild Strain"
            color_hex = "#ffb74d"
            if stare_sec >= 7.0:
                rec = "Blink consciously to hydrate cornea."
            elif dist_cm < 42.0:
                rec = "Sit back slightly past 45cm."
            else:
                rec = "Take a short visual pause."
        else:
            level = "HIGH_FATIGUE"
            tag = "High Fatigue"
            color_hex = "#ff4d6d"
            rec = "High strain detected! Practice 20-20-20 break."

        return {
            "strain_score": score,
            "level": level,
            "tag": tag,
            "color_hex": color_hex,
            "recommendation": rec,
        }
