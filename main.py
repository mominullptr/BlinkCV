# main.py — Hybrid MediaPipe + 64x64 ONNX/Pixel Eye-Patch Classifier + Desktop Floating Pill
import os
import sys

# Production ultra-low CPU thread governance (passive waiting & single-threaded execution)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import urllib.request
from enum import Enum, auto
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from ear_calculator import AdaptiveVisionEngine
from audio_alert import AudioAlertManager
from hud_renderer import draw_eye_contours, render_hud
from desktop_widget import DesktopOverlayWidget
from ai_gaze_tracker import TinyGazeTracker
from ai_fatigue_predictor import CognitiveEyeStrainPredictor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class State(Enum):
    IDLE_AFK = auto()           # User away from desk (Timers paused; >2m resets 20m session)
    MONITORING_NORMAL = auto()  # Standard coding state (Silent & non-intrusive)
    POSTURE_WARNING = auto()    # Distance < 38cm or sustained slouch > 5s (with Hysteresis)
    BREAK_PENDING = auto()      # 20 mins elapsed; waiting for user to look away or press [S] snooze
    BREAK_ACTIVE = auto()       # Optical gaze-verified 20-second eye relaxation countdown
    DND_PAUSED = auto()         # Meeting / Presentation silent mode


class ProximityHysteresis:
    """
    Schmitt-Trigger Distance Hysteresis Filter:
    - Trigger: dist < warn_cm sustained for debounce frames (ultra-fast 3-frame response).
    - Release: dist >= safe_cm sustained for debounce frames.
    """

    def __init__(self, warn_cm: float = 45.0, safe_cm: float = 48.0, debounce: int = 3):
        self.warn_cm = warn_cm
        self.safe_cm = safe_cm
        self.debounce = debounce
        self.is_too_close = False
        self.trigger_counter = 0
        self.release_counter = 0

    def update(self, dist_cm: float) -> bool:
        if dist_cm >= self.safe_cm:
            self.trigger_counter = 0
            self.release_counter += 1
            if self.release_counter >= self.debounce:
                self.is_too_close = False
            return self.is_too_close

        if dist_cm < self.warn_cm:
            self.release_counter = 0
            self.trigger_counter += 1
            if self.trigger_counter >= self.debounce:
                self.is_too_close = True
        else:
            self.trigger_counter = 0

        return self.is_too_close


class SystemFSM:
    """Production Finite State Machine handling all developer workflow edge cases."""

    def __init__(self, audio: AudioAlertManager):
        self.state = State.MONITORING_NORMAL
        self.audio = audio
        now = time.perf_counter()

        self.session_start = now
        self.state_enter_time = now
        self.afk_start_time = None
        self.slouch_start_time = None
        self.last_posture_sound_ts = 0.0

        self.break_accumulated = 0.0
        self.last_frame_time = now
        self.snooze_count = 0
        self.dnd_enabled = False

        self.proximity_filter = ProximityHysteresis(
            warn_cm=config.PROXIMITY_WARN_DISTANCE_CM,
            safe_cm=config.PROXIMITY_SAFE_DISTANCE_CM,
            debounce=config.PROXIMITY_CONSEC_FRAMES,
        )

    def transition_to(self, new_state: State, now: float):
        if self.state != new_state:
            print(f"[FSM] {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_enter_time = now
            if new_state in (State.BREAK_PENDING, State.BREAK_ACTIVE):
                self.audio.play_break_chime()

    def trigger_snooze(self, now: float):
        """[S] key or Snooze button: Snoozes 20-20-20 break for 5 minutes."""
        if self.state in (State.BREAK_PENDING, State.BREAK_ACTIVE):
            if self.snooze_count < config.MAX_SNOOZE_COUNT:
                self.snooze_count += 1
                self.session_start = now - (config.WORK_DURATION_SECONDS - config.SNOOZE_DURATION_SECONDS)
                self.break_accumulated = 0.0
                self.transition_to(State.MONITORING_NORMAL, now)
                return f"SNOOZED 5 MIN ({self.snooze_count}/{config.MAX_SNOOZE_COUNT})"
            return "MAX SNOOZE LIMIT REACHED!"
        return "SNOOZE ONLY IN BREAK MODE"

    def force_break_now(self, now: float):
        """[B] key or Break button: Immediately triggers 20-20-20 break."""
        self.break_accumulated = 0.0
        self.transition_to(State.BREAK_PENDING, now)

    def update(self, face_detected: bool, info: dict, now: float):
        dt = max(0.0, min(0.25, now - self.last_frame_time))
        self.last_frame_time = now

        # Priority 1: DND Mode
        if self.dnd_enabled:
            self.transition_to(State.DND_PAUSED, now)
            return self._status_dict(now, False)

        if self.state == State.DND_PAUSED and not self.dnd_enabled:
            self.transition_to(State.MONITORING_NORMAL, now)

        # Priority 2: AFK Detection & >2 Min Auto-Reset
        if not face_detected:
            if self.state != State.IDLE_AFK:
                self.afk_start_time = now
                self.transition_to(State.IDLE_AFK, now)
            return self._status_dict(now, False)

        if self.state == State.IDLE_AFK and face_detected:
            afk_dur = now - (self.afk_start_time or now)
            if afk_dur >= config.AFK_RESET_SECONDS:
                self.session_start = now
                self.snooze_count = 0
                print(f"[AFK] Rested {afk_dur:.0f}s (>2m). 20-20-20 session timer freshly reset!")
            self.transition_to(State.MONITORING_NORMAL, now)

        too_close_active = self.proximity_filter.update(info["dist_cm"])

        if info["is_slouching"]:
            if self.slouch_start_time is None:
                self.slouch_start_time = now
            sustained_slouch = (now - self.slouch_start_time) >= config.SLOUCH_TOLERANCE_SECONDS
        else:
            self.slouch_start_time = None
            sustained_slouch = False

        # --- State Execution ---
        if self.state == State.MONITORING_NORMAL:
            work_elapsed = now - self.session_start
            # Cognitive Task-Boundary Aware Break Trigger:
            # Syncs 20m break with natural mental punctuation mark (Blink Burst / Gaze Shift)
            # or triggers gracefully after 90s timeout
            is_task_boundary = info.get("is_task_boundary", False) or info.get("is_looking_away", False)
            if work_elapsed >= config.WORK_DURATION_SECONDS:
                if is_task_boundary or (work_elapsed >= config.WORK_DURATION_SECONDS + 90.0):
                    self.break_accumulated = 0.0
                    self.transition_to(State.BREAK_PENDING, now)
            elif too_close_active or sustained_slouch:
                self.transition_to(State.POSTURE_WARNING, now)
                if (now - self.last_posture_sound_ts) >= config.POSTURE_SOUND_COOLDOWN:
                    self.last_posture_sound_ts = now
                    self.audio.play_proximity_or_posture_alert()

        elif self.state == State.POSTURE_WARNING:
            if not too_close_active and not sustained_slouch:
                self.transition_to(State.MONITORING_NORMAL, now)
            else:
                if (now - self.last_posture_sound_ts) >= config.POSTURE_SOUND_COOLDOWN:
                    self.last_posture_sound_ts = now
                    self.audio.play_proximity_or_posture_alert()
                if (now - self.session_start) >= config.WORK_DURATION_SECONDS:
                    self.transition_to(State.BREAK_PENDING, now)

        elif self.state == State.BREAK_PENDING:
            if info["is_looking_away"]:
                self.transition_to(State.BREAK_ACTIVE, now)

        elif self.state == State.BREAK_ACTIVE:
            if info["is_looking_away"]:
                self.break_accumulated += dt
                if self.break_accumulated >= config.BREAK_DURATION_SECONDS:
                    print("[+] 20-20-20 Break Completed! Session Reset.")
                    self.session_start = now
                    self.snooze_count = 0
                    self.break_accumulated = 0.0
                    self.audio.play_break_chime()
                    self.transition_to(State.MONITORING_NORMAL, now)

        return self._status_dict(now, too_close_active)

    def _status_dict(self, now: float, too_close: bool):
        work_rem = max(0.0, config.WORK_DURATION_SECONDS - (now - self.session_start))
        break_rem = max(0.0, config.BREAK_DURATION_SECONDS - self.break_accumulated)
        return {
            "state_name": self.state.name,
            "too_close": too_close,
            "work_remaining": work_rem,
            "break_remaining": break_rem,
            "snooze_count": self.snooze_count,
        }


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[*] Downloading MediaPipe Face Landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Model ready.")


def preprocess_frame(bgr_frame, clahe=None):
    """Zero-overhead luminance check + direct RGB conversion (saves ~12% CPU)."""
    # Sample every 8th pixel of green channel for instant room lux calculation without full-frame LAB split
    ambient_lux = float(np.mean(bgr_frame[::8, ::8, 1]))
    if getattr(config, "ENABLE_CLAHE", False) and clahe is not None:
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = clahe.apply(l_chan)
        merged = cv2.merge((l_chan, a_chan, b_chan))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return rgb, ambient_lux


def main():
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    vision_engine = AdaptiveVisionEngine()
    gaze_tracker = TinyGazeTracker()
    strain_predictor = CognitiveEyeStrainPredictor()
    audio = AudioAlertManager()
    fsm = SystemFSM(audio)

    total_blinks = 0
    total_winks = 0
    session_start = time.perf_counter()
    last_blink_ts = session_start
    last_video_ms = 0

    banner_text = ""
    banner_until = 0.0
    face_lost_frames = 0
    compact_hud = False
    show_camera_window = True

    # Callbacks for Desktop Floating Pill
    def _toggle_audio():
        audio.muted = not audio.muted
        print(f"[Overlay] Audio: {'MUTED' if audio.muted else 'UNMUTED'}")

    def _toggle_camera():
        nonlocal show_camera_window
        show_camera_window = not show_camera_window
        if not show_camera_window:
            cv2.destroyAllWindows()
        print(f"[Overlay] Camera Window: {'VISIBLE' if show_camera_window else 'HIDDEN'}")

    def _snooze():
        fsm.trigger_snooze(time.perf_counter())

    def _force_break():
        fsm.force_break_now(time.perf_counter())

    desktop_pill = DesktopOverlayWidget({
        "toggle_audio": _toggle_audio,
        "toggle_camera": _toggle_camera,
        "snooze": _snooze,
        "force_break": _force_break,
    })

    print("=" * 76)
    print("  BlinkCV — Hybrid MediaPipe + 64x64 BlinkCNN-ONNX Engine")
    print(f"  • Pixel Verifier Backend: {vision_engine.pixel_classifier.backend_name}")
    print("  • Optical Iris Gaze Tracker + Cognitive Eye Strain Predictor Active")
    print("  • Floating Pill is ALWAYS ON TOP of VS Code, Chrome, Terminal")
    print("  • [S] Snooze 5m | [B] 20-20-20 Break | [P] Hide Cam | [M] Audio Mute")
    print("  • [D] DND Mode  | [C] Recalibrate    | [T] Test Audio | [Q] Quit")
    print("=" * 76)

    while desktop_pill.is_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.perf_counter()

        ts_ms = int((now - session_start) * 1000)
        if ts_ms <= last_video_ms:
            ts_ms = last_video_ms + 1
        last_video_ms = ts_ms

        rgb_frame, ambient_lux = preprocess_frame(frame, clahe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, ts_ms)

        if result.face_landmarks:
            face_lost_frames = 0
            landmarks = result.face_landmarks[0]
            blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None

            # Base Vision Engine processing with 64x64 Roll-Aligned Eye-Patch Verification
            info = vision_engine.process_frame(
                rgb_frame, landmarks, blendshapes, w, h, now, include_bgr_patches=show_camera_window
            )

            # Optical Iris Gaze Tracking
            gaze_data = gaze_tracker.update(landmarks, info["yaw"], info["pitch"], w, h, now)
            info["is_looking_away"] = gaze_data["is_looking_away"] or info["is_looking_away"]

            elapsed_min = max((now - session_start) / 60.0, 0.1)
            bpm = int(total_blinks / elapsed_min)
            fsm_status = fsm.update(True, info, now)

            if info["reset_timer_now"]:
                last_blink_ts = now

            mois = info.get("corneal_moisture", 100.0)

            if info["event"] == "BLINK":
                total_blinks += 1
                last_blink_ts = now
                banner_text = f"FULL BLINK (#{total_blinks}) | Mois: {mois:.0f}%"
                banner_until = now + 0.50
                print(
                    f"[+] Blink #{total_blinks} | Mois:{mois:.0f}% | PxProb L:{info['l_px_prob']:.2f} R:{info['r_px_prob']:.2f} "
                    f"| EAR L:{info['left_ear']:.2f} R:{info['right_ear']:.2f} | AI:{info['verifier_backend']}"
                )

            elif info["event"] == "PARTIAL_BLINK":
                total_blinks += 1
                last_blink_ts = now
                banner_text = f"PARTIAL BLINK DETECTED (Mois: {mois:.0f}%)"
                banner_until = now + 0.50
                print(
                    f"[*] Partial Blink #{total_blinks} (mEBAL) | Mois:{mois:.0f}% | PxProb L:{info['l_px_prob']:.2f} R:{info['r_px_prob']:.2f}"
                )

            elif info["event"] in ("WINK_LEFT", "WINK_RIGHT"):
                total_winks += 1
                last_blink_ts = now
                side = "LEFT" if info["event"] == "WINK_LEFT" else "RIGHT"
                banner_text = f"{side} WINK DETECTED!"
                banner_until = now + 0.60

            elif info["event"] == "YAWN":
                banner_text = "YAWN / FATIGUE DETECTED!"
                banner_until = now + 1.0

            time_since_blink = max(0.0, now - last_blink_ts)

            # AI Cognitive Eye Strain Prediction
            ai_data = strain_predictor.predict(
                perclos=info["perclos"],
                stare_sec=time_since_blink,
                bpm=bpm,
                dist_cm=info["dist_cm"],
                is_slouching=info["is_slouching"],
                ambient_lux=ambient_lux,
                yawns=info["yawns"],
                fixation_sec=gaze_data["fixation_seconds"],
            )

            # TFOS DEWS II Virtual Tear-Film Hydration Alert:
            # Alerts when stare exceeds 7.0s OR corneal moisture drops below 4.0%
            is_dry = (time_since_blink >= config.BLINK_ALERT_SECONDS) or (mois <= 4.0)
            if not audio.muted and is_dry:
                if fsm.state not in (State.BREAK_ACTIVE, State.DND_PAUSED):
                    audio.play_blink_alert()

            if show_camera_window:
                draw_eye_contours(frame, landmarks, info["state"])

            # Update Desktop Floating Pill
            pill_state = {
                "time_since_blink": time_since_blink,
                "dist_cm": info["dist_cm"],
                "too_close": fsm_status["too_close"],
                "is_slouching": info["is_slouching"],
                "fsm_state": fsm_status["state_name"],
                "work_remaining": fsm_status["work_remaining"],
                "break_remaining": fsm_status["break_remaining"],
                "audio_muted": audio.muted,
                "camera_visible": show_camera_window,
                "is_looking_away": info["is_looking_away"],
                "ai_strain_score": ai_data["strain_score"],
                "ai_tag": ai_data["tag"],
                "ai_color": ai_data["color_hex"],
                "gaze_zone": gaze_data["gaze_zone"],
                "corneal_moisture": mois,
                "needs_squeeze_blink": info.get("needs_squeeze_blink", False),
            }
            desktop_pill.update_state(pill_state)

            if show_camera_window:
                render_hud(
                    frame=frame,
                    info=info,
                    fsm_status=fsm_status,
                    time_since_blink=time_since_blink,
                    total_blinks=total_blinks,
                    total_winks=total_winks,
                    bpm=bpm,
                    ambient_lux=ambient_lux,
                    muted=audio.muted,
                    banner_text=banner_text if now < banner_until else "",
                    compact=compact_hud,
                    now=now,
                    ai_data=ai_data,
                    gaze_data=gaze_data,
                )
        else:
            face_lost_frames += 1
            if face_lost_frames > config.FACE_LOST_GRACE_FRAMES:
                last_blink_ts = now
                fsm_status = fsm.update(False, {}, now)
                desktop_pill.update_state({
                    "time_since_blink": 0.0,
                    "dist_cm": 0.0,
                    "too_close": False,
                    "is_slouching": False,
                    "fsm_state": "IDLE_AFK",
                    "work_remaining": fsm_status["work_remaining"],
                    "break_remaining": fsm_status["break_remaining"],
                    "audio_muted": audio.muted,
                    "camera_visible": show_camera_window,
                    "is_looking_away": False,
                    "ai_strain_score": 0.0,
                    "ai_tag": "Away",
                    "ai_color": "#9298b0",
                    "gaze_zone": "N/A",
                })
                if show_camera_window:
                    cv2.rectangle(frame, (w // 2 - 195, h // 2 - 24), (w // 2 + 195, h // 2 + 24), (18, 18, 26), -1)
                    cv2.putText(
                        frame,
                        f"FSM: {fsm_status['state_name']} - TIMERS PAUSED",
                        (w // 2 - 175, h // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 195, 255),
                        2,
                    )

        if show_camera_window:
            cv2.imshow("BlinkCV AI", frame)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            banner_text = fsm.trigger_snooze(now)
            banner_until = now + 1.2
        elif key in (ord("b"), ord("B")):
            fsm.force_break_now(now)
            banner_text = "20-20-20 BREAK TRIGGERED"
            banner_until = now + 1.0
        elif key in (ord("d"), ord("D")):
            fsm.dnd_enabled = not fsm.dnd_enabled
            banner_text = "DND MODE ON" if fsm.dnd_enabled else "DND MODE OFF"
            banner_until = now + 1.0
        elif key in (ord("c"), ord("C")):
            vision_engine.reset_calibration()
            last_blink_ts = now
            banner_text = "EYE & POSTURE RE-CALIBRATED"
            banner_until = now + 1.0
        elif key in (ord("t"), ord("T")):
            banner_text = "TESTING ALL 3 SOUNDS..."
            banner_until = now + 1.5
            audio.test_all_sounds()
        elif key in (ord("m"), ord("M")):
            _toggle_audio()
        elif key in (ord("p"), ord("P")):
            _toggle_camera()
        elif key in (ord("h"), ord("H")):
            compact_hud = not compact_hud
        elif key in (ord("+"), ord("=")):
            vision_engine.sensitivity_offset = min(0.15, vision_engine.sensitivity_offset + 0.02)
            banner_text = f"SENSITIVITY + ({config.CLOSE_RATIO + vision_engine.sensitivity_offset:.2f})"
            banner_until = now + 0.8
        elif key in (ord("-"), ord("_")):
            vision_engine.sensitivity_offset = max(-0.15, vision_engine.sensitivity_offset - 0.02)
            banner_text = f"SENSITIVITY - ({config.CLOSE_RATIO + vision_engine.sensitivity_offset:.2f})"
            banner_until = now + 0.8

        # Adaptive CPU Governor & Frame Pacer:
        # Paces execution to 28 FPS during active dynamic eye states (blinks, winks, strain, slouching, breaks)
        # and throttles down to 16 FPS when eyes are stably open (drops CPU load to ~5% - 8%)
        is_dynamic = (
            bool(result.face_landmarks)
            and (
                info.get("state", "OPEN") != "OPEN"
                or info.get("event") is not None
                or fsm_status.get("too_close", False)
                or info.get("is_slouching", False)
                or fsm.state in (State.BREAK_PENDING, State.BREAK_ACTIVE, State.POSTURE_WARNING)
                or time_since_blink >= (config.BLINK_ALERT_SECONDS - 0.5)
                or info.get("l_px_prob", 0.0) > 0.28
                or info.get("r_px_prob", 0.0) > 0.28
                or info.get("calibrating", False)
                or info.get("dist_cm", 50.0) < config.PROXIMITY_SAFE_DISTANCE_CM
            )
        )
        target_fps = getattr(config, "ACTIVE_TARGET_FPS", 30.0) if is_dynamic else getattr(config, "IDLE_TARGET_FPS", 25.0)
        elapsed_frame = time.perf_counter() - now
        sleep_needed = (1.0 / target_fps) - elapsed_frame
        if sleep_needed > 0.001:
            time.sleep(sleep_needed)

    desktop_pill.destroy()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
