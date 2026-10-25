# hud_renderer.py — FSM HUD + Live 64x64 Roll-Aligned Eye-Patch Pixel Verifier Panel
import math
import cv2
import numpy as np
import config
from ear_calculator import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR


def draw_eye_contours(frame, landmarks, state):
    h, w, _ = frame.shape
    if state == "OPEN":
        col_l, col_r = (0, 230, 115), (0, 230, 115)
    elif state == "LEFT CLOSED":
        col_l, col_r = (0, 180, 255), (0, 230, 115)
    elif state == "RIGHT CLOSED":
        col_l, col_r = (0, 230, 115), (0, 180, 255)
    else:
        col_l, col_r = (0, 70, 255), (0, 70, 255)

    pts_l = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE_CONTOUR], np.int32)
    pts_r = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE_CONTOUR], np.int32)

    cv2.polylines(frame, [pts_l], True, col_l, 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts_r], True, col_r, 1, cv2.LINE_AA)


def _draw_eye_patch_pip(frame, info):
    """
    Renders live 64x64 Roll-Aligned eye crops on top-right.
    Mirrored to match the webcam view: Screen-Left eye on left, Screen-Right eye on right.
    """
    h, w, _ = frame.shape
    panel_w, panel_h = 176, 134
    x0, y0 = w - panel_w - 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 160, 255), 1)

    backend_label = info.get("verifier_backend", "ONNX")
    cv2.putText(
        frame,
        f"PX-AI: {backend_label}",
        (x0 + 10, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 220, 255),
        1,
    )

    # In mirrored webcam view:
    # Screen-Left eye is anatomical right (r_patch, r_px_prob)
    # Screen-Right eye is anatomical left (l_patch, l_px_prob)
    screen_l_patch = info.get("r_patch_bgr")
    screen_r_patch = info.get("l_patch_bgr")
    prob_l = info.get("r_px_prob", 0.0)
    prob_r = info.get("l_px_prob", 0.0)

    if screen_l_patch is not None and screen_r_patch is not None:
        py = y0 + 26
        lx, rx = x0 + 14, x0 + 98
        frame[py : py + 64, lx : lx + 64] = screen_l_patch
        frame[py : py + 64, rx : rx + 64] = screen_r_patch

        l_closed = prob_l >= config.FUSED_CLOSE_PROB_THRESH
        r_closed = prob_r >= config.FUSED_CLOSE_PROB_THRESH

        cv2.rectangle(frame, (lx, py), (lx + 64, py + 64), (0, 60, 255) if l_closed else (0, 220, 110), 2)
        cv2.rectangle(frame, (rx, py), (rx + 64, py + 64), (0, 60, 255) if r_closed else (0, 220, 110), 2)

        glare_l = info.get("r_glare_ratio", 0.0)
        glare_r = info.get("l_glare_ratio", 0.0)
        if glare_l > 0.08:
            cv2.putText(frame, "GLARE", (lx + 10, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 215, 255), 1)
        if glare_r > 0.08:
            cv2.putText(frame, "GLARE", (rx + 10, py + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 215, 255), 1)

        cv2.putText(
            frame,
            f"L:{prob_l*100:.0f}%",
            (lx + 6, py + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (240, 240, 240),
            1,
        )
        cv2.putText(
            frame,
            f"R:{prob_r*100:.0f}%",
            (rx + 6, py + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (240, 240, 240),
            1,
        )


def render_subconscious_pulse(frame, time_since_blink, now):
    """Subconscious Optical Nudge (7s+): Renders a soft pulsating peripheral dot."""
    if time_since_blink < config.BLINK_VISUAL_PULSE_SECONDS:
        return

    h, w, _ = frame.shape
    pulse = 0.5 + 0.5 * math.sin((now * 4.2))
    radius = int(7 + 4 * pulse)
    center = (w - 200, 28)

    glow = frame.copy()
    cv2.circle(glow, center, radius + 8, (0, 165, 255), -1)
    cv2.addWeighted(glow, 0.35 * pulse, frame, 1.0 - (0.35 * pulse), 0, frame)
    cv2.circle(frame, center, radius, (0, 215, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "BLINK", (center[0] - 46, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 215, 255), 1)


def render_hud(
    frame,
    info,
    fsm_status,
    time_since_blink,
    total_blinks,
    total_winks,
    bpm,
    ambient_lux,
    muted,
    banner_text,
    compact=False,
    now=0.0,
    ai_data=None,
    gaze_data=None,
):
    h, w, _ = frame.shape
    overlay = frame.copy()

    card_w = 360
    card_h = 135 if compact else 322
    cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (15, 15, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    fsm_state = fsm_status["state_name"]
    dist_cm = info["dist_cm"]
    too_close = fsm_status["too_close"] and (dist_cm < config.PROXIMITY_SAFE_DISTANCE_CM)

    border_col = (0, 220, 120)
    if time_since_blink >= config.BLINK_ALERT_SECONDS or fsm_state == "POSTURE_WARNING":
        border_col = (0, 50, 255)
    elif fsm_state in ("BREAK_PENDING", "BREAK_ACTIVE"):
        border_col = (255, 190, 0)
    elif time_since_blink >= 5.0:
        border_col = (0, 180, 255)

    cv2.rectangle(frame, (12, 12), (12 + card_w, 12 + card_h), border_col, 2)

    cv2.putText(frame, "BLINKCV HYBRID AI", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2)
    audio_txt = "MUTED [M]" if muted else "AUDIO ON [M]"
    cv2.putText(
        frame,
        audio_txt,
        (255, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (100, 100, 255) if muted else (130, 230, 150),
        1,
    )

    cv2.putText(frame, f"FSM: {fsm_state}", (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 215, 100), 1)

    state = info["state"]
    state_col = (0, 235, 115) if state == "OPEN" else ((0, 195, 255) if "WINK" in state else (0, 80, 255))
    cv2.putText(frame, f"Eye State: {state}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_col, 2)

    # TFOS DEWS II Corneal Hydration & Stare Duration
    moisture = info.get("corneal_moisture", 100.0)
    m_col = (0, 230, 115) if moisture > 55 else ((0, 195, 255) if moisture > 25 else (0, 65, 255))
    t_col = (0, 65, 255) if time_since_blink >= config.BLINK_ALERT_SECONDS else (255, 255, 255)

    cv2.putText(
        frame,
        f"No-Blink: {time_since_blink:.1f}s | Tear Film: {moisture:.0f}%",
        (24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        t_col if time_since_blink >= config.BLINK_ALERT_SECONDS else m_col,
        2,
    )

    bar_x, bar_y, bar_w, bar_h = 24, 116, 324, 7
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, moisture / 100.0)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), m_col, -1)

    if not compact:
        # Multi-Modal Telemetry (Independent Anisometropia Baselines)
        cv2.putText(
            frame,
            f"EAR L:{info['left_ear']:.2f}(B:{info['l_base']:.2f}) R:{info['right_ear']:.2f}(B:{info['r_base']:.2f})",
            (24, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
        )
        full_b = info.get("full_blinks", total_blinks)
        part_b = info.get("incomplete_blinks", 0)
        part_pct = info.get("incomplete_blink_ratio", 0.0)
        cv2.putText(
            frame,
            f"Blinks: {total_blinks} ({bpm}/m) [Full:{full_b} Part:{part_b} ({part_pct:.0f}%)]",
            (24, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 220, 255),
            1,
        )

        d_col = (0, 65, 255) if (too_close or info["is_slouching"]) else (120, 240, 160)
        posture_tag = "SLOUCH!" if info["is_slouching"] else ("TOO CLOSE!" if too_close else "GOOD")
        cv2.putText(
            frame,
            f"Dist: {dist_cm:.0f}cm (Safe>={config.SAFE_DISTANCE_CM:.0f}) | Posture: {posture_tag}",
            (24, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            d_col,
            1,
        )

        rem_m = int(fsm_status["work_remaining"] // 60)
        rem_s = int(fsm_status["work_remaining"] % 60)
        cv2.putText(
            frame,
            f"20-20-20 Next: {rem_m:02d}:{rem_s:02d} | Snooze: {fsm_status['snooze_count']}/{config.MAX_SNOOZE_COUNT}",
            (24, 212),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (255, 220, 140),
            1,
        )

        # AI Cognitive Eye Strain Index & Optical Gaze
        if ai_data and gaze_data:
            ai_score = ai_data["strain_score"]
            ai_col = (0, 230, 115) if ai_score < 35 else ((0, 195, 255) if ai_score < 70 else (0, 80, 255))
            cv2.putText(
                frame,
                f"AI Strain: {ai_score:.0f}% ({ai_data['level']}) | Gaze: {gaze_data['gaze_zone']}",
                (24, 236),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                ai_col,
                1,
            )
        else:
            f_col = (0, 80, 255) if info["perclos"] >= config.PERCLOS_FATIGUE_THRESH else (170, 210, 170)
            light_tag = "LOW LIGHT!" if ambient_lux < config.LOW_LIGHT_LUX_THRESH else f"Lux:{ambient_lux:.0f}"
            cv2.putText(
                frame,
                f"Fatigue(PERCLOS): {info['perclos']:.0f}% | Room: {light_tag}",
                (24, 236),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                f_col,
                1,
            )

        cv2.putText(
            frame,
            f"Fixation: {gaze_data['fixation_seconds']:.1f}s | Room Lux: {ambient_lux:.0f}" if gaze_data else f"Room Lux: {ambient_lux:.0f}",
            (24, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (175, 195, 215),
            1,
        )

        cv2.putText(
            frame,
            "[S] Snooze  [B] BreakNow  [P] CamOff  [D] DND  [C] Calib  [Q] Quit",
            (24, 294),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (145, 145, 145),
            1,
        )

        # Draw live 64x64 eye-patch PIP verifier on top-right
        _draw_eye_patch_pip(frame, info)

    # Top-Center Event Flash Banner
    if info.get("calibrating", False):
        cv2.rectangle(frame, (w // 2 - 170, 14), (w // 2 + 170, 48), (20, 120, 220), -1)
        cv2.putText(frame, "CALIBRATING EYES... (HOLD STILL)", (w // 2 - 150, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    elif banner_text:
        cv2.rectangle(frame, (w // 2 - 155, 14), (w // 2 + 155, 48), (0, 170, 85), -1)
        cv2.putText(frame, banner_text, (w // 2 - 135, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    render_subconscious_pulse(frame, time_since_blink, now)

    if fsm_state == "BREAK_PENDING":
        cv2.rectangle(frame, (15, h - 78), (w - 15, h - 14), (20, 110, 220), -1)
        cv2.putText(
            frame,
            "20-20-20 BREAK READY! Look 20ft away (or close eyes) to start",
            (30, h - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Coding in flow? Press [S] on keyboard to Snooze for 5 mins",
            (30, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 240, 255),
            1,
        )

    elif fsm_state == "BREAK_ACTIVE":
        cv2.rectangle(frame, (15, h - 78), (w - 15, h - 14), (0, 150, 90), -1)
        left_s = fsm_status["break_remaining"]
        is_away = gaze_data["is_looking_away"] if gaze_data else info["is_looking_away"]
        gaze_ok = "GAZE AWAY RESTING" if is_away else "PLEASE LOOK AWAY FROM SCREEN!"
        cv2.putText(
            frame,
            f"20-20-20 RELAXATION: {left_s:.1f}s remaining ({gaze_ok})",
            (30, h - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )

    elif fsm_state == "POSTURE_WARNING":
        cv2.rectangle(frame, (w // 2 - 170, 54), (w // 2 + 170, 90), (0, 95, 255), -1)
        msg = f"TOO CLOSE ({dist_cm:.0f}cm)!" if too_close else "POSTURE ALERT: SIT UPRIGHT!"
        cv2.putText(frame, msg, (w // 2 - 145, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

    if time_since_blink >= config.BLINK_ALERT_SECONDS and fsm_state not in ("BREAK_ACTIVE", "DND_PAUSED"):
        cv2.rectangle(frame, (w // 2 - 155, 96), (w // 2 + 155, 132), (0, 35, 235), -1)
        cv2.putText(
            frame,
            "BLINK NOW! (7s DRY-EYE GUARD)",
            (w // 2 - 138, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
