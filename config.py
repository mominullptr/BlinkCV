# config.py — Hybrid MediaPipe + ONNX / Pixel Eye-Patch Classifier Configuration

# ==================== CAMERA & PERFORMANCE ====================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
IDLE_TARGET_FPS = 25.0               # Smooth pacing to reliably capture all fast micro-blinks (CPU stays <5%)
ACTIVE_TARGET_FPS = 30.0             # High-cadence pace during blink/wink/strain/movement
ENABLE_CLAHE = True                  # Normalizes lighting for eyeglasses glare & dark rooms (100% Backlight Immunity)

# ==================== AUDIO SETTINGS ====================
AUDIO_MUTED_BY_DEFAULT = False      # Can be toggled with [M] or desktop floating pill
BLINK_ALERT_SECONDS = 7.0           # Sound 1 (High Double Chime) at 7.0s
BLINK_VISUAL_PULSE_SECONDS = 7.0    # Visual pulse on screen
BLINK_AUDIO_ALERT_SECONDS = 7.0     # Audio chime timing
BLINK_SOUND_COOLDOWN = 3.2
POSTURE_SOUND_COOLDOWN = 3.0

# ==================== PIXEL / ONNX EYE-PATCH CLASSIFIER ====================
EYE_PATCH_SIZE = 64                 # 64x64 standard input resolution for BlinkCNN-ONNX
ONNX_MODEL_NAME = "blinkcnn-onnx"   # Pretrained ONNX model identifier
LOCAL_ONNX_FILENAME = "blinkcnn.onnx"

# Multi-Modal Fusion Weights (Must sum to 1.0)
WEIGHT_PIXEL_ONNX = 0.30            # Pixel-level eye patch neural verifier
WEIGHT_GEOMETRIC_EAR = 0.50         # Adaptive baseline-relative geometric EAR (most accurate & glare-immune)
WEIGHT_BLENDSHAPE = 0.20            # MediaPipe neural blendshape score

# Fused Closure Probability Thresholds (Schmitt Trigger Hysteresis in [0, 1])
FUSED_CLOSE_PROB_THRESH = 0.48      # Eye registers CLOSED when fused closure probability >= 48%
FUSED_OPEN_PROB_THRESH = 0.30       # Eye returns to OPEN when fused closure probability <= 30%

# Downward-Gaze Iris Guard
# Neutral face pitch is ~0.11. Only trigger guard when head is deeply pitched down (>0.24)
DOWNWARD_GAZE_PITCH_THRESH = 0.24
DOWNWARD_GAZE_IRIS_GUARD = 0.25     # Blocks false CLOSED only when static reading with clearly visible iris

# ==================== GEOMETRIC BASELINE & HYSTERESIS ====================
FAST_BASELINE_WINDOW = 45           # ~1.5s fast adaptation
SLOW_BASELINE_WINDOW = 120          # ~4.0s stability anchor
CALIBRATION_WARMUP_FRAMES = 30      # First 30 frames (~1.0s) learn user's exact open eye baselines
CLOSE_RATIO = 0.78                  # Relative EAR closure trigger (78% of open baseline)
OPEN_RATIO = 0.88                   # Relative EAR reopening trigger (88% of open baseline)
ABSOLUTE_MIN_EAR = 0.100            # Allow natural narrow eye / high power glasses frame shadow (down to 0.10)
VELOCITY_DIP_THRESH = 0.035         # 1-frame rapid blink velocity dip threshold (catches fast micro-blinks)

# Smile / Cheek-Squint & Yawn Thresholds
SMILE_WIDTH_RATIO = 0.38
SMILE_COMPENSATION = 0.08
YAWN_MAR_THRESH = 0.60
YAWN_CONSEC_FRAMES = 25

# Wink Differentiation
WINK_OTHER_EYE_MAX_PROB = 0.22      # Opposite eye closure prob must be <=22% (firmly open) for a single-eye wink
MIN_WINK_FRAMES = 2
MAX_YAW_FOR_WINK = 0.32

# ==================== PROXIMITY & POSTURE (BONY ORBIT + FACE WIDTH) ====================
SAFE_DISTANCE_CM = 48.0             # Normal safe laptop working distance (>= 48cm)
PROXIMITY_WARN_DISTANCE_CM = 45.0   # Hysteresis warn boundary (closer than 45cm triggers warning immediately)
PROXIMITY_SAFE_DISTANCE_CM = 48.0   # Hysteresis clear boundary (sitting back to >= 48cm clears warning)
PROXIMITY_CONSEC_FRAMES = 3         # Ultra-responsive 3-frame debounce (~100ms) eliminates lag
SLOUCH_TOLERANCE_SECONDS = 4.0

# Calibrated for standard laptop webcams (65° FOV)
FOCAL_LENGTH_PIXELS = 570.0
REAL_IPD_CM = 6.3                   # Interpupillary distance from eye corner midpoints
REAL_FACE_W_CM = 14.0               # Bizygomatic outer face width across cheekbones

# ==================== 20-20-20 FSM & SNOOZE ====================
WORK_DURATION_SECONDS = 20 * 60     # 20 minutes work interval
BREAK_DURATION_SECONDS = 20.0       # 20 seconds verified eye break
SNOOZE_DURATION_SECONDS = 5 * 60    # 5 minutes snooze on [S]
MAX_SNOOZE_COUNT = 2

GAZE_AWAY_YAW_THRESH = 0.22
GAZE_AWAY_PITCH_THRESH = -0.08

# ==================== AFK, FATIGUE & AMBIENT LIGHT ====================
FACE_LOST_GRACE_FRAMES = 12
AFK_RESET_SECONDS = 120.0
PERCLOS_WINDOW_SECONDS = 60.0
PERCLOS_FATIGUE_THRESH = 28.0
LOW_LIGHT_LUX_THRESH = 42.0
