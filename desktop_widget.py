# desktop_widget.py — Sleek Always-On-Top Floating Pill & AI Ergonomics HUD
import math
import tkinter as tk


class DesktopOverlayWidget:
    """
    Sleek, Frameless, Always-On-Top Desktop Floating Pill:
    - Floats unobtrusively on top of VS Code, Chrome, Terminal, and games.
    - Draggable anywhere on screen with zero jitter (screen-root coordinate tracking).
    - Solid dark midnight styling (no transparency bleed-through or ghost artifacts).
    - Dirty-checked label updating (eliminates GDI churn, frame drops, and flicker).
    - Shows live blink timer, distance, AI Eye Strain Index (%), Optical Gaze, and 20-20-20 countdown.
    - 1-click buttons: [Cam ON/OFF], [Sound Zen/Mute], [Break], [Snooze 5m].
    """

    def __init__(self, callbacks: dict):
        self.callbacks = callbacks
        self.root = tk.Tk()
        self.root.title("BlinkCV AI Pill")
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)

        # Style tokens
        self.bg_color = "#11131f"          # Solid deep midnight slate (crisp, zero bleed)
        self.border_color = "#2b2f48"      # Sleek 1px border framing
        self.text_primary = "#f0f2f8"      # Crisp bright white-blue
        self.text_secondary = "#8e95ae"    # Muted subtle label
        self.accent_green = "#00e676"      # Fresh green
        self.accent_amber = "#ff9100"      # Warning amber
        self.accent_blue = "#4d94ff"       # Vibrant modern blue
        self.accent_red = "#ff4d6d"        # Alert crimson

        # Dimensions & Default Position (Top-Right of screen)
        self.width = 345
        self.height = 110
        screen_w = self.root.winfo_screenwidth()
        self.x = max(10, screen_w - self.width - 24)
        self.y = 20
        self.root.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")
        self.root.configure(bg=self.border_color)

        # Mouse Drag State (Screen Root Coordinates — Zero Jitter / Zero Drift)
        self._is_dragging = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._win_start_x = self.x
        self._win_start_y = self.y

        # State / Widget caching to eliminate Tkinter redraw churn
        self._label_cache = {}
        self._dot_fill = self.accent_green
        self._current_border = self.border_color
        self.pulse_phase = 0.0

        # Main Container Frame
        self.container = tk.Frame(self.root, bg=self.bg_color, padx=10, pady=6)
        self.container.pack(fill="both", expand=True, padx=1, pady=1)

        # Row 1: Status Dot, App Title, Camera Toggle & Sound Toggle
        self.top_row = tk.Frame(self.container, bg=self.bg_color)
        self.top_row.pack(fill="x", expand=False)

        self.dot_canvas = tk.Canvas(self.top_row, width=10, height=10, bg=self.bg_color, highlightthickness=0)
        self.dot_canvas.pack(side="left", padx=(0, 6))
        self.dot_id = self.dot_canvas.create_oval(1, 1, 9, 9, fill=self.accent_green, outline="")

        self.lbl_title = tk.Label(
            self.top_row,
            text="BLINKCV AI",
            font=("Segoe UI", 8, "bold"),
            fg=self.text_secondary,
            bg=self.bg_color,
            cursor="fleur",
        )
        self.lbl_title.pack(side="left")

        # Top Buttons (Pack from right to left)
        self.btn_audio = tk.Label(
            self.top_row,
            text="🔔 Zen Sound",
            font=("Segoe UI", 7, "bold"),
            fg=self.accent_green,
            bg="#18281e",
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.btn_audio.pack(side="right", padx=(4, 0))
        self.btn_audio.bind("<Button-1>", lambda e: self._handle_btn_click("toggle_audio"))

        self.btn_camera = tk.Label(
            self.top_row,
            text="Cam ON",
            font=("Segoe UI", 7, "bold"),
            fg="#70b4ff",
            bg="#182436",
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.btn_camera.pack(side="right", padx=(4, 2))
        self.btn_camera.bind("<Button-1>", lambda e: self._handle_btn_click("toggle_camera"))

        # Row 2: Live Blink Status & Distance
        self.mid_row = tk.Frame(self.container, bg=self.bg_color)
        self.mid_row.pack(fill="x", expand=True, pady=(3, 1))

        self.lbl_blink = tk.Label(
            self.mid_row,
            text="Blink: 0.0s",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.bg_color,
        )
        self.lbl_blink.pack(side="left")

        self.lbl_dist = tk.Label(
            self.mid_row,
            text="57cm • Good",
            font=("Segoe UI", 8, "bold"),
            fg=self.accent_green,
            bg=self.bg_color,
        )
        self.lbl_dist.pack(side="right")

        # Row 3: AI Cognitive Strain Index & Optical Gaze Zone
        self.ai_row = tk.Frame(self.container, bg=self.bg_color)
        self.ai_row.pack(fill="x", expand=True, pady=(1, 2))

        self.lbl_ai_strain = tk.Label(
            self.ai_row,
            text="AI Strain: 12% (Fresh 🟢)",
            font=("Segoe UI", 8, "bold"),
            fg="#80e27e",
            bg=self.bg_color,
        )
        self.lbl_ai_strain.pack(side="left")

        self.lbl_gaze = tk.Label(
            self.ai_row,
            text="Gaze: Screen ✓",
            font=("Segoe UI", 8),
            fg="#90caf9",
            bg=self.bg_color,
        )
        self.lbl_gaze.pack(side="right")

        # Row 4: 20-20-20 Timer & Snooze / Break Controls
        self.bot_row = tk.Frame(self.container, bg=self.bg_color)
        self.bot_row.pack(fill="x", expand=False, pady=(1, 0))

        self.lbl_break = tk.Label(
            self.bot_row,
            text="20-20-20 in 20:00",
            font=("Segoe UI", 8),
            fg=self.accent_blue,
            bg=self.bg_color,
        )
        self.lbl_break.pack(side="left")

        self.btn_snooze = tk.Label(
            self.bot_row,
            text="Snooze 5m",
            font=("Segoe UI", 7, "bold"),
            fg="#ffcc80",
            bg="#2c241e",
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.btn_snooze.pack(side="right", padx=(4, 0))
        self.btn_snooze.bind("<Button-1>", lambda e: self._handle_btn_click("snooze"))
        self.btn_snooze.bind("<Enter>", lambda e: self.btn_snooze.configure(bg="#3f3227"))
        self.btn_snooze.bind("<Leave>", lambda e: self.btn_snooze.configure(bg="#2c241e"))

        self.btn_break_now = tk.Label(
            self.bot_row,
            text="Break",
            font=("Segoe UI", 7, "bold"),
            fg="#80d8ff",
            bg="#1b2a3a",
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.btn_break_now.pack(side="right", padx=(4, 2))
        self.btn_break_now.bind("<Button-1>", lambda e: self._handle_btn_click("force_break"))
        self.btn_break_now.bind("<Enter>", lambda e: self.btn_break_now.configure(bg="#263f59"))
        self.btn_break_now.bind("<Leave>", lambda e: self.btn_break_now.configure(bg="#1b2a3a"))

        # Bind Draggable Targets (Everything except the action buttons)
        for w in (
            self.root,
            self.container,
            self.top_row,
            self.lbl_title,
            self.dot_canvas,
            self.mid_row,
            self.lbl_blink,
            self.lbl_dist,
            self.ai_row,
            self.lbl_ai_strain,
            self.lbl_gaze,
            self.bot_row,
            self.lbl_break,
        ):
            self._bind_drag(w)

        self.is_running = True
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.root.update_idletasks()

    def _bind_drag(self, widget):
        """Binds mouse dragging using absolute screen coordinates to eliminate feedback loops."""
        widget.bind("<Button-1>", self._start_drag, add="+")
        widget.bind("<B1-Motion>", self._on_drag, add="+")
        widget.bind("<ButtonRelease-1>", self._stop_drag, add="+")

    def _handle_btn_click(self, callback_name: str):
        """Executes button callback and halts event propagation so drag is never triggered."""
        cb = self.callbacks.get(callback_name, lambda: None)
        try:
            cb()
        except Exception as e:
            print(f"[Overlay] Button callback error ({callback_name}): {e}")
        return "break"

    def _start_drag(self, event):
        """Captures global mouse screen position and initial window anchor."""
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self._win_start_x = self.root.winfo_x()
        self._win_start_y = self.root.winfo_y()
        self._is_dragging = True

    def _on_drag(self, event):
        """Moves window smoothly based on screen delta and clamps to desktop boundaries."""
        if not self._is_dragging:
            return

        delta_x = event.x_root - self._drag_start_x
        delta_y = event.y_root - self._drag_start_y
        new_x = self._win_start_x + delta_x
        new_y = self._win_start_y + delta_y

        # Clamp within visible desktop screen boundaries
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.x = max(0, min(screen_w - self.width, new_x))
        self.y = max(0, min(screen_h - self.height - 35, new_y))

        # Always enforce full width & height so geometry manager never collapses
        self.root.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")

    def _stop_drag(self, event=None):
        self._is_dragging = False

    def _update_label(self, lbl: tk.Label, text: str, fg: str = None, bg: str = None, font=None):
        """Applies configuration only if attributes have actually changed (dirty check)."""
        key = id(lbl)
        cached = self._label_cache.get(key)
        new_state = (text, fg, bg, font)
        if cached != new_state:
            cfg = {"text": text}
            if fg is not None:
                cfg["fg"] = fg
            if bg is not None:
                cfg["bg"] = bg
            if font is not None:
                cfg["font"] = font
            lbl.configure(**cfg)
            self._label_cache[key] = new_state

    def _update_dot(self, fill_color: str):
        if self._dot_fill != fill_color:
            self.dot_canvas.itemconfig(self.dot_id, fill=fill_color)
            self._dot_fill = fill_color

    def _update_border(self, border_color: str):
        if self._current_border != border_color:
            self.root.configure(bg=border_color)
            self._current_border = border_color

    def update_state(self, state_dict: dict):
        """Called every frame (~30ms) to update the floating pill."""
        if not self.is_running:
            return

        time_since_blink = state_dict.get("time_since_blink", 0.0)
        dist_cm = state_dict.get("dist_cm", 56.0)
        is_too_close = state_dict.get("too_close", False)
        is_slouching = state_dict.get("is_slouching", False)
        fsm_state = state_dict.get("fsm_state", "MONITORING_NORMAL")
        work_rem = state_dict.get("work_remaining", 1200.0)
        break_rem = state_dict.get("break_remaining", 20.0)
        audio_muted = state_dict.get("audio_muted", False)
        camera_visible = state_dict.get("camera_visible", True)
        is_looking_away = state_dict.get("is_looking_away", False)

        # AI Neural Telemetry
        ai_strain = state_dict.get("ai_strain_score", 12.0)
        ai_tag = state_dict.get("ai_tag", "Fresh 🟢")
        ai_color = state_dict.get("ai_color", "#80e27e")
        gaze_zone = state_dict.get("gaze_zone", "CENTER")

        # Audio button state
        if audio_muted:
            self._update_label(self.btn_audio, text="🔇 Muted", fg="#8e95ae", bg="#1e202e")
        else:
            self._update_label(self.btn_audio, text="🔔 Zen Sound", fg=self.accent_green, bg="#18281e")

        # Camera preview button
        if camera_visible:
            self._update_label(self.btn_camera, text="Cam ON", fg="#70b4ff", bg="#182436")
        else:
            self._update_label(self.btn_camera, text="Cam OFF", fg="#8e95ae", bg="#1e202e")

        corneal_moisture = state_dict.get("corneal_moisture", 100.0)
        needs_squeeze_blink = state_dict.get("needs_squeeze_blink", False)

        # AFK / Face Lost State
        if fsm_state == "IDLE_AFK" or dist_cm <= 0.0:
            self._update_dot("#505670")
            self._update_label(self.lbl_blink, text="Blink: --", fg=self.text_secondary)
            self._update_label(self.lbl_dist, text="-- • Away", fg=self.text_secondary)
            self._update_label(self.lbl_ai_strain, text="AI Strain: -- (Away)", fg=self.text_secondary)
            self._update_label(self.lbl_gaze, text="Gaze: Away", fg=self.text_secondary)
            self._update_border(self.border_color)
        else:
            # Blink Status & Pulse Animation
            if time_since_blink >= 7.0 or corneal_moisture <= 8.0:
                self.pulse_phase += 0.25
                alpha = 0.5 + 0.5 * math.sin(self.pulse_phase)
                amber_hex = "#ff9100" if alpha > 0.5 else "#ffb74d"
                self._update_dot(amber_hex)
                self._update_label(
                    self.lbl_blink,
                    text=f"👁 BLINK ({time_since_blink:.1f}s • {corneal_moisture:.0f}%💧)",
                    fg=amber_hex,
                )
                self._update_border("#ff9100" if alpha > 0.6 else "#2b2f48")
            elif needs_squeeze_blink:
                self._update_dot("#00e5ff")
                self._update_label(
                    self.lbl_blink,
                    text=f"👁 Try Squeeze Blink (Mois:{corneal_moisture:.0f}%)",
                    fg="#00e5ff",
                )
                self._update_border(self.border_color)
            else:
                self._update_dot(self.accent_green)
                self._update_label(
                    self.lbl_blink,
                    text=f"Blink: {time_since_blink:.1f}s • {corneal_moisture:.0f}%💧",
                    fg=self.text_primary,
                )
                self._update_border(self.border_color)

            # Distance & Posture
            if is_too_close:
                self._update_label(self.lbl_dist, text=f"⚠️ {dist_cm:.0f}cm • Too Close", fg=self.accent_red)
                self._update_border(self.accent_red)
            elif is_slouching:
                self._update_label(self.lbl_dist, text="🧘 Sit Upright", fg=self.accent_amber)
            else:
                self._update_label(self.lbl_dist, text=f"{dist_cm:.0f}cm • Good", fg=self.accent_green)

            # AI Strain & Optical Gaze
            self._update_label(
                self.lbl_ai_strain,
                text=f"AI Strain: {ai_strain:.0f}% ({ai_tag})",
                fg=ai_color,
            )
            gaze_txt = "Gaze: Away" if is_looking_away else f"Gaze: {gaze_zone} ✓"
            self._update_label(self.lbl_gaze, text=gaze_txt, fg="#90caf9")

        # 20-20-20 Break State
        if fsm_state == "BREAK_PENDING":
            self._update_label(
                self.lbl_break,
                text="🌿 Look 20ft Away Now!",
                fg=self.accent_amber,
                font=("Segoe UI", 8, "bold"),
            )
        elif fsm_state == "BREAK_ACTIVE":
            status_tag = "Resting ✓" if is_looking_away else "Look Away!"
            self._update_label(
                self.lbl_break,
                text=f"🌿 Rest: {break_rem:.1f}s ({status_tag})",
                fg=self.accent_green,
                font=("Segoe UI", 8, "bold"),
            )
        else:
            rem_m = int(work_rem // 60)
            rem_s = int(work_rem % 60)
            self._update_label(
                self.lbl_break,
                text=f"20-20-20 in {rem_m:02d}:{rem_s:02d}",
                fg=self.accent_blue,
                font=("Segoe UI", 8),
            )

        try:
            self.root.update()
        except Exception:
            pass

    def destroy(self):
        self.is_running = False
        try:
            self.root.destroy()
        except Exception:
            pass
