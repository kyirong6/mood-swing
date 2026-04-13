"""
MIDI Generation Engine — Quantized Grid + Lookahead Queue

Drawing sets the harmonic context that steers the DistilGPT2 model.
The canvas does not play MIDI directly.

Architecture:
  - Quantized beat grid: all notes snap to a steady rhythmic pulse
  - Lookahead phrase queue: model always generates ahead of playback
  - Smoothed chord transitions: chord only commits after settling
  - Continuous music: no gaps between phrases

Mappings:
  - Y position → Chord/mood context (happy → dissonant)
  - X position → Key zone (3 vertical tonic segments)
  - Drawing speed → Velocity
  - Stroke density → AI generation frequency
  - Color hue → Model temperature (wildness)
  - Brush size → Note duration scaling
"""

import threading
import time
from collections import deque
import mido
from harmony import HarmonicMap, MelodyModel, midi_to_name, name_to_midi


def clamp(value, low, high):
    return max(low, min(high, value))


class DrawingState:
    """Thread-safe container for the current drawing parameters."""

    def __init__(self):
        self.lock = threading.Lock()
        self.y_position = 0.5
        self.x_position = 0.5
        self.speed = 0.0
        self.density = 0.0
        self.hue = 0.0
        self.brush_size = 0.5
        self.is_drawing = False
        self.last_update = time.time()

    def update(self, data):
        with self.lock:
            self.y_position = clamp(data.get("y", self.y_position), 0.0, 1.0)
            self.x_position = clamp(data.get("x", self.x_position), 0.0, 1.0)
            self.speed = clamp(data.get("speed", self.speed), 0.0, 1.0)
            self.density = clamp(data.get("density", self.density), 0.0, 1.0)
            self.hue = clamp(data.get("hue", self.hue), 0.0, 1.0)
            self.brush_size = clamp(data.get("brush_size", self.brush_size), 0.0, 1.0)
            self.is_drawing = data.get("is_drawing", self.is_drawing)
            self.last_update = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "y": self.y_position,
                "x": self.x_position,
                "speed": self.speed,
                "density": self.density,
                "hue": self.hue,
                "brush_size": self.brush_size,
                "is_drawing": self.is_drawing,
                "age": time.time() - self.last_update,
            }


class MIDIEngine:
    """
    AI-only MIDI generation engine with quantized playback grid.

    Architecture:
      - Context thread: smoothly tracks chord from Y position (no MIDI)
      - Model thread: generates phrases into a lookahead queue
      - Playback thread: consumes queued events on a steady beat grid

    The beat grid ensures musical timing regardless of when events arrive.
    The lookahead queue reduces silence gaps when generation spikes.
    """

    # Quantization grid: eighth-note subdivisions
    GRID_BPM = 100
    TICKS_PER_BEAT = 2  # eighth notes
    MIN_BPM = 40
    MAX_BPM = 220
    TARGET_LOOKAHEAD_SECONDS = 6.0
    MIN_LOOKAHEAD_BEATS = 8.0
    MAX_LOOKAHEAD_BEATS = 36.0
    MAX_QUEUED_PHRASES = 20
    DRONE_OCTAVE = 2
    PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    PITCH_CLASS_ALIAS = {
        "DB": "C#",
        "EB": "D#",
        "GB": "F#",
        "AB": "G#",
        "BB": "A#",
        "CB": "B",
        "FB": "E",
        "E#": "F",
        "B#": "C",
    }
    DRUM_CHANNEL = 2  # MIDI channel 3 (0-based index)
    DRUM_GROOVE_DEFAULT = "electro_pulse"
    DRUM_GROOVES = {
        "electro_pulse": {
            "label": "Electro Pulse",
            "steps": [
                [(36, 116), (42, 70), (53, 64)],
                [(42, 66)],
                [(36, 104), (39, 90), (42, 72)],
                [(42, 68), (56, 62)],
                [(36, 110), (38, 102), (42, 72)],
                [(42, 68)],
                [(36, 100), (46, 72)],
                [(42, 66), (53, 62)],
                [(36, 114), (42, 70)],
                [(42, 66), (56, 58)],
                [(36, 102), (39, 88), (42, 72)],
                [(42, 68)],
                [(36, 108), (38, 100), (42, 72), (53, 64)],
                [(42, 68)],
                [(36, 98), (46, 70)],
                [(42, 66), (56, 60)],
            ],
        },
        "boom_bap_night": {
            "label": "Boom Bap Night",
            "steps": [
                [(36, 112), (42, 62)],
                [(42, 58)],
                [(38, 104), (42, 64)],
                [(36, 84), (37, 52)],
                [(42, 60), (56, 54)],
                [(38, 60), (42, 58)],
                [(36, 104), (39, 84), (42, 62)],
                [(42, 58)],
                [(36, 114), (42, 62)],
                [(42, 58), (37, 50)],
                [(38, 106), (42, 64)],
                [(42, 58)],
                [(36, 90), (56, 56), (42, 60)],
                [(38, 62), (42, 58)],
                [(36, 106), (39, 86), (42, 62)],
                [(42, 58)],
            ],
        },
        "midnight_swing": {
            "label": "Midnight Swing",
            "steps": [
                [(36, 112), (42, 66), (53, 60)],
                [(42, 58), (37, 48)],
                [(38, 96), (46, 68)],
                [(36, 88), (42, 62)],
                [(42, 58), (56, 56)],
                [(38, 64), (42, 56)],
                [(36, 102), (42, 64)],
                [(42, 56), (53, 58)],
                [(36, 110), (42, 66)],
                [(42, 58)],
                [(38, 98), (46, 70)],
                [(36, 90), (42, 62), (56, 56)],
                [(42, 58)],
                [(38, 66), (42, 56)],
                [(36, 104), (42, 64)],
                [(42, 56), (53, 60)],
            ],
        },
        "halftime_glitch": {
            "label": "Half-Time Glitch",
            "steps": [
                [(36, 116), (42, 68)],
                [(42, 60), (37, 48)],
                [(36, 78), (56, 56)],
                [(42, 58)],
                [(38, 114), (46, 78), (39, 74)],
                [(45, 66), (53, 58)],
                [(36, 102), (42, 66)],
                [(42, 60)],
                [(36, 112), (42, 68), (53, 60)],
                [(42, 60)],
                [(36, 82), (37, 50)],
                [(42, 58), (56, 56)],
                [(38, 112), (46, 78), (39, 72)],
                [(45, 66)],
                [(36, 104), (42, 66)],
                [(42, 60), (53, 58)],
            ],
        },
    }

    def __init__(
        self,
        melody_port_name="Mood Swing Melody Out",
        drone_port_name="Mood Swing Drone Out",
        channel=0,
        drone_channel=1,
        drum_channel=DRUM_CHANNEL,
    ):
        self.melody_port_name = melody_port_name
        self.drone_port_name = drone_port_name
        self.channel = channel
        self.drone_channel = clamp(int(drone_channel), 0, 15)
        self.drum_channel = clamp(int(drum_channel), 0, 15)
        self.drawing_state = DrawingState()
        self.running = False
        self.melody_port = None
        self.drone_port = None

        # Harmonic context (smoothed — NOT played directly)
        self.harmonic_map = HarmonicMap()
        self.current_chord = None
        self._pending_chord = None
        self._pending_chord_since = 0.0
        self._chord_settle_time = 0.4  # seconds before committing a chord change

        # Phrase queue: playback consumes while model continuously fills ahead.
        self.queue_lock = threading.Lock()
        self.phrase_queue = deque()  # items: {"events": [...], "beats": float}

        # Active MIDI notes for note-off tracking
        self.active_melody_notes = []  # (note, end_time)

        # Drone state (sustained tone on its own channel)
        self.drone_lock = threading.Lock()
        self.drone_note = None
        self.drone_velocity = 72

        # Drum state (one-shot hits on a fixed channel, tick-aligned)
        self.drum_lock = threading.Lock()
        self.drum_enabled = False
        self.drum_groove = self.DRUM_GROOVE_DEFAULT
        self.active_drum_notes = []  # (note, end_time)

        # Tempo (mutable at runtime)
        self.bpm_lock = threading.Lock()
        self.grid_bpm = float(self.GRID_BPM)

        # Melody model
        self.melody_model = MelodyModel()

        # Threads
        self.context_thread = None
        self.playback_thread = None
        self.model_thread = None

        # Callbacks for frontend updates
        self.on_chord_change = None
        self.on_key_map_change = None
        self.on_model_phrase = None
        self.on_bpm_change = None
        self.on_drone_change = None
        self.on_drum_change = None

    def get_bpm(self):
        with self.bpm_lock:
            return int(self.grid_bpm)

    def set_bpm(self, bpm):
        """Set playback BPM at runtime."""
        try:
            bpm_val = int(round(float(bpm)))
        except (TypeError, ValueError):
            return self.get_bpm()

        bpm_val = clamp(bpm_val, self.MIN_BPM, self.MAX_BPM)
        changed = False
        with self.bpm_lock:
            if bpm_val != int(self.grid_bpm):
                self.grid_bpm = float(bpm_val)
                changed = True

        if changed:
            print(f"[Tempo] BPM set to {bpm_val}")
            if self.on_bpm_change:
                self.on_bpm_change({"bpm": bpm_val})
        return bpm_val

    def _get_tick_duration(self):
        return 60.0 / (self.get_bpm() * self.TICKS_PER_BEAT)

    def _target_lookahead_beats(self):
        bpm = float(self.get_bpm())
        beats = (self.TARGET_LOOKAHEAD_SECONDS * bpm) / 60.0
        return clamp(beats, self.MIN_LOOKAHEAD_BEATS, self.MAX_LOOKAHEAD_BEATS)

    def get_key_map(self):
        return self.harmonic_map.get_segment_keys()

    def randomize_key_map(self):
        key_map = self.harmonic_map.randomize_segment_keys()
        with self.queue_lock:
            self.phrase_queue.clear()
        self.current_chord = None
        self._pending_chord = None
        self._pending_chord_since = 0.0
        print("[Harmony] Key zones randomized:", ", ".join(k["key"] for k in key_map))
        if self.on_key_map_change:
            self.on_key_map_change({"key_map": key_map})
        return key_map

    def open_ports(self):
        """Open virtual MIDI output ports for melody and drone."""
        melody_ok = False
        drone_ok = False

        try:
            self.melody_port = mido.open_output(self.melody_port_name, virtual=True)
            melody_ok = True
            print(f"[MIDI] Melody port '{self.melody_port_name}' opened successfully")
        except Exception as e:
            print(f"[MIDI] Could not open melody port '{self.melody_port_name}': {e}")
            self.melody_port = None

        try:
            self.drone_port = mido.open_output(self.drone_port_name, virtual=True)
            drone_ok = True
            print(f"[MIDI] Drone port '{self.drone_port_name}' opened successfully")
        except Exception as e:
            print(f"[MIDI] Could not open drone port '{self.drone_port_name}': {e}")
            self.drone_port = None

        if not melody_ok and not drone_ok:
            print("[MIDI] Falling back to logging mode (no actual MIDI output)")
        elif melody_ok and not drone_ok:
            print("[MIDI] Drone output unavailable; drone messages will be silent")
        elif not melody_ok and drone_ok:
            print("[MIDI] Melody output unavailable; melody messages will be silent")

        return melody_ok or drone_ok

    def start(self):
        """Start all generation threads."""
        if self.running:
            return
        self.open_ports()
        self.running = True

        self.context_thread = threading.Thread(target=self._context_loop, daemon=True)
        self.context_thread.start()

        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

        self.model_thread = threading.Thread(target=self._model_loop, daemon=True)
        self.model_thread.start()
        print("[MIDI] Engine started (context + playback + model threads)")

    def stop(self):
        """Stop all threads and clean up."""
        self.running = False
        for t in [self.context_thread, self.playback_thread, self.model_thread]:
            if t:
                t.join(timeout=2.0)
        self._all_notes_off()
        if self.melody_port:
            self.melody_port.close()
        if self.drone_port:
            self.drone_port.close()
        print("[MIDI] Engine stopped")

    def update_drawing(self, data):
        """Update drawing state from frontend data."""
        self.drawing_state.update(data)

    def _send(self, msg):
        """Send melody/control MIDI message to melody port."""
        if self.melody_port:
            self.melody_port.send(msg)

    def _send_drone(self, msg):
        """Send drone MIDI message to drone port."""
        if self.drone_port:
            self.drone_port.send(msg)

    def _pitch_to_midi(self, pitch, octave=None):
        """Convert pitch input (0-11 or note name) to an absolute MIDI note."""
        target_octave = self.DRONE_OCTAVE if octave is None else int(octave)
        target_octave = clamp(target_octave, -1, 9)

        if isinstance(pitch, (int, float)):
            pitch_class = int(pitch) % 12
            return clamp((target_octave + 1) * 12 + pitch_class, 0, 127)

        if isinstance(pitch, str):
            token = pitch.strip()
            if not token:
                return None

            # Full note name path: C3, F#2, Bb4, etc.
            maybe_midi = name_to_midi(token)
            if maybe_midi is not None:
                return clamp(maybe_midi, 0, 127)

            # Pitch-class path: C, C#, Db...
            normalized = token.upper().replace("♯", "#").replace("♭", "B")
            normalized = self.PITCH_CLASS_ALIAS.get(normalized, normalized)
            if normalized in self.PITCH_CLASS_NAMES:
                pitch_class = self.PITCH_CLASS_NAMES.index(normalized)
                return clamp((target_octave + 1) * 12 + pitch_class, 0, 127)

        return None

    def get_drone_state(self):
        with self.drone_lock:
            if self.drone_note is None:
                return {
                    "active": False,
                    "midi": None,
                    "pitch": None,
                    "octave": self.DRONE_OCTAVE,
                    "channel": self.drone_channel + 1,
                }
            return {
                "active": True,
                "midi": int(self.drone_note),
                "pitch": midi_to_name(self.drone_note),
                "pitch_class": self.PITCH_CLASS_NAMES[self.drone_note % 12],
                "octave": (self.drone_note // 12) - 1,
                "channel": self.drone_channel + 1,
            }

    def _emit_drone_change(self):
        if self.on_drone_change:
            self.on_drone_change(self.get_drone_state())

    def set_drone(self, pitch, octave=None, velocity=72):
        """Start/update sustained drone note on the drone channel."""
        midi_note = self._pitch_to_midi(pitch, octave=octave)
        if midi_note is None:
            return self.get_drone_state()

        vel = clamp(int(velocity), 1, 127)
        changed = False
        unchanged = False
        with self.drone_lock:
            if self.drone_note == midi_note and self.drone_velocity == vel:
                unchanged = True
            else:
                if self.drone_note is not None:
                    self._send_drone(mido.Message("note_off", note=self.drone_note, velocity=0, channel=self.drone_channel))
                self._send_drone(mido.Message("note_on", note=midi_note, velocity=vel, channel=self.drone_channel))
                self.drone_note = midi_note
                self.drone_velocity = vel
                changed = True

        if unchanged:
            return self.get_drone_state()

        if changed:
            print(f"[Drone] ON {midi_to_name(midi_note)} (Ch {self.drone_channel + 1})")
            self._emit_drone_change()
        return self.get_drone_state()

    def stop_drone(self, emit_update=True):
        """Stop sustained drone note if active."""
        stopped_note = None
        with self.drone_lock:
            if self.drone_note is not None:
                stopped_note = self.drone_note
                self._send_drone(mido.Message("note_off", note=self.drone_note, velocity=0, channel=self.drone_channel))
                self.drone_note = None

        if stopped_note is not None:
            print(f"[Drone] OFF {midi_to_name(stopped_note)}")
            if emit_update:
                self._emit_drone_change()
        return self.get_drone_state()

    def _coerce_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}
        return bool(value)

    def get_drum_grooves(self):
        return [
            {"id": groove_id, "label": data["label"]}
            for groove_id, data in self.DRUM_GROOVES.items()
        ]

    def get_drum_state(self):
        with self.drum_lock:
            groove = self.drum_groove
            enabled = self.drum_enabled
        groove_info = self.DRUM_GROOVES.get(groove, self.DRUM_GROOVES[self.DRUM_GROOVE_DEFAULT])
        return {
            "enabled": enabled,
            "groove": groove,
            "groove_label": groove_info["label"],
            "channel": self.drum_channel + 1,
        }

    def _emit_drum_change(self):
        if self.on_drum_change:
            self.on_drum_change(self.get_drum_state())

    def set_drum_enabled(self, enabled):
        target = self._coerce_bool(enabled)
        changed = False
        disabled_now = False
        with self.drum_lock:
            if target != self.drum_enabled:
                self.drum_enabled = target
                changed = True
                disabled_now = not target
        if disabled_now:
            self._all_drum_notes_off()
        if changed:
            print(f"[Drums] {'ON' if target else 'OFF'} (Ch {self.drum_channel + 1})")
            self._emit_drum_change()
        return self.get_drum_state()

    def set_drum_groove(self, groove):
        if not isinstance(groove, str):
            return self.get_drum_state()
        groove_key = groove.strip()
        if groove_key not in self.DRUM_GROOVES:
            return self.get_drum_state()

        changed = False
        with self.drum_lock:
            if groove_key != self.drum_groove:
                self.drum_groove = groove_key
                changed = True
        if changed:
            print(f"[Drums] Groove -> {self.DRUM_GROOVES[groove_key]['label']}")
            self._emit_drum_change()
        return self.get_drum_state()

    def _all_drum_notes_off(self):
        for note, _ in self.active_drum_notes:
            self._send(mido.Message("note_off", note=note, velocity=0, channel=self.drum_channel))
        self.active_drum_notes.clear()

    def _all_notes_off(self):
        """Silence everything."""
        for note, _ in self.active_melody_notes:
            self._send(mido.Message("note_off", note=note, velocity=0, channel=self.channel))
        self.active_melody_notes.clear()
        self._all_drum_notes_off()
        self.stop_drone(emit_update=True)

    def _estimate_phrase_beats(self, events):
        beats = 0.0
        for event in events:
            if event.get("type") in ("note", "chord", "rest"):
                try:
                    beats += max(0.25, float(event.get("duration", 0.5)))
                except (TypeError, ValueError):
                    beats += 0.5
        return max(0.5, beats)

    # -------------------------------------------------------------------
    # CONTEXT TRACKER — smoothed chord transitions
    # -------------------------------------------------------------------

    def _context_loop(self):
        """
        Track harmonic context from drawing Y position.
        Smoothed: a new chord only commits after the cursor stays in
        its region for _chord_settle_time seconds. Prevents chaotic
        chord jumps from fast cursor movement.
        """
        while self.running:
            state = self.drawing_state.snapshot()
            segment_index = self.harmonic_map.get_segment_index(state["x"])
            key_tonic = self.harmonic_map.get_segment_key(segment_index)
            candidate = self.harmonic_map.get_chord(
                state["y"],
                key_tonic=key_tonic,
                segment_index=segment_index,
            )

            # Is this a new candidate different from both current and pending?
            if self._pending_chord is None or candidate["name"] != self._pending_chord["name"]:
                self._pending_chord = candidate
                self._pending_chord_since = time.time()

            # Has the pending chord been stable long enough to commit?
            settled = (time.time() - self._pending_chord_since) >= self._chord_settle_time

            if settled and (self.current_chord is None or self._pending_chord["name"] != self.current_chord["name"]):
                self.current_chord = self._pending_chord

                if self.on_chord_change:
                    self.on_chord_change({
                        "name": self.current_chord["name"],
                        "key": self.current_chord["key"],
                        "segment": self.current_chord["segment"],
                        "mood": self.current_chord["mood"],
                        "y": self.current_chord["y"],
                        "notes": [midi_to_name(n) for n in self.current_chord["notes"]],
                    })

                print(f"[Context] → {self.current_chord['name']} in {self.current_chord['key']} ({self.current_chord['mood']})")

            time.sleep(0.05)  # 20 Hz

    # -------------------------------------------------------------------
    # MODEL GENERATION — lookahead phrase generation
    # -------------------------------------------------------------------

    def _model_loop(self):
        """
        Continuously generate phrases into a lookahead queue.
        Playback runs independently from this thread.
        """
        while self.running:
            target_lookahead_beats = self._target_lookahead_beats()
            with self.queue_lock:
                queued_beats = sum(item["beats"] for item in self.phrase_queue)
                queue_len = len(self.phrase_queue)

            # Use seconds-based runway (converted to beats) so fast BPM still has enough
            # real-time buffer. Phrase count only caps memory when runway is already healthy.
            if queued_beats >= target_lookahead_beats:
                time.sleep(0.02)
                continue
            if queue_len >= self.MAX_QUEUED_PHRASES and queued_beats >= (target_lookahead_beats * 0.7):
                time.sleep(0.03)
                continue

            state = self.drawing_state.snapshot()
            bpm = self.get_bpm()
            tempo_boost = clamp((bpm - 100) / 80.0, 0.0, 1.0)

            # While actively drawing, follow live X/Y position for stronger control.
            # Otherwise use the settled context to keep continuity.
            if state["is_drawing"] and state["age"] < 0.75:
                segment_index = self.harmonic_map.get_segment_index(state["x"])
                key_tonic = self.harmonic_map.get_segment_key(segment_index)
                chord = self.harmonic_map.get_chord(
                    state["y"],
                    key_tonic=key_tonic,
                    segment_index=segment_index,
                )
            else:
                if self.current_chord is not None:
                    chord = self.current_chord
                else:
                    mid_key = self.harmonic_map.get_segment_key(1)
                    chord = self.harmonic_map.get_chord(0.5, key_tonic=mid_key, segment_index=1)

            # Temperature from hue
            temperature = 0.5 + state["hue"] * 1.0

            # More tokens when actively drawing
            if state["is_drawing"] and state["age"] < 1.0:
                # At high BPM, keep generation chunks shorter to reduce long
                # blocking model calls that can starve playback scheduling.
                max_tokens = int(18 + state["density"] * 26 + tempo_boost * 4)
            else:
                max_tokens = int(16 + tempo_boost * 4)

            # Generate phrase
            events = self.melody_model.generate_phrase(
                chord,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )

            if not events:
                time.sleep(0.1)
                continue

            phrase_beats = self._estimate_phrase_beats(events)
            with self.queue_lock:
                self.phrase_queue.append({
                    "events": events,
                    "beats": phrase_beats,
                })
                queued_beats = sum(item["beats"] for item in self.phrase_queue)

            if self.on_model_phrase:
                self.on_model_phrase({
                    "num_events": len(events),
                    "chord": chord["name"],
                    "temperature": round(temperature, 2),
                    "queue_beats": round(queued_beats, 2),
                    "target_beats": round(target_lookahead_beats, 2),
                })

            # Keep generation aggressive when runway is shallow.
            runway_ratio = queued_beats / max(target_lookahead_beats, 0.001)
            if runway_ratio < 0.45:
                time.sleep(0.005)
            elif state["is_drawing"] and state["age"] < 1.0:
                time.sleep(0.015)
            else:
                time.sleep(0.04)

    # -------------------------------------------------------------------
    # PLAYBACK — quantized beat grid with lookahead queue
    # -------------------------------------------------------------------

    def _play_drum_tick(self, tick_index, tick_duration, state):
        with self.drum_lock:
            if not self.drum_enabled:
                return
            groove_key = self.drum_groove

        groove = self.DRUM_GROOVES.get(groove_key, self.DRUM_GROOVES[self.DRUM_GROOVE_DEFAULT])
        steps = groove["steps"]
        step_hits = steps[tick_index % len(steps)]
        if not step_hits:
            return

        density = clamp(state.get("density", 0.0), 0.0, 1.0)
        vel_scale = 0.85 + (density * 0.35)
        note_len = max(0.03, min(0.16, tick_duration * 0.8))
        now = time.time()

        for note, base_velocity in step_hits:
            velocity = clamp(int(round(base_velocity * vel_scale)), 28, 127)
            self._send(mido.Message("note_on", note=note, velocity=velocity, channel=self.drum_channel))
            self.active_drum_notes.append((note, now + note_len))

    def _build_underrun_phrase(self, chord):
        """
        Emergency phrase used when model queue underruns.
        Keeps timing and musical continuity while generation catches up.
        """
        chord_tones = list(chord["notes"])
        if not chord_tones:
            root = clamp(chord.get("root", 60), 36, 84)
            chord_tones = [root, root + 4, root + 7]
        anchor = [clamp(n, 36, 90) for n in chord_tones[:4]]
        while len(anchor) < 4:
            anchor.append(anchor[-1])
        # 4/4 bar as eighth notes (0.5 beat each)
        contour = [0, 2, 1, 2, 0, 3, 2, 1]
        return [
            {"type": "note", "midi": anchor[idx], "duration": 0.5, "fallback": True}
            for idx in contour
        ]

    def _playback_loop(self):
        """
        Play events from the phrase queue on a quantized beat grid.
        Timing is anchored to a monotonic clock to reduce jitter/drift.
        """
        current_events = []
        event_index = 0
        # How many grid ticks to wait before playing the next event
        wait_ticks = 0
        tick_index = 0
        next_tick = time.monotonic()

        while self.running:
            tick_duration = self._get_tick_duration()
            now = time.monotonic()

            # Monotonic clock discipline: avoid cumulative drift.
            if now < next_tick:
                time.sleep(next_tick - now)
            elif now - next_tick > (tick_duration * 10.0):
                # Very large stalls (CPU pause, debugger halt, etc): soft realign.
                # Small/medium lag is handled by fast catch-up ticks without skipping.
                next_tick = now

            # --- Expire old notes ---
            self._expire_melody_notes()
            self._expire_drum_notes()
            state = self.drawing_state.snapshot()
            self._play_drum_tick(tick_index, tick_duration, state)

            # --- If we have no events, pull next phrase from queue ---
            if event_index >= len(current_events):
                got_phrase = False
                with self.queue_lock:
                    if self.phrase_queue:
                        phrase = self.phrase_queue.popleft()
                        current_events = phrase["events"]
                        got_phrase = True

                if got_phrase:
                    event_index = 0
                    wait_ticks = 0
                else:
                    # Queue underrun fallback: keep a full in-time bar running.
                    chord = self.current_chord
                    if chord is None:
                        mid_key = self.harmonic_map.get_segment_key(1)
                        chord = self.harmonic_map.get_chord(0.5, key_tonic=mid_key, segment_index=1)
                    current_events = self._build_underrun_phrase(chord)
                    event_index = 0
                    wait_ticks = 0

            # --- Waiting ticks (for rests / rhythmic spacing) ---
            if wait_ticks > 0:
                wait_ticks -= 1
                tick_index += 1
                next_tick += tick_duration
                continue

            # --- Play the next event ---
            if event_index < len(current_events):
                event = current_events[event_index]
                event_index += 1

                if event["type"] == "note":
                    midi_note = clamp(event["midi"], 0, 127)
                    velocity = clamp(int(50 + state["speed"] * 60), 30, 110)
                    # Duration in ticks: map model duration to grid ticks
                    dur_ticks = max(1, round(event["duration"] * self.TICKS_PER_BEAT * (0.5 + state["brush_size"])))
                    dur_seconds = dur_ticks * tick_duration

                    self._send(mido.Message("note_on", note=midi_note, velocity=velocity, channel=self.channel))
                    self.active_melody_notes.append((midi_note, time.time() + dur_seconds))

                    # Note duration drives rhythmic spacing on the quantized grid.
                    wait_ticks = max(0, dur_ticks - 1)

                elif event["type"] == "rest":
                    # Rest: wait proportional grid ticks
                    rest_ticks = max(1, round(event["duration"] * self.TICKS_PER_BEAT))
                    wait_ticks = rest_ticks - 1  # -1 because this tick counts

                elif event["type"] == "chord":
                    velocity = clamp(int(40 + state["speed"] * 50), 30, 100)
                    dur_ticks = max(1, round(event["duration"] * self.TICKS_PER_BEAT * (0.5 + state["brush_size"])))
                    dur_seconds = dur_ticks * tick_duration

                    for mn in event["midis"]:
                        mn = clamp(mn, 0, 127)
                        self._send(mido.Message("note_on", note=mn, velocity=velocity, channel=self.channel))
                        self.active_melody_notes.append((mn, time.time() + dur_seconds))

                    wait_ticks = max(0, dur_ticks - 1)

            tick_index += 1
            next_tick += tick_duration

    def _expire_melody_notes(self):
        """Turn off melody notes that have exceeded their duration."""
        now = time.time()
        still_active = []
        for note, end_time in self.active_melody_notes:
            if now >= end_time:
                self._send(mido.Message("note_off", note=note, velocity=0, channel=self.channel))
            else:
                still_active.append((note, end_time))
        self.active_melody_notes = still_active

    def _expire_drum_notes(self):
        """Turn off short drum hits after their gate time."""
        now = time.time()
        still_active = []
        for note, end_time in self.active_drum_notes:
            if now >= end_time:
                self._send(mido.Message("note_off", note=note, velocity=0, channel=self.drum_channel))
            else:
                still_active.append((note, end_time))
        self.active_drum_notes = still_active
