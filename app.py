"""
Mood Swing - Generative Music System with Drawing Interface

A real-time system where drawing acts as a mood controller for an AI
music generator. Users draw on an HTML5 canvas; the Y-axis sets the
harmonic context (happy → dissonant) that steers a DistilGPT2 model
to generate melodic phrases.

Usage:
    python app.py

Then open http://localhost:5001 in your browser and connect your DAW
to both virtual MIDI ports:
  - Mood Swing Melody Out
  - Mood Swing Drone Out
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from midi_engine import MIDIEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "moodswing-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global MIDI engine instance
engine = MIDIEngine(
    melody_port_name="Mood Swing Melody Out",
    drone_port_name="Mood Swing Drone Out",
    channel=0,
    drone_channel=1,
    drum_channel=2,
)

# ---------------------------------------------------------------------------
# Engine callbacks → WebSocket events
# ---------------------------------------------------------------------------

def on_chord_change(data):
    """Broadcast chord/mood context changes to all connected clients."""
    socketio.emit("chord_change", data)

def on_model_phrase(data):
    """Broadcast model generation events to all connected clients."""
    socketio.emit("model_phrase", data)

def on_key_map_change(data):
    """Broadcast key-segment map changes to all connected clients."""
    socketio.emit("key_map_update", data)

def on_bpm_change(data):
    """Broadcast tempo changes to all connected clients."""
    socketio.emit("bpm_update", data)

def on_drone_change(data):
    """Broadcast drone state changes to all connected clients."""
    socketio.emit("drone_update", data)

def on_drum_change(data):
    """Broadcast drum state changes to all connected clients."""
    socketio.emit("drum_update", data)

engine.on_chord_change = on_chord_change
engine.on_key_map_change = on_key_map_change
engine.on_model_phrase = on_model_phrase
engine.on_bpm_change = on_bpm_change
engine.on_drone_change = on_drone_change
engine.on_drum_change = on_drum_change

# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main drawing interface."""
    return render_template("index.html")


@app.route("/status")
def status():
    """Health check / status endpoint."""
    state = engine.drawing_state.snapshot()
    return jsonify({
        "running": engine.running,
        "melody_port": engine.melody_port_name,
        "drone_port": engine.drone_port_name,
        "melody_connected": engine.melody_port is not None,
        "drone_connected": engine.drone_port is not None,
        "key_map": engine.get_key_map(),
        "bpm": engine.get_bpm(),
        "drone": engine.get_drone_state(),
        "drum": engine.get_drum_state(),
        "drum_grooves": engine.get_drum_grooves(),
        "model_loaded": engine.melody_model.loaded,
        "model_error": engine.melody_model.load_error,
        "drawing_state": state,
        "chords": engine.harmonic_map.get_all_chords(),
    })

# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    print("[WS] Client connected")
    emit("server_status", {
        "midi_melody_port": engine.melody_port_name,
        "midi_drone_port": engine.drone_port_name,
        "midi_melody_connected": engine.melody_port is not None,
        "midi_drone_connected": engine.drone_port is not None,
        "key_map": engine.get_key_map(),
        "bpm": engine.get_bpm(),
        "drone": engine.get_drone_state(),
        "drum": engine.get_drum_state(),
        "drum_grooves": engine.get_drum_grooves(),
        "model_loaded": engine.melody_model.loaded,
        "model_error": engine.melody_model.load_error,
        "chords": engine.harmonic_map.get_all_chords(),
    })


@socketio.on("disconnect")
def handle_disconnect():
    print("[WS] Client disconnected")


@socketio.on("drawing_data")
def handle_drawing_data(data):
    """
    Receive real-time drawing data from the frontend.

    Expected data format:
    {
        "x": float (0-1),         # horizontal position → key zone
        "y": float (0-1),         # vertical position → mood/chord context
        "speed": float (0-1),     # drawing speed → velocity
        "density": float (0-1),   # stroke density → AI generation rate
        "hue": float (0-1),       # color hue → model temperature
        "brush_size": float (0-1),# brush size → note duration
        "is_drawing": bool        # whether user is actively drawing
    }
    """
    engine.update_drawing(data)


@socketio.on("panic")
def handle_panic():
    """MIDI panic: turn off all notes."""
    engine._all_notes_off()
    emit("panic_done")


@socketio.on("set_bpm")
def handle_set_bpm(data):
    """Update engine tempo from the frontend."""
    if not isinstance(data, dict):
        return
    engine.set_bpm(data.get("bpm"))


@socketio.on("randomize_keys")
def handle_randomize_keys():
    """Randomize tonic keys assigned to the three X-axis zones."""
    engine.randomize_key_map()


@socketio.on("set_drone")
def handle_set_drone(data):
    """Set drone pitch from the frontend."""
    if not isinstance(data, dict):
        return
    pitch = data.get("pitch")
    octave = data.get("octave")
    velocity = data.get("velocity", 72)
    engine.set_drone(pitch=pitch, octave=octave, velocity=velocity)


@socketio.on("stop_drone")
def handle_stop_drone():
    """Stop sustained drone note."""
    engine.stop_drone()


@socketio.on("set_drum_enabled")
def handle_set_drum_enabled(data):
    """Enable/disable drum groove playback."""
    if not isinstance(data, dict):
        return
    engine.set_drum_enabled(data.get("enabled"))


@socketio.on("set_drum_groove")
def handle_set_drum_groove(data):
    """Select active drum groove style."""
    if not isinstance(data, dict):
        return
    engine.set_drum_groove(data.get("groove"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Mood Swing — Mood Controller + AI Melody")
    print("=" * 60)
    print()

    # Load the DistilGPT2 model
    engine.melody_model.load()

    # Start the MIDI engine
    engine.start()

    print()
    print("  Open http://localhost:5001 in your browser")
    print(f"  Connect melody track to MIDI port: '{engine.melody_port_name}'")
    print(f"  Connect drone track to MIDI port:  '{engine.drone_port_name}'")
    print("  Drawing controls mood → AI generates music")
    print()
    print("=" * 60)

    try:
        socketio.run(app, host="0.0.0.0", port=5001, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
