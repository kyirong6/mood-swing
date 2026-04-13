"""
Harmony Module — Chord Map + DistilGPT2 Melody Generator

Provides two core components:
1. HarmonicMap: Y-axis (0=happy → 1=dissonant) mapped to chord anchor points
2. MelodyModel: DistilGPT2 fine-tuned on music that generates melodic phrases
   constrained to the active chord tones

The model uses a text-based note encoding:
  - Notes:  ns[pitch]s[duration]  e.g. nsC4s1p0
  - Rests:  rs[duration]          e.g. rs0p5
  - Chords: cs[n]s[pitches]s[dur] e.g. cs2sE4sG4s1p0
  Special char swaps: '.' → 'p', '/' → 'q', '-' → 't'
"""

import re
import threading
import random
import os


# ---------------------------------------------------------------------------
# Note / pitch utilities
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(midi_note):
    """Convert MIDI note number to name string, e.g. 60 → 'C4'."""
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"

def name_to_midi(name):
    """Convert note name to MIDI number, e.g. 'C4' → 60. Returns None on failure."""
    match = re.match(r"^([A-Ga-g][#bwt]?)(\-?\d+)$", name)
    if not match:
        return None
    note_str = match.group(1).upper()
    octave = int(match.group(2))
    # Handle enharmonic: 'w' and 't' were used as sharp/flat in the model's vocab
    note_str = note_str.replace("W", "#").replace("T", "b")
    if note_str in NOTE_NAMES:
        return (octave + 1) * 12 + NOTE_NAMES.index(note_str)
    # Handle flats
    flat_map = {"DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#", "CB": "B"}
    if note_str in flat_map:
        return (octave + 1) * 12 + NOTE_NAMES.index(flat_map[note_str])
    return None


# ---------------------------------------------------------------------------
# Harmonic Map — Y-axis emotional chord gradient
# ---------------------------------------------------------------------------

MAJOR_SCALE_OFFSETS = [0, 2, 4, 5, 7, 9, 11]
DEFAULT_SEGMENT_KEYS = [0, 7, 2]  # C, G, D

# Each entry:
# (y_anchor, roman_label, mood, scale_degree_index, intervals_from_degree_root, quality_suffix)
# scale_degree_index is in a major scale (0=I ... 6=vii)
MOOD_TEMPLATES = [
    (0.00, "I",   "bright",      0, [0, 4, 7, 14],     "maj9"),
    (0.05, "I",   "radiant",     0, [0, 4, 7, 9, 14],  "6/9"),
    (0.10, "V",   "joyful",      4, [0, 4, 7, 14],     "add9"),
    (0.15, "IV",  "warm",        3, [0, 4, 7, 11],     "maj7"),
    (0.20, "ii",  "airy",        1, [0, 2, 7, 10],     "sus2"),
    (0.25, "I",   "open",        0, [0, 4, 7, 9],      "6"),
    (0.30, "V",   "lifted",      4, [0, 5, 7, 10],     "7sus4"),
    (0.35, "IV",  "floating",    3, [0, 4, 7, 14],     "add9"),
    (0.40, "I",   "mellow",      0, [0, 4, 7, 11],     "maj7"),
    (0.45, "V",   "driving",     4, [0, 4, 7, 10],     "7"),
    (0.50, "IV",  "dusky",       3, [0, 4, 7, 9],      "6"),
    # Minor colors intentionally begin in the lower-middle Y range.
    (0.54, "vi",  "wistful",     5, [0, 3, 7, 10],     "min7"),
    (0.58, "ii",  "reflective",  1, [0, 3, 7, 10],     "min7"),
    (0.62, "iii", "pensive",     2, [0, 3, 7, 10],     "min7"),
    (0.66, "vi",  "shadowed",    5, [0, 3, 7, 14],     "min(add9)"),
    (0.70, "ii",  "somber",      1, [0, 3, 7],         "min"),
    (0.74, "iii", "dark",        2, [0, 3, 7],         "min"),
    (0.78, "vi",  "brooding",    5, [0, 3, 7, 10, 14], "min9"),
    (0.82, "vii", "uneasy",      6, [0, 3, 6],         "dim"),
    (0.86, "vii", "tense",       6, [0, 3, 6, 9],      "dim7"),
    (0.90, "vii", "anxious",     6, [0, 3, 6, 10],     "m7b5"),
    (0.94, "V",   "volatile",    4, [0, 1, 4, 7],      "alt"),
    (0.97, "V",   "dissonant",   4, [0, 1, 3],         "cluster"),
    (1.00, "vii", "chaotic",     6, [0, 1, 3],         "cluster"),
]


class HarmonicMap:
    """
    Maps Y to mood/chord quality and X-segment to key tonic.
    The result is a key-aware chord context for generation.
    """

    NUM_SEGMENTS = 3

    def __init__(self, segment_keys=None):
        self.templates = MOOD_TEMPLATES
        self._key_lock = threading.Lock()
        if segment_keys and len(segment_keys) == self.NUM_SEGMENTS:
            self._segment_keys = [int(k) % 12 for k in segment_keys]
        else:
            self._segment_keys = list(DEFAULT_SEGMENT_KEYS)

    def get_segment_index(self, x_position):
        x = max(0.0, min(0.999999, x_position))
        return min(self.NUM_SEGMENTS - 1, int(x * self.NUM_SEGMENTS))

    def get_segment_key(self, segment_index):
        idx = max(0, min(self.NUM_SEGMENTS - 1, int(segment_index)))
        with self._key_lock:
            return self._segment_keys[idx]

    def get_segment_keys(self):
        with self._key_lock:
            keys = list(self._segment_keys)
        return [
            {"segment": i, "tonic_pc": keys[i], "key": NOTE_NAMES[keys[i]]}
            for i in range(self.NUM_SEGMENTS)
        ]

    def randomize_segment_keys(self):
        with self._key_lock:
            self._segment_keys = random.sample(range(12), self.NUM_SEGMENTS)
        return self.get_segment_keys()

    def _root_midi_for_pc(self, pitch_class):
        # Keep roots around the center register for stable generation prompts.
        root = 60 + (pitch_class % 12)
        if root > 72:
            root -= 12
        return root

    def get_chord(self, y_position, key_tonic=0, segment_index=None):
        """
        Find the nearest mood template for Y and apply it to the given key tonic.

        Returns dict:
            name: str
            mood: str
            root: int
            notes: list[int]
            intervals: list[int]
            y: float
            key: str
            tonic_pc: int
            segment: int|None
        """
        y = max(0.0, min(1.0, y_position))
        best = min(self.templates, key=lambda c: abs(c[0] - y))
        y_pos, roman, mood, degree_idx, intervals, quality = best

        tonic_pc = int(key_tonic) % 12
        root_pc = (tonic_pc + MAJOR_SCALE_OFFSETS[degree_idx]) % 12
        root = self._root_midi_for_pc(root_pc)
        notes = [root + i for i in intervals]
        chord_name = f"{NOTE_NAMES[root_pc]} {quality}"

        return {
            "name": chord_name,
            "roman": roman,
            "mood": mood,
            "root": root,
            "notes": notes,
            "intervals": intervals,
            "y": y_pos,
            "key": NOTE_NAMES[tonic_pc],
            "tonic_pc": tonic_pc,
            "segment": segment_index,
        }

    def get_all_chords(self, key_tonic=0):
        """Return chord anchors for frontend display (for a representative key)."""
        tonic_pc = int(key_tonic) % 12
        result = []
        for y_pos, roman, mood, degree_idx, intervals, quality in self.templates:
            root_pc = (tonic_pc + MAJOR_SCALE_OFFSETS[degree_idx]) % 12
            result.append({
                "y": y_pos,
                "name": f"{NOTE_NAMES[root_pc]} {quality}",
                "roman": roman,
                "mood": mood,
            })
        return result


# ---------------------------------------------------------------------------
# Model token parsing utilities
# ---------------------------------------------------------------------------

def _decode_duration(dur_str):
    """Decode model duration string back to float. 'p'→'.', 'q'→'/', 't'→'-'."""
    s = dur_str.replace("p", ".").replace("q", "/").replace("t", "-")
    try:
        if "/" in s:
            parts = s.split("/")
            return float(parts[0]) / float(parts[1])
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.5  # fallback


def _encode_duration(dur_float):
    """Encode a float duration for the model. Simple approach: use decimal with 'p' for '.'"""
    # Round to nearest 0.25
    quantized = round(dur_float * 4) / 4
    quantized = max(0.25, min(4.0, quantized))
    s = f"{quantized:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def parse_model_tokens(text):
    """
    Parse model output text into a list of musical events.

    The model outputs tokens concatenated WITHOUT spaces, e.g.:
        nsC4s1p0nsE4s0p5rs0p25cs2sG4sC4s0p5

    We split on token boundaries using regex (lookahead for ns/rs/cs prefixes).

    Returns list of dicts:
        {"type": "note", "pitch": "C4", "midi": 60, "duration": 0.5}
        {"type": "rest", "duration": 0.25}
        {"type": "chord", "pitches": ["E4","G4"], "midis": [64,67], "duration": 1.0}
    """
    events = []
    # Split on token boundaries: split before 'ns', 'rs', 'cs' prefixes
    tokens = re.split(r'(?=(?:ns|rs|cs))', text.strip())
    tokens = [t.strip() for t in tokens if t.strip()]

    for token in tokens:
        try:
            if token.startswith("ns"):
                # Note: ns[pitch]s[duration]
                parts = token[2:].split("s", 1)
                if len(parts) == 2:
                    pitch_name = parts[0]
                    midi = name_to_midi(pitch_name)
                    dur = _decode_duration(parts[1])
                    if midi is not None:
                        events.append({
                            "type": "note",
                            "pitch": pitch_name,
                            "midi": midi,
                            "duration": dur,
                        })

            elif token.startswith("rs"):
                # Rest: rs[duration]
                dur = _decode_duration(token[2:])
                events.append({"type": "rest", "duration": dur})

            elif token.startswith("cs"):
                # Chord: cs[n]s[pitch1]s[pitch2]...s[duration]
                parts = token[2:].split("s")
                if len(parts) >= 3:
                    n = int(parts[0])
                    pitches = parts[1:1+n]
                    dur_str = parts[1+n] if len(parts) > 1+n else "1p0"
                    midis = [name_to_midi(p) for p in pitches]
                    midis = [m for m in midis if m is not None]
                    if midis:
                        events.append({
                            "type": "chord",
                            "pitches": pitches,
                            "midis": midis,
                            "duration": _decode_duration(dur_str),
                        })
        except (ValueError, IndexError):
            continue

    return events


def build_seed_prompt(chord_info, num_seed_notes=3):
    """
    Build a model prompt from the current chord.
    Creates a short sequence of chord-tone notes to seed the model.
    """
    notes = chord_info["notes"]
    y = chord_info.get("y", 0.5)
    # Stronger seeding near extremes to make mood shifts obvious.
    if y < 0.2 or y > 0.8:
        num_seed_notes = max(num_seed_notes, 5)
    tokens = []
    for i in range(num_seed_notes):
        note = notes[i % len(notes)]
        # Vary octave slightly for interest
        if i > 0 and random.random() > 0.5:
            note += random.choice([-12, 12])
        note = max(36, min(96, note))  # keep in reasonable range
        name = midi_to_name(note)
        dur = random.choice(["0p5", "0p25", "1p0"])
        tokens.append(f"ns{name}s{dur}")
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# DistilGPT2 Music Model Wrapper
# ---------------------------------------------------------------------------

class MelodyModel:
    """
    Wrapper around the DancingIguana/music-generation DistilGPT2 model.
    Generates melodic phrases seeded by chord-tone prompts.
    Thread-safe: generation is serialized via a lock.
    """

    def __init__(self, model_name="DancingIguana/music-generation"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.lock = threading.Lock()
        self.loaded = False
        self._load_error = None

    @property
    def load_error(self):
        return self._load_error

    def _load_from_pretrained(self, local_files_only):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=local_files_only,
        )
        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load(self):
        """Load the model and tokenizer. Call once at startup."""
        # If model is cached, prefer local files to avoid slow/failing network calls.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        try:
            print(f"[Model] Loading {self.model_name}...")
            try:
                self._load_from_pretrained(local_files_only=True)
                print("[Model] Loaded from local cache")
            except Exception:
                self._load_from_pretrained(local_files_only=False)
                print("[Model] Loaded using remote hub access")

            self.loaded = True
            self._load_error = None
            print(f"[Model] Loaded successfully ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params)")
        except Exception as e:
            self._load_error = str(e)
            self.loaded = False
            print(f"[Model] Failed to load: {e}")
            print("[Model] Will use fallback chord-tone generation")

    def generate_phrase(self, chord_info, temperature=0.9, max_new_tokens=30):
        """
        Generate a melodic phrase that fits over the given chord.

        Args:
            chord_info: dict from HarmonicMap.get_chord()
            temperature: float 0.1-2.0 (lower=predictable, higher=wild)
            max_new_tokens: number of tokens to generate

        Returns:
            list of parsed musical events (see parse_model_tokens)
        """
        seed = build_seed_prompt(chord_info)

        if not self.loaded:
            # Fallback: return the seed notes parsed
            return parse_model_tokens(seed)

        with self.lock:
            try:
                inputs = self.tokenizer(seed, return_tensors="pt", padding=True)
                import torch
                with torch.no_grad():
                    output = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=max(0.1, min(2.0, temperature)),
                        do_sample=True,
                        top_k=40,
                        top_p=0.92,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                events = parse_model_tokens(generated_text)

                # Post-process: constrain melody to chord-compatible notes
                events = self._constrain_to_chord(events, chord_info)

                return events

            except Exception as e:
                print(f"[Model] Generation error: {e}")
                return parse_model_tokens(seed)

    def _constrain_to_chord(self, events, chord_info):
        """
        Constrain generated notes with Y-dependent harmonic strictness:
        - Top (happy): mostly consonant/chordal
        - Middle: scale-compatible
        - Bottom (dissonant): allow and bias semitone tension
        """
        chord_pcs = {note % 12 for note in chord_info["notes"]}
        root_pc = chord_info["root"] % 12
        intervals = chord_info["intervals"]
        y = chord_info.get("y", 0.5)

        # Build allowed pitch classes based on Y-axis mood.
        allowed_pcs = set(chord_pcs)
        if y < 0.45:
            # Happy/consonant: stay close to chord tones + a few stable tones.
            if 4 in intervals:
                for deg in [2, 9]:
                    allowed_pcs.add((root_pc + deg) % 12)
            elif 3 in intervals:
                for deg in [2, 5]:
                    allowed_pcs.add((root_pc + deg) % 12)
        elif y < 0.8:
            # Middle: allow full major/minor color around chord.
            if 3 in intervals:
                for deg in [0, 2, 3, 5, 7, 8, 10]:
                    allowed_pcs.add((root_pc + deg) % 12)
            elif 4 in intervals:
                for deg in [0, 2, 4, 5, 7, 9, 11]:
                    allowed_pcs.add((root_pc + deg) % 12)
        else:
            # Dark/dissonant: add semitone tension around chord tones + tritone.
            for pc in list(chord_pcs):
                allowed_pcs.add((pc + 1) % 12)
                allowed_pcs.add((pc - 1) % 12)
            allowed_pcs.add((root_pc + 6) % 12)

        def nearest_pc(pc, targets):
            best = min(targets, key=lambda cp: min(abs(cp - pc), 12 - abs(cp - pc)))
            diff = best - pc
            if abs(diff) > 6:
                diff = diff - 12 if diff > 0 else diff + 12
            return diff

        constrained = []
        for event in events:
            if event["type"] == "note":
                pc = event["midi"] % 12
                if pc not in allowed_pcs:
                    event["midi"] = max(36, min(96, event["midi"] + nearest_pc(pc, allowed_pcs)))

                # Extra bias: very top stays chordal, very bottom adds semitone rub.
                if y < 0.2 and (event["midi"] % 12) not in chord_pcs:
                    event["midi"] = max(36, min(96, event["midi"] + nearest_pc(event["midi"] % 12, chord_pcs)))
                elif y > 0.8 and random.random() < 0.35:
                    event["midi"] = max(36, min(96, event["midi"] + random.choice([-1, 1])))

                event["pitch"] = midi_to_name(event["midi"])
                constrained.append(event)
            elif event["type"] == "chord":
                adjusted = []
                for mn in event["midis"]:
                    pc = mn % 12
                    if pc not in allowed_pcs:
                        mn = max(36, min(96, mn + nearest_pc(pc, allowed_pcs)))
                    adjusted.append(mn)
                if adjusted:
                    event["midis"] = adjusted
                    event["pitches"] = [midi_to_name(m) for m in adjusted]
                constrained.append(event)
            else:
                constrained.append(event)

        return constrained
