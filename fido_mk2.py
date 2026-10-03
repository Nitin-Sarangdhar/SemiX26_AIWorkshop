"""
FIDO — Embodied Cognitive Operating System
================================================

Features
--------
✓ Whisper Base conversational ASR (OpenVINO, Intel iGPU)
✓ Silero Neural VAD (Continuous InputStream for gapless listening, CPU)
✓ Persistent live camera & YOLOv8n OpenVINO tracking (Intel iGPU)
✓ Live detections only (the LLM sees objects detected in the last 1 s, no long-term memory)
✓ One command at a time: while busy only "fido stop" is accepted (cancels the task), nothing is queued
✓ Param-1 Reasoning (spoken command sent verbatim, LLM picks from detected objects, Intel iGPU)
✓ Discrete motion: forward/back = 1 m, left/right = 90° turn, return = U-turn + drive back to start
✓ Fetch missions: NAVIGATE → ALIGN → GRASP → RETURN → DELIVER
✓ Live HUD (camera view, target lock, motors, odometry, gripper, mission steps, brain, in-view objects, activity log)
"""

import cv2
import math
import re
import sys
import time
import threading
import numpy as np
import sounddevice as sd
import torch
import openvino as ov
import openvino_genai as ov_genai
from collections import deque
from ultralytics import YOLO
from typing import Callable, Dict, List, Optional


# ============================================================================
# Intel Device Selection
# ============================================================================

def find_intel_igpu() -> str:
    """Returns the OpenVINO id of the Intel integrated GPU (e.g. 'GPU.0'), or 'CPU'."""
    core = ov.Core()
    for dev in core.available_devices:
        if not dev.startswith("GPU"):
            continue
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
            dev_type = str(core.get_property(dev, "DEVICE_TYPE"))
        except Exception:
            continue
        if "INTEGRATED" in dev_type.upper() and "intel" in name.lower():
            return dev
    return "CPU"

IGPU = find_intel_igpu()
print(f"[System] Intel iGPU → {IGPU} | CPU fallback enabled")


# ============================================================================
# Messaging Bus
# ============================================================================

class ROSCmd:
    def __init__(self):
        self._topics = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            self._topics.setdefault(topic, []).append(callback)

    def publish(self, topic: str, message):
        with self._lock:
            callbacks = list(self._topics.get(topic, []))
        for cb in callbacks:
            cb(message)


# ============================================================================
# Motion Structure
# ============================================================================

class Twist:
    def __init__(self, linear: float = 0.0, angular: float = 0.0):
        self.linear = linear
        self.angular = angular


# ============================================================================
# World State (The Cognitive Layer)
# ============================================================================

class WorldState:
    def __init__(self):
        self.system_state: str = "IDLE"  # IDLE, FETCHING, ERROR
        self.battery_level: int = 100

        # Task Management
        self.active_task: Optional[str] = None

        # Telemetry for the HUD
        self.mission_phase: str = "IDLE"
        self.target_label: Optional[str] = None
        self.speech_status: str = "LOADING"
        self.last_heard: str = ""
        self.last_heard_t: float = 0.0
        self.last_request: str = ""
        self.last_objects: List[str] = []  # exact object list given to the LLM
        self.last_reply: str = ""
        self.last_decision: Optional[str] = None
        self.last_brain_ms: float = 0.0


# ============================================================================
# Camera
# ============================================================================

class CameraThread(threading.Thread):
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = False
        self.cap = None

    def start_capture(self):
        print("[Camera] Initializing...")
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            self.cap.read()

        self.running = True
        self.start()
        print("[Camera] Ready.")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                self.new_frame.set()
            else:
                time.sleep(0.001)

    def get_latest(self, timeout=1.0):
        if self.new_frame.wait(timeout=timeout):
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
            self.new_frame.clear()
            return frame
        return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped.")


# ============================================================================
# Vision
# ============================================================================

class FidoVision:
    def __init__(self, device=IGPU):
        # Ultralytics targets OpenVINO devices via "intel:<device>"
        self.device = f"intel:{device.lower()}"
        print(f"[Vision] Loading YOLOv8n OpenVINO on {device}...")

        self.model = YOLO("yolov8n_openvino_model/", task="detect")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        print(f"[Vision] Warming up...")
        for _ in range(5):
            self.model.predict(dummy, device=self.device, verbose=False)
        print("[Vision] Ready.")

    def detect(self, frame):
        t0 = time.perf_counter()
        result = self.model.predict(frame, device=self.device, verbose=False)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        labels = []
        dets = []
        for box in result.boxes:
            label = self.model.names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            dets.append((label, float(box.conf[0]), (x1, y1, x2, y2)))
            if label not in labels:
                labels.append(label)

        return labels, dets, latency_ms


# ============================================================================
# Helpers
# ============================================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", normalize(text))


# ============================================================================
# Brain (Context-Aware)
# ============================================================================

class FidoBrain:
    def __init__(self, model_path="param-1-ov-int4/", device=IGPU):
        self.pipe = None
        self.device = "OFFLINE"
        for dev in dict.fromkeys([device, "CPU"]):
            print(f"[Brain] Loading Param-1 on {dev}...")
            try:
                self.pipe = ov_genai.LLMPipeline(model_path, dev)
                self._warmup()
                self.device = dev
                print("[Brain] Ready.")
                break
            except Exception as e:
                print(f"[Brain] Failed on {dev}: {e}")
                self.pipe = None

    def _generate(self, prompt, max_new_tokens=12):
        # Prompt is already in Param-1's ChatML format
        return self.pipe.generate(
            prompt, max_new_tokens=max_new_tokens, do_sample=False, apply_chat_template=False
        )

    def _warmup(self):
        warm = (
            "<|im_start|>system\nYou are a robot.<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self._generate(warm)

    def _parse_choice(self, raw, labels):
        """Maps the LLM's free-text reply onto the first object name it mentions (or 'none')."""
        reply = normalize(raw)
        hits = []
        for label in labels:
            m = re.search(rf"\b{re.escape(normalize(label))}\b", reply)
            if m:
                hits.append((m.start(), -len(label), label))
        none = re.search(r"\bnone\b", reply)
        if not hits:
            return None
        first = min(hits)
        if none and none.start() < first[0]:
            return None
        return first[2]

    def _llm_decide(self, command, labels, world_state: WorldState):
        # The spoken command goes to the LLM verbatim; it picks from what the camera has seen.
        # (Small models answer numbered lists with "1" by reflex, so it replies with the name.)
        objects = "\n".join(f"- {obj}" for obj in labels)
        prompt = f"""<|im_start|>system
You are FIDO, a home robot. Your camera currently sees these objects:
{objects}

The user gives you a spoken command. Choose the ONE object from the list above that best fulfils the command. Reply with only the object name, exactly as written in the list. If none of the objects can fulfil it, reply none.<|im_end|>
<|im_start|>user
{command}<|im_end|>
<|im_start|>assistant
"""
        t0 = time.perf_counter()
        raw = str(self._generate(prompt)).strip()
        latency = (time.perf_counter() - t0) * 1000

        choice = self._parse_choice(raw, labels)
        world_state.last_reply = raw
        world_state.last_decision = choice
        world_state.last_brain_ms = latency

        if choice is None:
            print(f"[Brain] LLM said '{raw}' → no suitable object ({latency:.0f} ms)")
            return None

        print(f"[Brain] LLM said '{raw}' → '{choice}' ({latency:.0f} ms)")
        return choice

    def decide(self, command, seen_objects, world_state: WorldState):
        world_state.last_request = command
        world_state.last_objects = list(seen_objects)
        world_state.last_reply = ""
        world_state.last_decision = None
        world_state.last_brain_ms = 0.0
        print(f"[Brain] Objects sent to LLM: {', '.join(seen_objects) or '(none)'}")

        if not seen_objects:
            print("[Brain] Scene empty.")
            return None

        if self.pipe is None:
            return None

        return self._llm_decide(command, seen_objects, world_state)


# ============================================================================
# Speech (Continuous Streaming)
# ============================================================================

class SpeechSystem:
    MODEL = "whisper-base-ov"
    DEVICE = IGPU
    JUNK = {"thank you", "thanks", "bye", "hmm", "uh", "um"}

    # Sentences must start with FIDO's name (common Whisper spellings), optionally after a greeting
    WAKE_WORDS = {"fido", "phido", "fedo", "fydo", "fideo", "feido"}
    GREETINGS = {"hey", "hi", "ok", "okay"}
    # A sentence made only of these words is a motion command, not a fetch
    MOTION_WORDS = WAKE_WORDS | GREETINGS | {
        "please", "now", "go", "turn", "move", "drive", "come", "to", "the",
        "a", "bit", "little", "forward", "forwards", "ahead", "back", "backward", "backwards",
        "left", "right", "home", "return", "u", "uturn", "around",
    }

    def __init__(self, messenger):
        self.messenger = messenger
        self.is_busy: Callable[[], bool] = lambda: False  # set by main to orchestrator.busy

        print(f"[Speech] Loading {self.MODEL} on {self.DEVICE}...")
        try:
            self.pipe = ov_genai.WhisperPipeline(self.MODEL, self.DEVICE)
        except Exception as e:
            print(f"[Speech] {self.DEVICE} failed: {e}\nFalling back to CPU...")
            self.pipe = ov_genai.WhisperPipeline(self.MODEL, "CPU")

        self.cfg = self.pipe.get_generation_config()
        self.cfg.max_new_tokens = 128
        self.prompt = (
            "Natural language robot assistant commands. Examples: "
            "fido fetch me something to drink, fido bring me something useful, "
            "fido go forward, fido turn left"
        )
        self.fs = 16000

        print("[Speech] Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )

        self.vad_chunk_samples = 512  # Silero strict requirement for 16kHz
        self.vad_chunk_s = self.vad_chunk_samples / self.fs  # ~0.032 seconds
        
        self._max_s = 10.0
        self._pre_s = 0.25
        self.silence_threshold = 0.5 
        self.max_silence_s = 1.0 
        
        print("[Speech] Ready.")

    def _record_vad(self):
        self.vad_model.reset_states()
        
        chunks = []
        speech_s = 0.0
        silence_s = 0.0

        # Continuous InputStream to prevent dropped audio
        with sd.InputStream(samplerate=self.fs, channels=1, dtype="float32", blocksize=self.vad_chunk_samples) as stream:
            while (len(chunks) * self.vad_chunk_s) < self._max_s:
                
                chunk, overflowed = stream.read(self.vad_chunk_samples)
                chunks.append(chunk)

                audio_tensor = torch.from_numpy(chunk.flatten()).float()
                confidence = self.vad_model(audio_tensor, self.fs).item()

                if confidence >= self.silence_threshold:
                    speech_s += self.vad_chunk_s
                    silence_s = 0.0 
                else:
                    if speech_s > 0:
                        silence_s += self.vad_chunk_s

                    if speech_s >= self._pre_s and silence_s >= self.max_silence_s:
                        break

        if speech_s < self._pre_s:
            return None

        return np.concatenate(chunks, axis=0)

    def _transcribe(self, audio):
        result = self.pipe.generate(audio.flatten(), config=self.cfg, initial_prompt=self.prompt)
        text = result.texts[0].lower().replace(",", " ").replace(".", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _addressed(self, tokens):
        """True if the sentence starts with "fido" (or "hey fido", "ok fido", ...)."""
        i = 1 if tokens[0] in self.GREETINGS and len(tokens) > 1 else 0
        return tokens[i] in self.WAKE_WORDS

    def _is_hallucination(self, text):
        if not text or text in self.JUNK:
            return True
        words = text.split()
        if not words:
            return True
        if max(words.count(w) for w in set(words)) > 6:
            return True
        return False

    def run_loop(self):
        print("\n[FIDO READY]")
        print("Speak naturally.\n")

        while True:
            audio = self._record_vad()
            if audio is None:
                continue

            text = self._transcribe(audio)
            if self._is_hallucination(text):
                continue

            tokens = tokenize(text)
            if not tokens:
                continue

            # Every command must start with "fido" (optionally "hey fido"); anything else is ignored
            if not self._addressed(tokens):
                continue

            # While FIDO is busy (thinking, moving or fetching) only "fido stop" is accepted:
            # it cancels the current task. Nothing is queued.
            if self.is_busy():
                if "stop" in tokens:
                    self.messenger.publish("speech_heard", text)
                    self.messenger.publish("voice_intent", "STOP")
                else:
                    print(f"[OS] Busy - ignored '{text}'. Only 'fido stop' works now.")
                continue

            # Only recognised commands are shown on the HUD ("speech_heard")
            if "stop" in tokens:
                self.messenger.publish("speech_heard", text)
                self.messenger.publish("voice_intent", "SHUTDOWN")
                break

            # Motion: only when the sentence is made of motion words alone ("fido turn left"),
            # so "i left my phone" is still a fetch request for the LLM
            if set(tokens) <= self.MOTION_WORDS:
                if "home" in tokens:
                    self.messenger.publish("speech_heard", text)
                    self.messenger.publish("robot_action", "HOME")
                    continue

                if "return" in tokens or "uturn" in tokens or re.search(r"\bu turn\b|\bturn around\b", text):
                    self.messenger.publish("speech_heard", text)
                    self.messenger.publish("robot_action", "RETURN")
                    continue

                cmd = Twist()
                if "forward" in tokens:
                    cmd.linear = 1.0
                elif "back" in tokens:
                    cmd.linear = -1.0
                elif "left" in tokens:
                    cmd.angular = 0.5
                elif "right" in tokens:
                    cmd.angular = -0.5
                else:
                    continue

                self.messenger.publish("speech_heard", text)
                self.messenger.publish("cmd_vel", cmd)
                continue

            # Everything else is a fetch request: the full spoken command goes to the
            # brain unchanged and the LLM picks the object
            print(f"[Speech] Fetch request → {text}")
            self.messenger.publish("speech_heard", text)
            self.messenger.publish("voice_intent", f"FETCH_REQUEST|{text}")


# ============================================================================
# Robot Hardware (Maneuver Executor)
# ============================================================================

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class RobotHardware(threading.Thread):
    WHEEL_BASE = 0.5      # metres between wheels (differential drive)
    TURN_SPEED = 1.2      # rad/s while turning
    DRIVE_SPEED = 0.6     # m/s while driving
    STEP_M = 1.0          # distance for one "forward" / "back" command
    APPROACH_M = 1.2      # simulated distance to a fetch target
    HOME_HEADING = math.pi / 2  # orientation at start (facing "up" on the map)
    TICK = 0.05           # 20Hz motor loop

    FETCH_STEPS = ("NAVIGATE", "ALIGN", "GRASP", "RETURN", "DELIVER")
    STEP_OF = {"NAVIGATING": "NAVIGATE", "ALIGNING": "ALIGN", "GRASPING": "GRASP",
               "RETURNING": "RETURN", "DELIVERING": "DELIVER"}

    def __init__(self, messenger):
        super().__init__(daemon=True)
        self.current_twist = Twist()   # what the motors are doing right now

        # Dead-reckoned pose (start point = home)
        self.x = 0.0
        self.y = 0.0
        self.theta = self.HOME_HEADING
        self.trail = deque(maxlen=600)

        # Telemetry for the HUD
        self.motor_phase = "STANDBY"
        self.fetch_step: Optional[str] = None
        self.fetch_target: Optional[str] = None
        self.gripper = "OPEN"
        self.seg_progress = 0.0

        # Plan = segments (label, kind, value) of the ONE current command, run in order
        self._plan = deque()
        self._lock = threading.Lock()
        self._busy = False    # True from the moment a command is accepted until the motors are idle
        self._abort = False   # set by stop()

        messenger.subscribe("cmd_vel", self._update_twist)
        messenger.subscribe("robot_action", self.handle_action)

        self.start()

    def queued(self) -> int:
        with self._lock:
            return len(self._plan)

    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def _enqueue(self, segments):
        with self._lock:
            self._plan.extend(segments)
            self._busy = True

    def stop(self):
        """Emergency stop: drop the rest of the current command and halt the motors."""
        with self._lock:
            pending = [value for _, kind, value in self._plan if kind == "signal"]
            self._plan.clear()
            self._abort = True
        for done in pending:
            done.set()  # release anyone waiting on a fetch
        print("[Robot] STOPPED - motors halted")

    def _update_twist(self, twist):
        """Voice moves are discrete: 1 m forward/back, or a 90° turn, then stop and wait."""
        if twist.linear:
            d = math.copysign(self.STEP_M, twist.linear)
            label = "DRIVING FORWARD" if d > 0 else "REVERSING"
            print(f"[Robot] {label} {abs(d):.1f} m")
            self._enqueue([(label, "drive", d)])
        elif twist.angular:
            a = math.copysign(math.pi / 2, twist.angular)
            label = "TURNING LEFT" if a > 0 else "TURNING RIGHT"
            print(f"[Robot] {label} 90°")
            self._enqueue([(label, "turn", a)])

    def handle_action(self, action):
        if action == "RETURN":
            print("[Robot] U-TURN 180° and returning to start")
            self._enqueue([("U-TURN", "turn", math.pi), ("RETURNING HOME", "home", None)])
        elif action == "HOME":
            print("[Robot] Going home and resetting pose")
            self._enqueue([("GOING HOME", "home", None), ("GOING HOME", "face", self.HOME_HEADING),
                           ("GOING HOME", "reset", None)])

    def fetch(self, target: str, bearing: float) -> threading.Event:
        """Queues a full fetch; the returned event is set once the object is delivered."""
        done = threading.Event()
        print(f"[Robot] HARDWARE STARTING FETCH SEQUENCE → {target}")
        self._enqueue([
            ("NAVIGATING", "target", target),
            ("NAVIGATING", "turn", bearing),
            ("NAVIGATING", "drive", self.APPROACH_M),
            ("ALIGNING", "wait", 0.8),
            ("GRASPING", "grip", "CLOSING"),
            ("GRASPING", "wait", 1.5),
            ("GRASPING", "grip", f"HOLDING {target.upper()}"),
            ("RETURNING", "turn", math.pi),
            ("RETURNING", "home", None),
            ("RETURNING", "face", self.HOME_HEADING),
            ("RETURNING", "reset", None),
            ("DELIVERING", "grip", "RELEASING"),
            ("DELIVERING", "wait", 1.0),
            ("DELIVERING", "grip", "OPEN"),
            ("DELIVERING", "signal", done),
        ])
        return done

    def wheel_speeds(self):
        t = self.current_twist
        half = t.angular * self.WHEEL_BASE / 2
        return t.linear - half, t.linear + half

    def _next_segment(self):
        with self._lock:
            return self._plan.popleft() if self._plan else None

    def run(self):
        dt = self.TICK
        seg, remaining, total, sign = None, 0.0, 1.0, 1.0
        while True:
            if self._abort:
                with self._lock:
                    self._abort = False
                seg = None
                self.fetch_target = None

            if seg is None:
                seg = self._next_segment()
                if seg is None:
                    self.current_twist = Twist()
                    self.motor_phase = "STANDBY"
                    self.fetch_step = None
                    self.seg_progress = 0.0
                    with self._lock:
                        if not self._plan:
                            self._busy = False
                    time.sleep(dt)
                    continue

                label, kind, value = seg
                if label != self.motor_phase and label in self.STEP_OF:
                    print(f"[Robot] {label}...")
                self.motor_phase = label
                self.fetch_step = self.STEP_OF.get(label)

                # Instant segments
                if kind == "target":
                    self.fetch_target = value
                    seg = None
                    continue
                if kind == "grip":
                    self.gripper = value
                    seg = None
                    continue
                if kind == "signal":
                    self.fetch_target = None
                    value.set()
                    seg = None
                    continue
                if kind == "home":
                    # Expand into "face home" + "drive home" from wherever we are now
                    dist = math.hypot(self.x, self.y)
                    if dist > 0.02:
                        turn = wrap_angle(math.atan2(-self.y, -self.x) - self.theta)
                        with self._lock:
                            self._plan.appendleft((label, "drive", dist))
                            self._plan.appendleft((label, "turn", turn))
                    seg = None
                    continue
                if kind == "face":
                    # Turn (shortest way) to an absolute heading
                    turn = wrap_angle(value - self.theta)
                    if abs(turn) > 1e-3:
                        with self._lock:
                            self._plan.appendleft((label, "turn", turn))
                    seg = None
                    continue
                if kind == "reset":
                    # Back at the start: same coordinates and orientation as at boot
                    self.x, self.y, self.theta = 0.0, 0.0, self.HOME_HEADING
                    self.trail.clear()
                    print("[Robot] At home - pose reset (x 0, y 0, heading 90°)")
                    seg = None
                    continue

                remaining = abs(value)
                total = max(remaining, 1e-6)
                sign = 1.0 if value >= 0 else -1.0

            label, kind, value = seg
            if kind == "turn":
                step = min(remaining, self.TURN_SPEED * dt)
                self.theta = wrap_angle(self.theta + sign * step)
                self.current_twist = Twist(0.0, sign * self.TURN_SPEED)
            elif kind == "drive":
                step = min(remaining, self.DRIVE_SPEED * dt)
                self.x += sign * step * math.cos(self.theta)
                self.y += sign * step * math.sin(self.theta)
                self.trail.append((self.x, self.y))
                self.current_twist = Twist(sign * self.DRIVE_SPEED, 0.0)
            else:  # wait
                step = dt
                self.current_twist = Twist()

            remaining -= step
            self.seg_progress = 1.0 - max(remaining, 0.0) / total
            if remaining <= 1e-9:
                seg = None
            time.sleep(dt)


# ============================================================================
# Cognitive Orchestrator (The Operating System)
# ============================================================================

class CognitiveOrchestrator:
    CAMERA_HFOV = math.radians(60)  # used to turn toward the target before driving
    RECENT_S = 1.0                  # objects are remembered for this long, no longer

    def __init__(self, messenger, camera, brain, robot: RobotHardware, shutdown: threading.Event):
        self.messenger = messenger
        self.camera = camera
        self.brain = brain
        self.robot = robot
        self.shutdown = shutdown
        self.vision = FidoVision(device=IGPU)

        # Central World State
        self.world = WorldState()

        # Latest perception snapshot for the HUD: (frame, detections, latency_ms)
        self.view_lock = threading.Lock()
        self.view = (None, [], 0.0)
        self.vision_fps = 0.0

        # Short-term record of what was detected in the last RECENT_S seconds:
        # label -> (last_seen, count in that frame, best confidence, box centre x 0..1)
        self.recent: Dict[str, tuple] = {}

        self.cancel = threading.Event()  # set by "fido stop" during a mission

        self.messenger.subscribe("voice_intent", self.handle_intent)
        self.messenger.subscribe("speech_heard", self._on_heard)

        # Perception runs at camera rate
        self.perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self.perception_thread.start()

    def _on_heard(self, text: str):
        self.world.last_heard = text
        self.world.last_heard_t = time.time()

    def busy(self) -> bool:
        """True while a mission runs (thinking or fetching) or the motors execute a command."""
        return self.world.system_state != "IDLE" or self.robot.busy()

    def handle_intent(self, intent: str):
        if intent == "SHUTDOWN":
            print("[Orchestrator] OS Shutting down...")
            self.shutdown.set()
            return

        if intent == "STOP":
            print("[OS] STOP - cancelling current task")
            self.cancel.set()
            self.robot.stop()
            return

        if "FETCH_REQUEST" in intent:
            desc = intent.split("|", 1)[1]

            # One command at a time - nothing is ever queued
            if self.busy():
                print(f"[OS] Busy - ignored '{desc}'. Only 'fido stop' works now.")
                return

            if self.world.battery_level < 5:
                print("[Orchestrator] REJECTED: Battery too low for fetch mission.")
                return

            self.cancel.clear()
            self.world.active_task = desc
            self.world.system_state = "FETCHING"
            threading.Thread(target=self._execute_fetch_mission, args=(desc,), daemon=True).start()

    def _perception_loop(self):
        """Runs detection on every new camera frame and keeps a RECENT_S-second record."""
        last = time.perf_counter()
        while not self.shutdown.is_set():
            frame = self.camera.get_latest(timeout=0.1)
            if frame is None:
                continue
            _, dets, ms = self.vision.detect(frame)

            now_t = time.time()
            fw = frame.shape[1]
            seen: Dict[str, tuple] = {}
            for label, conf, (x1, _, x2, _) in dets:
                n, best, cx = seen.get(label, (0, -1.0, 0.5))
                if conf > best:
                    best, cx = conf, (x1 + x2) / 2 / fw
                seen[label] = (n + 1, best, cx)

            with self.view_lock:
                self.view = (frame, dets, ms)
                for label, (n, conf, cx) in seen.items():
                    self.recent[label] = (now_t, n, conf, cx)
                for label in [l for l, v in self.recent.items() if now_t - v[0] > self.RECENT_S]:
                    del self.recent[label]

            now = time.perf_counter()
            self.vision_fps = 0.9 * self.vision_fps + 0.1 / max(now - last, 1e-3)
            last = now

    def recent_objects(self) -> Dict[str, tuple]:
        """Snapshot of objects detected within the last RECENT_S seconds."""
        now_t = time.time()
        with self.view_lock:
            return {l: v for l, v in self.recent.items() if now_t - v[0] <= self.RECENT_S}

    def _live_labels(self) -> List[str]:
        """Unique labels detected in the last RECENT_S seconds - what the LLM may choose from."""
        return list(self.recent_objects())

    def _bearing_of(self, label: str) -> float:
        """Angle (rad, left positive) to where `label` was last seen, 0 if not in the record."""
        entry = self.recent_objects().get(label)
        if entry is None:
            return 0.0
        return (0.5 - entry[3]) * self.CAMERA_HFOV

    def _execute_fetch_mission(self, description: str):
        print(f"\n[OS] Starting Mission: {description}")
        t0 = time.perf_counter()

        try:
            # Only objects detected in the last RECENT_S (1) second, no long-term memory
            self.world.mission_phase = "LOOKING"
            known_objects = self._live_labels()

            if not known_objects:
                print("[OS] Nothing in view right now. Waiting for detections...")
                self.world.mission_phase = "SCANNING"
                deadline = time.time() + 2.0
                while not known_objects and time.time() < deadline:
                    time.sleep(0.1)
                    known_objects = self._live_labels()

            if not known_objects:
                print("[OS] Mission Failed: No objects in view.")
                self.world.mission_phase = "FAILED: NOTHING IN VIEW"
                return

            # Feed the spoken command + what the camera sees into the Brain
            self.world.mission_phase = "THINKING"
            decision = self.brain.decide(description, known_objects, self.world)

            if self.cancel.is_set():
                print("[OS] Mission cancelled by 'fido stop'.")
                self.world.mission_phase = "CANCELLED"
                return

            if decision is None:
                print(f"[OS] Mission Aborted: Param-1 chose nothing for '{description}' (in view: {', '.join(known_objects)})")
                self.world.mission_phase = "ABORTED: NO MATCH"
                return

            print(f"[OS] Target Acquired: {decision.upper()} ({(time.perf_counter() - t0)*1000:.0f} ms)")
            self.world.target_label = decision
            self.world.mission_phase = f"FETCHING {decision.upper()}"
            done = self.robot.fetch(decision, self._bearing_of(decision))

            while not done.wait(0.1):
                if self.shutdown.is_set():
                    return

            if self.cancel.is_set():
                print(f"[OS] Fetch of {decision} cancelled by 'fido stop'.")
                self.world.mission_phase = "CANCELLED"
                return

            print(f"[OS] Delivered {decision} ({time.perf_counter() - t0:.1f} s)")
            self.world.mission_phase = f"DELIVERED {decision.upper()}"

        except Exception as e:
            print(f"[OS] FATAL MISSION ERROR: {e}")
            self.world.mission_phase = "ERROR"
        finally:
            print("[OS] Mission Concluded. Returning to IDLE.")
            self.world.active_task = None
            self.world.target_label = None
            self.world.system_state = "IDLE"


# ============================================================================
# Activity Log (stdout tee for the HUD)
# ============================================================================

class HudLog:
    """Tees stdout so tagged console lines ([OS], [Robot], ...) also show in the HUD."""

    TAGS = {"Speech", "OS", "Robot", "Brain", "Orchestrator"}

    def __init__(self, stream, keep: int = 30):
        self._out = stream
        self._buf = ""
        self._lock = threading.Lock()
        self._lines = deque(maxlen=keep)

    def write(self, s):
        self._out.write(s)
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                m = re.match(r"\s*\[(\w+)\]\s*(.*)", line)
                if m and m.group(1) in self.TAGS and m.group(2).strip():
                    self._lines.append((time.time(), m.group(1), m.group(2).strip()))
        return len(s)

    def flush(self):
        self._out.flush()

    def recent(self, n):
        with self._lock:
            return list(self._lines)[-n:]

    def __getattr__(self, name):
        return getattr(self._out, name)


# ============================================================================
# HUD (Visualization)
# ============================================================================

class FidoHUD:
    WINDOW = "FIDO // Cognitive OS"
    W, H = 1120, 848
    CAM = (16, 56, 640, 480)  # x, y, w, h
    RX = 672                  # x of the right-hand column

    # BGR palette
    BG = (20, 16, 12)
    PANEL = (38, 31, 24)
    EDGE = (92, 76, 54)
    ACCENT = (235, 205, 70)
    TEXT = (232, 232, 232)
    DIM = (150, 145, 140)
    GOOD = (120, 215, 95)
    WARN = (60, 185, 250)
    BAD = (85, 85, 240)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    TAG_COLORS = {"Speech": (235, 205, 70), "OS": (232, 232, 232), "Robot": (60, 185, 250),
                  "Brain": (120, 215, 95), "Orchestrator": (85, 85, 240)}

    def __init__(self, orchestrator: CognitiveOrchestrator, robot: RobotHardware, llm_device: str,
                 log: Optional[HudLog] = None):
        self.os = orchestrator
        self.robot = robot
        self.llm_device = llm_device
        self.log = log
        self.ui_fps = 0.0
        self._last = time.perf_counter()

    # ---- drawing helpers ----------------------------------------------------

    def _text(self, img, s, x, y, scale=0.45, color=None, thick=1):
        # Hershey fonts are ASCII-only
        s = s.replace("→", "->").replace("°", " deg").encode("ascii", "replace").decode()
        cv2.putText(img, s, (int(x), int(y)), self.FONT, scale, color or self.TEXT, thick, cv2.LINE_AA)

    def _panel(self, img, x, y, w, h, title=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), self.PANEL, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.EDGE, 1)
        if title:
            cv2.rectangle(img, (x, y), (x + 4, y + 22), self.ACCENT, -1)
            self._text(img, title, x + 12, y + 16, 0.45, self.ACCENT, 1)

    @staticmethod
    def _wrap(s, width):
        words, lines, cur = s.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        return lines

    def _brackets(self, img, x1, y1, x2, y2, color, thick=3, k=0.22):
        lx, ly = int((x2 - x1) * k), int((y2 - y1) * k)
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * lx, cy), color, thick, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * ly), color, thick, cv2.LINE_AA)

    # ---- sections -----------------------------------------------------------

    def _top_bar(self, img, world):
        cv2.rectangle(img, (0, 0), (self.W, 44), (28, 22, 16), -1)
        cv2.line(img, (0, 44), (self.W, 44), self.EDGE, 1)
        self._text(img, "FIDO", 16, 31, 0.9, self.ACCENT, 2)
        self._text(img, "COGNITIVE OS", 96, 30, 0.5, self.DIM, 1)

        state = world.system_state
        col = self.GOOD if state == "IDLE" else self.WARN if state == "FETCHING" else self.BAD
        cv2.rectangle(img, (230, 12), (360, 34), col, 1)
        self._text(img, f"STATE {state}", 238, 28, 0.45, col, 1)

        if self.os.busy():
            self._text(img, "STOP ONLY", 372, 28, 0.45, self.WARN, 1)
        else:
            self._text(img, "LISTENING", 372, 28, 0.45, self.GOOD, 1)

        # Battery
        bx = 470
        self._text(img, "BATT", bx, 28, 0.45, self.DIM)
        cv2.rectangle(img, (bx + 44, 15), (bx + 124, 31), self.DIM, 1)
        fill = int(78 * world.battery_level / 100)
        bcol = self.GOOD if world.battery_level > 30 else self.WARN if world.battery_level > 10 else self.BAD
        cv2.rectangle(img, (bx + 45, 16), (bx + 45 + fill, 30), bcol, -1)
        self._text(img, f"{world.battery_level}%", bx + 132, 28, 0.45, self.TEXT)

        sp = world.speech_status
        scol = self.GOOD if sp == "LISTENING" else self.WARN if sp == "LOADING" else self.BAD
        cv2.circle(img, (662, 23), 6, scol, -1)
        self._text(img, f"MIC {sp}", 674, 28, 0.45, scol)

        self._text(img, f"VISION {self.os.vision_fps:4.1f} fps   UI {self.ui_fps:4.1f} fps   "
                        f"iGPU {IGPU}", 818, 28, 0.42, self.DIM)

    def _camera(self, img, world):
        x0, y0, w, h = self.CAM
        with self.os.view_lock:
            frame, dets, vis_ms = self.os.view

        if frame is None:
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (10, 8, 6), -1)
            self._text(img, "NO SIGNAL", x0 + w // 2 - 70, y0 + h // 2, 0.9, self.DIM, 2)
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), self.EDGE, 1)
            return

        fh, fw = frame.shape[:2]
        sx, sy = w / fw, h / fh
        view = cv2.resize(frame, (w, h))
        target = world.target_label

        for label, conf, (x1, y1, x2, y2) in dets:
            X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            if label == target:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 8)
                col = tuple(int(c * (0.6 + 0.4 * pulse)) for c in self.WARN)
                cv2.rectangle(view, (X1, Y1), (X2, Y2), col, 1)
                self._brackets(view, X1, Y1, X2, Y2, col, 3)
                tag = f"TARGET  {label.upper()}  {conf:.2f}"
            else:
                col = self.ACCENT
                cv2.rectangle(view, (X1, Y1), (X2, Y2), col, 1, cv2.LINE_AA)
                tag = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, self.FONT, 0.45, 1)
            ty = max(Y1, th + 6)
            cv2.rectangle(view, (X1, ty - th - 6), (X1 + tw + 8, ty), col, -1)
            self._text(view, tag, X1 + 4, ty - 4, 0.45, (15, 15, 15), 1)

        # Reticle
        cx, cy = w // 2, h // 2
        cv2.line(view, (cx - 18, cy), (cx - 6, cy), self.ACCENT, 1, cv2.LINE_AA)
        cv2.line(view, (cx + 6, cy), (cx + 18, cy), self.ACCENT, 1, cv2.LINE_AA)
        cv2.line(view, (cx, cy - 18), (cx, cy - 6), self.ACCENT, 1, cv2.LINE_AA)
        cv2.line(view, (cx, cy + 6), (cx, cy + 18), self.ACCENT, 1, cv2.LINE_AA)

        # Info strip
        strip = view[0:28, :].copy()
        cv2.rectangle(strip, (0, 0), (w, 28), (0, 0, 0), -1)
        view[0:28, :] = cv2.addWeighted(view[0:28, :], 0.45, strip, 0.55, 0)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(view, (14, 14), 5, self.BAD, -1)
        self._text(view, f"LIVE  YOLOv8n-OV @ {IGPU}   {vis_ms:5.1f} ms   {len(dets)} detections",
                   26, 19, 0.45, self.TEXT)

        # Fetch banner (whole mission)
        if self.robot.fetch_target:
            step = self.robot.fetch_step or self.robot.motor_phase
            msg = f"FETCH {self.robot.fetch_target.upper()}  |  {step}"
            (tw, _), _ = cv2.getTextSize(msg, self.FONT, 0.8, 2)
            bx = (w - tw) // 2
            cv2.rectangle(view, (bx - 16, h - 64), (bx + tw + 16, h - 24), (0, 0, 0), -1)
            cv2.rectangle(view, (bx - 16, h - 64), (bx + tw + 16, h - 24), self.WARN, 2)
            self._text(view, msg, bx, h - 35, 0.8, self.WARN, 2)

        img[y0:y0 + h, x0:x0 + w] = view
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), self.EDGE, 1)

    def _speech_strip(self, img, world):
        x, y, w, h = 16, 546, 640, 40
        self._panel(img, x, y, w, h)
        self._text(img, "COMMAND", x + 12, y + 26, 0.45, self.ACCENT, 1)
        if world.last_heard:
            age = time.time() - world.last_heard_t
            col = self.TEXT if age < 8 else self.DIM
            txt = world.last_heard if len(world.last_heard) < 48 else world.last_heard[:45] + "..."
            self._text(img, f'"{txt}"', x + 96, y + 26, 0.55, col, 1)
            self._text(img, f"{age:4.0f}s ago", x + w - 82, y + 26, 0.42, self.DIM)
        else:
            self._text(img, "say \"fido ...\"", x + 96, y + 26, 0.5, self.DIM)

    def _motors(self, img):
        x, y, w, h = self.RX, 56, 432, 292
        self._panel(img, x, y, w, h, "MOTORS / ODOMETRY")
        twist = self.robot.current_twist
        phase = self.robot.motor_phase
        moving = phase != "STANDBY"

        # ---- odometry map (robot-centred, world-fixed grid) ----
        M = 200
        mx, my = x + 12, y + 32
        scale = 30.0  # px per metre
        rx, ry, th = self.robot.x, self.robot.y, self.robot.theta
        cmap = np.full((M, M, 3), (24, 19, 14), np.uint8)
        c = M // 2

        def to_px(px, py):
            return int(c + (px - rx) * scale), int(c - (py - ry) * scale)

        g0x = math.floor(rx - M / scale / 2)
        g0y = math.floor(ry - M / scale / 2)
        for i in range(int(M / scale) + 2):
            gx, _ = to_px(g0x + i, 0)
            _, gy = to_px(0, g0y + i)
            cv2.line(cmap, (gx, 0), (gx, M), (48, 40, 32), 1)
            cv2.line(cmap, (0, gy), (M, gy), (48, 40, 32), 1)

        hx, hy = to_px(0, 0)
        if 0 <= hx < M and 0 <= hy < M:
            cv2.drawMarker(cmap, (hx, hy), self.DIM, cv2.MARKER_TILTED_CROSS, 10, 1)
            self._text(cmap, "home", hx + 6, hy - 6, 0.35, self.DIM)

        trail = list(self.robot.trail)
        for i in range(1, len(trail)):
            a = i / len(trail)
            col = tuple(int(v * a) for v in self.ACCENT)
            cv2.line(cmap, to_px(*trail[i - 1]), to_px(*trail[i]), col, 1, cv2.LINE_AA)

        # Robot body + heading
        def rot(dx, dy):
            return (int(c + dx * math.cos(th) - dy * math.sin(th)),
                    int(c - (dx * math.sin(th) + dy * math.cos(th))))
        body = np.array([rot(12, 0), rot(-8, 8), rot(-4, 0), rot(-8, -8)], np.int32)
        cv2.fillPoly(cmap, [body], self.WARN if moving else self.ACCENT, cv2.LINE_AA)

        if twist.linear != 0:
            tip = rot(12 + 30 * twist.linear, 0)
            cv2.arrowedLine(cmap, rot(12 if twist.linear > 0 else -8, 0), tip, self.GOOD, 2,
                            cv2.LINE_AA, tipLength=0.3)
        if twist.angular != 0:
            start = -math.degrees(th)
            sweep = -math.copysign(120, twist.angular)
            cv2.ellipse(cmap, (c, c), (22, 22), 0, start, start + sweep, self.GOOD, 2, cv2.LINE_AA)
            end = math.radians(-(start + sweep))
            ex, ey = int(c + 22 * math.cos(end)), int(c - 22 * math.sin(end))
            cv2.circle(cmap, (ex, ey), 4, self.GOOD, -1)

        img[my:my + M, mx:mx + M] = cmap
        cv2.rectangle(img, (mx, my), (mx + M, my + M), self.EDGE, 1)
        self._text(img, f"x {rx:+.2f}  y {ry:+.2f}  hdg {math.degrees(th) % 360:5.1f}",
                   mx, my + M + 18, 0.4, self.DIM)

        # ---- wheel bars (differential drive) ----
        vl, vr = self.robot.wheel_speeds()
        top, bot = my + 4, my + 150
        mid = (top + bot) // 2
        vmax = 1.0
        for i, (name, v) in enumerate((("L", vl), ("R", vr))):
            bx = x + 236 + i * 52
            cv2.rectangle(img, (bx, top), (bx + 30, bot), (24, 19, 14), -1)
            cv2.rectangle(img, (bx, top), (bx + 30, bot), self.EDGE, 1)
            cv2.line(img, (bx - 4, mid), (bx + 34, mid), self.DIM, 1)
            hgt = int(max(-1, min(1, v / vmax)) * (mid - top))
            if hgt:
                col = self.GOOD if v > 0 else self.BAD
                cv2.rectangle(img, (bx + 3, min(mid, mid - hgt)), (bx + 27, max(mid, mid - hgt)), col, -1)
            self._text(img, name, bx + 10, bot + 18, 0.5, self.TEXT, 1)
            self._text(img, f"{v:+.2f}", bx - 2, bot + 36, 0.4, self.DIM)
        self._text(img, "WHEELS m/s", x + 236, bot + 54, 0.38, self.DIM)

        # ---- motor phase, progress, gripper ----
        tx = x + 344
        self._text(img, "MOTION", tx, my + 12, 0.38, self.DIM)
        for i, line in enumerate(self._wrap(phase, 10)[:2]):
            self._text(img, line, tx, my + 32 + i * 17, 0.45, self.WARN if moving else self.GOOD, 1)
        cv2.rectangle(img, (tx, my + 58), (tx + 76, my + 64), (24, 19, 14), -1)
        if moving:
            cv2.rectangle(img, (tx, my + 58), (tx + int(76 * self.robot.seg_progress), my + 64), self.WARN, -1)
        q = self.robot.queued()
        self._text(img, f"steps {q}", tx, my + 80, 0.38, self.WARN if q else self.DIM)
        self._text(img, f"LIN {twist.linear:+.2f}", tx, my + 104, 0.4, self.TEXT)
        self._text(img, f"ANG {twist.angular:+.2f}", tx, my + 122, 0.4, self.TEXT)
        grip = self.robot.gripper
        self._text(img, "GRIPPER", tx, my + 150, 0.38, self.DIM)
        gcol = self.GOOD if grip == "OPEN" else self.WARN
        for i, line in enumerate(self._wrap(grip, 10)[:2]):
            self._text(img, line, tx, my + 168 + i * 16, 0.42, gcol, 1)

        # Linear / angular gauges
        gy = my + M + 28
        for i, (name, v, vm) in enumerate((("LIN", twist.linear, 1.0), ("ANG", twist.angular, 1.5))):
            yy = gy + i * 18
            gx0, gx1 = x + 52, x + w - 16
            gm = (gx0 + gx1) // 2
            self._text(img, name, x + 12, yy + 8, 0.38, self.DIM)
            cv2.rectangle(img, (gx0, yy), (gx1, yy + 9), (24, 19, 14), -1)
            cv2.line(img, (gm, yy - 2), (gm, yy + 11), self.DIM, 1)
            fill = int(max(-1, min(1, v / vm)) * (gx1 - gm))
            if fill:
                cv2.rectangle(img, (min(gm, gm + fill), yy + 1), (max(gm, gm + fill), yy + 8),
                              self.GOOD if v > 0 else self.BAD, -1)

    def _mission_steps(self, img, x, y, w):
        """NAVIGATE > ALIGN > GRASP > RETURN > DELIVER, current step highlighted."""
        steps = RobotHardware.FETCH_STEPS
        cur = self.robot.fetch_step
        idx = steps.index(cur) if cur in steps else -1
        bw = (w - (len(steps) - 1) * 6) // len(steps)
        for i, name in enumerate(steps):
            bx = x + i * (bw + 6)
            if idx < 0:
                fill, col = None, self.DIM
            elif i < idx:
                fill, col = (40, 70, 35), self.GOOD
            elif i == idx:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
                fill, col = tuple(int(c * (0.25 + 0.2 * pulse)) for c in self.WARN), self.WARN
            else:
                fill, col = None, self.DIM
            if fill:
                cv2.rectangle(img, (bx, y), (bx + bw, y + 22), fill, -1)
            cv2.rectangle(img, (bx, y), (bx + bw, y + 22), col, 1)
            (tw, _), _ = cv2.getTextSize(name, self.FONT, 0.36, 1)
            self._text(img, name, bx + (bw - tw) // 2, y + 15, 0.36, col)

    def _brain(self, img, world):
        x, y, w, h = self.RX, 358, 432, 196
        self._panel(img, x, y, w, h, "BRAIN")
        self._text(img, f"param-1-ov-int4 @ {self.llm_device}", x + 140, y + 16, 0.4, self.DIM)

        phase = world.mission_phase
        pcol = self.WARN if phase in ("THINKING", "SCANNING", "LOOKING") or phase.startswith("FETCHING") \
            else self.BAD if phase.startswith(("ABORTED", "FAILED", "ERROR", "CANCELLED")) else self.GOOD
        if phase == "THINKING":
            phase += "." * (int(time.time() * 3) % 4)
        self._text(img, "PHASE", x + 12, y + 44, 0.4, self.DIM)
        self._text(img, phase, x + 84, y + 44, 0.5, pcol, 1)

        self._text(img, "COMMAND", x + 12, y + 66, 0.4, self.DIM)
        req = world.last_request or "-"
        self._text(img, req[:44], x + 84, y + 66, 0.45, self.TEXT)

        # The exact object list the LLM was given for this decision
        self._text(img, "OBJECTS", x + 12, y + 88, 0.4, self.DIM)
        objs = ", ".join(world.last_objects) if world.last_request else "-"
        self._text(img, objs if len(objs) <= 44 else objs[:41] + "...", x + 84, y + 88, 0.45, self.TEXT)

        self._text(img, "LLM SAID", x + 12, y + 110, 0.4, self.DIM)
        reply = world.last_reply or "-"
        self._text(img, reply if len(reply) <= 40 else reply[:37] + "...", x + 84, y + 110, 0.45, self.ACCENT)

        self._text(img, "DECISION", x + 12, y + 140, 0.4, self.DIM)
        if world.last_decision:
            self._text(img, world.last_decision.upper(), x + 84, y + 143, 0.7, self.WARN, 2)
        elif world.last_request:
            self._text(img, "NONE", x + 84, y + 143, 0.7, self.BAD, 2)
        if world.last_brain_ms:
            self._text(img, f"{world.last_brain_ms:.0f} ms", x + w - 80, y + 143, 0.45, self.DIM)

        self._mission_steps(img, x + 12, y + 160, w - 24)

    def _in_view(self, img, world):
        """Objects detected in the last RECENT_S seconds - exactly the list the LLM receives."""
        x, y, w, h = self.RX, 564, 432, 140
        self._panel(img, x, y, w, h, f"IN VIEW (LAST {self.os.RECENT_S:.0f}s)")
        self._text(img, "-> sent to LLM", x + w - 118, y + 16, 0.38, self.DIM)
        items = self.os.recent_objects()

        if not items:
            self._text(img, "nothing detected", x + 12, y + 50, 0.45, self.DIM)
        # Names only - the LLM gets no confidences or counts
        now = time.time()
        for i, (label, (t, _, _, _)) in enumerate(list(items.items())[:15]):
            col_x = x + 12 + (i // 5) * 140
            yy = y + 42 + (i % 5) * 20
            is_target = label == world.target_label
            tcol = self.WARN if is_target else self.TEXT if now - t < 0.3 else self.DIM
            self._text(img, f"- {label}", col_x, yy, 0.45, tcol)

    def _activity(self, img):
        x, y, w, h = self.RX, 714, 432, 124
        self._panel(img, x, y, w, h, "ACTIVITY")
        if self.log is None:
            return
        now = time.time()
        for i, (t, tag, msg) in enumerate(reversed(self.log.recent(4))):
            yy = y + 46 + i * 22
            col = self.TAG_COLORS.get(tag, self.TEXT) if i == 0 else self.DIM
            self._text(img, f"{now - t:3.0f}s", x + 8, yy, 0.38, self.DIM)
            self._text(img, f"[{tag}]", x + 44, yy, 0.4, col)
            self._text(img, msg if len(msg) <= 36 else msg[:33] + "...", x + 130, yy, 0.4,
                       self.TEXT if i == 0 else self.DIM)

    # Voice commands: (what to say, what FIDO does). Every command starts with "fido" (or "hey fido").
    COMMANDS = (
        ("fido <your request>", "LLM picks an object in view, fetches it"),
        ("fido forward / back", "drive 1 m, then wait"),
        ("fido left / right", "turn 90°, then wait"),
        ("fido return", "U-turn, drive back to start"),
        ("fido home", "go to start, face forward, reset"),
        ("fido stop", "cancel task (busy) / shut down (idle)"),
    )

    def _commands(self, img):
        x, y, w, h = 16, 596, 640, 242
        busy = self.os.busy()
        self._panel(img, x, y, w, h, "VOICE COMMANDS")
        hint = "BUSY - only 'fido stop' works now" if busy else "start every command with 'fido'"
        self._text(img, hint, x + 200, y + 17, 0.5, self.WARN if busy else self.DIM)
        for i, (say, does) in enumerate(self.COMMANDS):
            yy = y + 56 + i * 34
            usable = not busy or say == "fido stop"
            self._text(img, say, x + 16, yy, 0.62, self.ACCENT if usable else (70, 62, 52), 2 if usable else 1)
            self._text(img, does, x + 262, yy, 0.55, self.TEXT if usable else (70, 62, 52))

    # ---- main loop ----------------------------------------------------------

    def render(self):
        world = self.os.world
        img = np.full((self.H, self.W, 3), self.BG, np.uint8)
        self._top_bar(img, world)
        self._camera(img, world)
        self._speech_strip(img, world)
        self._motors(img)
        self._brain(img, world)
        self._in_view(img, world)
        self._activity(img)
        self._commands(img)
        return img

    def run(self, shutdown: threading.Event):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.W, self.H)
        while not shutdown.is_set():
            img = self.render()
            cv2.imshow(self.WINDOW, img)

            now = time.perf_counter()
            self.ui_fps = 0.9 * self.ui_fps + 0.1 / max(now - self._last, 1e-3)
            self._last = now

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        shutdown.set()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    activity_log = HudLog(sys.stdout)
    sys.stdout = activity_log

    broker = ROSCmd()
    shutdown = threading.Event()

    camera = CameraThread()
    camera.start_capture()

    brain = FidoBrain(device=IGPU)
    robot = RobotHardware(broker)

    # Initialize the new Cognitive OS
    orchestrator = CognitiveOrchestrator(broker, camera, brain, robot, shutdown)
    hud = FidoHUD(orchestrator, robot, llm_device=brain.device, log=activity_log)

    # Speech runs in its own thread so the HUD can own the main (GUI) thread
    def speech_worker():
        try:
            speech = SpeechSystem(broker)
            speech.is_busy = orchestrator.busy
            orchestrator.world.speech_status = "LISTENING"
            speech.run_loop()
        except Exception as e:
            if shutdown.is_set():
                return  # audio stream torn down during exit - not a real crash
            print(f"[Speech] Crashed: {e}")
            orchestrator.world.speech_status = "ERROR"
        finally:
            if orchestrator.world.speech_status == "LISTENING":
                orchestrator.world.speech_status = "OFF"

    threading.Thread(target=speech_worker, daemon=True).start()

    try:
        hud.run(shutdown)
    except KeyboardInterrupt:
        print("\n[FIDO] Interrupted by user.")
    finally:
        shutdown.set()
        camera.stop()
        print("[FIDO] Offline.")
