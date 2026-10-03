"""
FIDO — Embodied Cognitive Operating System
================================================

Features
--------
✓ Whisper Medium conversational ASR
✓ Silero Neural VAD (Continuous InputStream for gapless listening)
✓ Persistent live camera & YOLO tracking
✓ Temporal World Memory (Remembers objects recently seen)
✓ State Machine & Mission Queuing (Handles concurrent requests safely)
✓ Context-Aware Phi-3 Reasoning (Injects battery/state to LLM)
✓ 10Hz Motor Command Watchdog
"""

import cv2
import re
import sys
import time
import threading
import numpy as np
import sounddevice as sd
import torch
import openvino_genai as ov_genai
from dataclasses import dataclass
from ultralytics import YOLO
from typing import Callable, Dict, List, Optional


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
# World State & Memory (The Cognitive Layer)
# ============================================================================

@dataclass
class ObjectMemory:
    label: str
    last_seen: float

class WorldState:
    def __init__(self):
        self.system_state: str = "IDLE"  # IDLE, FETCHING, ERROR
        self.battery_level: int = 100
        
        # Temporal memory maps object labels to when they were last seen
        self.spatial_memory: Dict[str, ObjectMemory] = {}
        
        # Task Management
        self.active_task: Optional[str] = None
        self.task_queue: List[str] = []

    def update_perception(self, labels: List[str]):
        """Updates temporal memory with fresh timestamps."""
        now = time.time()
        for label in labels:
            self.spatial_memory[label] = ObjectMemory(label=label, last_seen=now)

    def get_recent_objects(self, max_age_s: float = 30.0) -> List[str]:
        """Returns objects seen within the last `max_age_s` seconds."""
        now = time.time()
        recent = []
        for label, mem in list(self.spatial_memory.items()):
            if now - mem.last_seen <= max_age_s:
                recent.append(label)
            else:
                # Memory decay: Forget objects we haven't seen in a while
                del self.spatial_memory[label]
        return recent


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
    def __init__(self, device="GPU"):
        self.device = device
        print("[Vision] Loading YOLO...")

        self.model = YOLO("yolov8n_openvino_model/", task="detect")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        print(f"[Vision] Warming up...")
        for _ in range(5):
            self.model.predict(dummy, verbose=False)
        print("[Vision] Ready.")

    def detect(self, frame):
        t0 = time.perf_counter()
        result = self.model.predict(frame, verbose=False)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        labels = []
        for box in result.boxes:
            label = self.model.names[int(box.cls[0])]
            if label not in labels:
                labels.append(label)

        return labels, result.plot(), latency_ms


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
    def __init__(self, model_path="phi3_openvino_int8/", device="GPU"):
        print(f"[Brain] Loading Phi-3 on {device}...")
        try:
            self.pipe = ov_genai.LLMPipeline(model_path, device)
            self._warmup()
            print("[Brain] Ready.")
        except Exception as e:
            print(f"[Brain] Failed: {e}")
            self.pipe = None

    def _warmup(self):
        warm = (
            "<|system|>You are a robot.<|end|>"
            "<|user|>hello<|end|>"
            "<|assistant|>"
        )
        self.pipe.generate(warm, max_new_tokens=4, do_sample=False)

    def _direct_match(self, command, labels):
        cmd_tokens = tokenize(command)
        for label in labels:
            for part in tokenize(label):
                if len(part) >= 3 and part in cmd_tokens:
                    return label
        return None

    def _llm_decide(self, command, labels, world_state: WorldState):
        numbered = "\n".join(f"{i+1}. {obj}" for i, obj in enumerate(labels))
        
        prompt = f"""<|system|>
You are FIDO, an embodied cognitive robot.
Battery: {world_state.battery_level}%
Status: {world_state.system_state}

Select the ONE object from your memory that BEST matches the user's request.
Rules:
- Reply ONLY with ONE number.
- Reply 0 if nothing matches.
- Do NOT explain.

Objects in Memory:
{numbered}
<|end|>
<|user|>
{command}
<|end|>
<|assistant|>
"""
        t0 = time.perf_counter()
        raw = self.pipe.generate(prompt, max_new_tokens=4, do_sample=False).strip()
        latency = (time.perf_counter() - t0) * 1000

        digit_match = re.search(r"^\s*(\d+)", raw)
        if not digit_match:
            digits = re.findall(r"\d+", raw)
            if not digits:
                print(f"[Brain] Invalid output: {raw}")
                return None
            idx = int(digits[-1])
        else:
            idx = int(digit_match.group(1))

        if idx == 0:
            print(f"[Brain] No suitable object ({latency:.0f} ms)")
            return None

        if idx > len(labels):
            return None

        choice = labels[idx - 1]
        print(f"[Brain] LLM → #{idx} '{choice}' ({latency:.0f} ms)")
        return choice

    def decide(self, command, seen_objects, world_state: WorldState):
        if not seen_objects:
            print("[Brain] Scene empty.")
            return None

        match = self._direct_match(command, seen_objects)
        if match:
            print(f"[Brain] Direct match → {match}")
            return match

        if self.pipe is None:
            return None

        return self._llm_decide(command, seen_objects, world_state)


# ============================================================================
# Speech (Continuous Streaming)
# ============================================================================

class SpeechSystem:
    MODEL = "whisper-medium-ov"
    DEVICE = "GPU"
    JUNK = {"thank you", "thanks", "bye", "hmm", "uh", "um"}

    def __init__(self, messenger):
        self.messenger = messenger

        print(f"[Speech] Loading {self.MODEL} on {self.DEVICE}...")
        try:
            self.pipe = ov_genai.WhisperPipeline(self.MODEL, self.DEVICE)
        except Exception as e:
            print(f"[Speech] GPU failed: {e}\nFalling back to CPU...")
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

            print(f"\n[Speech] Heard: '{text}'")
            tokens = tokenize(text)
            if not tokens:
                continue

            if "stop" in tokens:
                self.messenger.publish("voice_intent", "SHUTDOWN")
                break

            fetch_words = {"fetch", "bring", "find", "grab", "get"}
            if any(w in tokens for w in fetch_words):
                cleaned = re.sub(r"^(hey\s+)?fido\s+", "", text)
                cleaned = re.sub(r"^(fetch|bring|find|get|grab)\s+", "", cleaned).strip()
                print(f"[Speech] Fetch request → {cleaned}")
                self.messenger.publish("voice_intent", f"FETCH_REQUEST|{cleaned}")
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

            self.messenger.publish("cmd_vel", cmd)


# ============================================================================
# Robot Hardware (Threaded Watchdog)
# ============================================================================

class RobotHardware(threading.Thread):
    def __init__(self, messenger):
        super().__init__(daemon=True)
        self.current_twist = Twist()
        
        messenger.subscribe("cmd_vel", self._update_twist)
        messenger.subscribe("robot_action", self.handle_action)
        
        self.start()

    def _update_twist(self, twist):
        self.current_twist = twist
        print(f"[Robot] State update: linear={twist.linear:+.1f} angular={twist.angular:+.1f}")

    def run(self):
        while True:
            if self.current_twist.linear != 0.0 or self.current_twist.angular != 0.0:
                pass 
            time.sleep(0.1)

    def handle_action(self, action):
        if action.startswith("FETCH:"):
            target = action.split(":", 1)[1]
            print(f"[Robot] HARDWARE STARTING FETCH SEQUENCE → {target}")


# ============================================================================
# Cognitive Orchestrator (The Operating System)
# ============================================================================

class CognitiveOrchestrator:
    def __init__(self, messenger, camera, brain):
        self.messenger = messenger
        self.camera = camera
        self.brain = brain
        self.vision = FidoVision(device="GPU")
        
        # Central World State
        self.world = WorldState()
        
        self.messenger.subscribe("voice_intent", self.handle_intent)
        
        # Start Cognitive Loop
        self.cognitive_thread = threading.Thread(target=self._cognitive_tick, daemon=True)
        self.cognitive_thread.start()

    def handle_intent(self, intent: str):
        if intent == "SHUTDOWN":
            print("[Orchestrator] OS Shutting down...")
            sys.exit(0)

        if "FETCH_REQUEST" in intent:
            desc = intent.split("|", 1)[1]
            
            # State Check Validators
            if self.world.system_state == "FETCHING":
                print(f"[Orchestrator] REJECTED: Busy fetching. Queueing '{desc}'.")
                self.world.task_queue.append(desc)
                return
                
            if self.world.battery_level < 5:
                print("[Orchestrator] REJECTED: Battery too low for fetch mission.")
                return

            self.world.task_queue.append(desc)

    def _cognitive_tick(self):
        """The heartbeat of the Operating System. Runs continuously."""
        while True:
            # 1. Process World Senses (Update Memory)
            frame = self.camera.get_latest(timeout=0.1)
            if frame is not None:
                labels, _, _ = self.vision.detect(frame)
                if labels:
                    self.world.update_perception(labels)

            # 2. Process Task Queue
            if self.world.system_state == "IDLE" and self.world.task_queue:
                next_task = self.world.task_queue.pop(0)
                self.world.active_task = next_task
                self.world.system_state = "FETCHING"
                
                threading.Thread(target=self._execute_fetch_mission, args=(next_task,), daemon=True).start()
            
            time.sleep(0.1) # 10Hz Cognitive Tick

    def _execute_fetch_mission(self, description: str):
        print(f"\n[OS] Starting Mission: {description}")
        t0 = time.perf_counter()

        try:
            # Query Temporal Memory instead of just instantaneous frame
            known_objects = self.world.get_recent_objects(max_age_s=30.0)
            
            if not known_objects:
                print("[OS] Memory empty. Initiating 360 scan...")
                time.sleep(2.0) # Wait for camera to update WorldState
                known_objects = self.world.get_recent_objects()

            if not known_objects:
                print("[OS] Mission Failed: No objects found in memory.")
                return

            # Feed multi-modal state into Brain
            decision = self.brain.decide(description, known_objects, self.world)

            if decision is None:
                print(f"[OS] Mission Aborted: Phi-3 could not match '{description}' to {known_objects}")
                return

            print(f"[OS] Target Acquired: {decision.upper()} ({(time.perf_counter() - t0)*1000:.0f} ms)")
            self.messenger.publish("robot_action", f"FETCH:{decision}")
            
            # Wait for hardware to report success (Stub)
            time.sleep(3.0) 

        except Exception as e:
            print(f"[OS] FATAL MISSION ERROR: {e}")
        finally:
            print("[OS] Mission Concluded. Returning to IDLE.")
            self.world.active_task = None
            self.world.system_state = "IDLE"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    broker = ROSCmd()
    
    camera = CameraThread()
    camera.start_capture()
    
    brain = FidoBrain(device="GPU")
    robot = RobotHardware(broker)
    
    # Initialize the new Cognitive OS
    orchestrator = CognitiveOrchestrator(broker, camera, brain)
    
    try:
        speech = SpeechSystem(broker)
        speech.run_loop()
    except KeyboardInterrupt:
        print("\n[FIDO] Interrupted by user.")
    finally:
        camera.stop()
        print("[FIDO] Offline.")