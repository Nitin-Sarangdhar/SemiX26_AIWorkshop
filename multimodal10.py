import numpy as np
import openvino_genai as ov_genai
import sounddevice as sd
import time
import sys
import os
import cv2
import json  # <<< CHANGE 1: Added import
from ultralytics import YOLO
from typing import Callable, Dict, List

# --- ROScmd.py (Internal Messaging) ---
class ROSCmd:
    def __init__(self):
        self.topics: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.topics: 
            self.topics[topic] = []
        self.topics[topic].append(callback)

    def publish(self, topic: str, message: any):
        if topic in self.topics:
            for callback in self.topics[topic]: 
                callback(message)

# --- Data Structures ---
class Twist:
    def __init__(self, linear: float = 0.0, angular: float = 0.0):
        self.linear = linear
        self.angular = angular

class CameraFrame:
    def __init__(self, raw_img: np.ndarray, timestamp: float):
        self.image = raw_img
        self.timestamp = timestamp

# --- Local Vision Layer (Multi-HW Benchmarking) ---
class FidoVision:
    def __init__(self):
        print("[Vision] Initializing OpenVINO Engines for CPU, GPU, and NPU...")
        try:
            # Using the pre-exported OpenVINO model folder
            model_path = 'yolov8n_openvino_model/'
            self.model = YOLO(model_path, task='detect')
            
            # Map logical names to OpenVINO device strings
            self.hw_map = {
                "CPU": "cpu",
                "GPU": "intel:gpu",
                "NPU": "intel:npu"
            }
            
            # Warm up phase: Compiles the kernels for each hardware unit
            print("[Vision] Warming up hardware engines (Compilation)...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for name, dev in self.hw_map.items():
                try:
                    self.model.predict(dummy_img, device=dev, verbose=False)
                    print(f"  ✅ {name} Engine Ready")
                except Exception as e:
                    print(f"  ⚠️ {name} initialization skipped: {e}")

        except Exception as e:
            print(f"[Vision] Critical Error: {e}")
            sys.exit(1)

    def detect_and_annotate(self, frame_np: np.ndarray):
        """Runs inference across all available units and returns labels + timings."""
        timings = {}
        found_labels = []
        annotated_frame = None

        for name, dev in self.hw_map.items():
            try:
                start = time.perf_counter()
                results = self.model.predict(frame_np, device=dev, verbose=False)[0]
                end = time.perf_counter()
                
                timings[name] = (end - start) * 1000
                
                # Use results from the first successful run for the UI/Labels
                if annotated_frame is None:
                    annotated_frame = results.plot()
                    for box in results.boxes:
                        label = self.model.names[int(box.cls[0])]
                        if label not in found_labels:
                            found_labels.append(label)
            except:
                timings[name] = None # Hardware unit busy or unavailable

        return found_labels, annotated_frame, timings

# --- Speech Layer (SpeechTiny) ---
class SpeechTiny:
    def __init__(self, messenger: ROSCmd):
        self.messenger = messenger
        print("[System] Loading Whisper Tiny...")
        try:
            # Whisper runs on the CPU Kernel for stability
            self.pipe = ov_genai.WhisperPipeline("whisper-tiny-ov", "CPU")
            self.config = self.pipe.get_generation_config()
            self.config.max_new_tokens = 64
        except Exception as e:
            print(f"[Critical] Failed to load OpenVINO model: {e}")
            sys.exit(1)
        
        self.steering_prompt = "Fido, Forward, Back, Left, Right, Stop, Fetch."
        self.fs, self.seconds, self.silence_threshold = 16000, 3, 0.0008 

    def run_loop(self):
        print("\n[READY] Fido is listening...")
        print("\nExact Commands: FIDO Forward, FIDO back, FIDO Left, FIDO right, FIDO stop")
        print("\nAmbiguous commands: Fetch Something to play, Fetch something to drink form,...")
        while True:
            audio = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1, dtype='float32')
            sd.wait()
            
            if np.sqrt(np.mean(audio**2)) < self.silence_threshold: 
                continue

            result = self.pipe.generate(audio.flatten(), config=self.config, initial_prompt=self.steering_prompt)
            text = result.texts[0].strip().lower()
            
            if text:
                print(f"[SpeechTiny] Voice heard: '{text}'")
                if "fetch" in text or "find" in text: 
                    self.messenger.publish("voice_intent", f"FETCH_REQUEST|{text}")
                elif "stop" in text:
                    self.messenger.publish("voice_intent", "SHUTDOWN")
                    break
                
                intent = None
                
                
                if "forward" in text: intent = "MOVE_FORWARD"
                elif "back" in text: intent = "MOVE_BACKWARD"
                elif "left" in text: intent = "TURN_LEFT"
                elif "right" in text: intent = "TURN_RIGHT"

                if intent: 
                    self.messenger.publish("voice_intent", intent)

# --- Brain (Orchestrator) ---
class Orchestrator:
    def __init__(self, messenger: ROSCmd):
        self.messenger = messenger
        self.vision = FidoVision()
        self.current_speech = ""  # <<< FIX 1: Initialize this variable
        self.messenger.subscribe("voice_intent", self.handle_intent)
        self.messenger.subscribe("camera_feed", self.process_vision)

    def handle_intent(self, intent: str):
        if intent == "SHUTDOWN": 
            sys.exit(0)

        if "FETCH_REQUEST" in intent:
            # Splits "FETCH_REQUEST|find a cup" and takes the second part
            self.current_speech = intent.split("|")[1] if "|" in intent else "fetch"
            print(f"[Orchestrator] Fido, capturing frame for analysis...")
            self.messenger.publish("robot_action", "TRIGGER_CAMERA")
            return

        cmd = Twist()
        if intent == "MOVE_FORWARD": cmd.linear = 1.0
        elif intent == "MOVE_BACKWARD": cmd.linear = -1.0
        elif intent == "TURN_LEFT": cmd.angular = 0.5
        elif intent == "TURN_RIGHT": cmd.angular = -0.5
        self.messenger.publish("cmd_vel", cmd)

    def process_vision(self, frame: CameraFrame):
        # Perform multi-hardware inference
        found_labels, annotated_img, timings = self.vision.detect_and_annotate(frame.image)
        
        filename = f"fido_vision_{int(frame.timestamp)}.jpg"
        cv2.imwrite(filename, annotated_img)
        senses_data = {
            "command": self.current_speech,
            "seen": found_labels,
            "timestamp": frame.timestamp
        }
        with open("fido_senses.json", "w") as f:
            json.dump(senses_data, f)
        print(f"[Orchestrator] 💾 Senses saved for Brain analysis.")
                
        # Performance Report
        print("\n" + "="*40)
        if found_labels:
            print(f"[Vision] I see: {', '.join(found_labels)}")
            print(f"[Vision] Image saved: {filename}")
        else:
            print("[Vision] No recognized objects detected.")
        
        print("-" * 40)
        print(f" HW Unit   | Accelerator Latency ")
        print("-" * 40)
        for hw, ms in timings.items():
            val = f"{ms:.2f} ms" if ms else "N/A"
            print(f" {hw:<10} | {val}")
        print("="*40 + "\n")

# --- Actuators ---
class RobotHardware:
    def __init__(self, messenger: ROSCmd):
        self.messenger = messenger
        self.messenger.subscribe("cmd_vel", self.execute_move)
        self.messenger.subscribe("robot_action", self.handle_action)

    def execute_move(self, twist: Twist):
        if twist.linear != 0 or twist.angular != 0:
            print(f"[Robot] >>> MOVING: L={twist.linear}, A={twist.angular}")

    def handle_action(self, action: str):
        if action == "TRIGGER_CAMERA":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): return
            # Flush buffer
            for _ in range(5): cap.read()
            ret, frame = cap.read()
            if ret:
                self.messenger.publish("camera_feed", CameraFrame(frame, time.time()))
            cap.release()

if __name__ == "__main__":
    broker = ROSCmd()
    robot = RobotHardware(broker)
    brain = Orchestrator(broker)
    try:
        ear = SpeechTiny(broker)
        ear.run_loop()
    except (KeyboardInterrupt, SystemExit):
        print("\n[Fido] Offline.")
