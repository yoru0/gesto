import cv2
import sys
import json
import time
import logging
import collections
import numpy as np
import tracking as htm

from config import *
from typing import Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def fingers_up(landmark_list: List[Tuple[int, int, int]], hand_label: str) -> Tuple[List[int], float]:
    """Determine which fingers are up and calculate confidence.
    
    Args:
        landmark_list: List of hand landmarks as (id, x, y) tuples
        hand_label: "Left" or "Right" hand
        
    Returns:
        Tuple of (fingers_up_list, confidence_score)
    """
    try:
        points = {landmark_id: (x, y) for (landmark_id, x, y) in landmark_list}
        fingers_up_state = [0, 0, 0, 0, 0]

        if 4 in points and 3 in points:
            if hand_label == "Right":
                fingers_up_state[0] = 1 if points[4][0] < points[3][0] else 0
            else:
                fingers_up_state[0] = 1 if points[4][0] > points[3][0] else 0

        for finger_index, (tip, pip) in enumerate(zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]), start=1):
            if tip in points and pip in points:
                fingers_up_state[finger_index] = 1 if points[tip][1] < points[pip][1] else 0

        distances = []
        for tip, pip in zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]):
            if tip in points and pip in points:
                distances.append(abs(points[pip][1] - points[tip][1]))
        
        confidence = min(1.0, (np.mean(distances) / 40.0) if distances else 0.0)
        return fingers_up_state, confidence
    
    except Exception as e:
        logger.error(f"Error in fingers_up detection: {e}")
        return [0, 0, 0, 0, 0], 0.0


def classify_gesture(landmark_list: List[Tuple[int, int, int]], hand_label: str) -> Tuple[Optional[str], float, Optional[int]]:
    """Classify hand gesture from landmarks.
    
    Args:
        landmark_list: List of hand landmarks
        hand_label: "Left" or "Right" hand
        
    Returns:
        Tuple of (gesture_name, confidence, center_x)
    """
    try:
        x_coordinates = [x for (_, x, _) in landmark_list]
        center_x = int(sum(x_coordinates) / len(x_coordinates)) if x_coordinates else None

        fingers_up_state, confidence = fingers_up(landmark_list, hand_label)

        # Two fingers (peace sign / swipe gesture)
        if fingers_up_state[1] == 1 and fingers_up_state[2] == 1 and fingers_up_state[3] == 0 and fingers_up_state[4] == 0:
            return "two_fingers", max(0.5, confidence), center_x

        # Open palm (all fingers up)
        if sum(fingers_up_state) >= 4:
            return "open_palm", max(0.5, confidence), center_x

        # Fist (all fingers down)
        if sum(fingers_up_state[1:]) == 0:
            return "fist", max(0.5, 1.0 - confidence / 1.2), center_x

        return None, 0.0, center_x
    
    except Exception as e:
        logger.error(f"Error in gesture classification: {e}")
        return None, 0.0, None


def emit_gesture_event(gesture: str, confidence: float, hand: str = "Right") -> None:
    """Emit a gesture event as JSON to stdout.
    
    Args:
        gesture: The detected gesture name
        confidence: Confidence score (0.0 to 1.0)
        hand: Which hand ("Left" or "Right")
    """
    try:
        timestamp_ms = int(time.time() * 1000)
        event = {
            "gesture": gesture,
            "confidence": round(float(confidence), 3),
            "hand": hand,
            "ts": int(timestamp_ms / 1000)
        }
        print(json.dumps(event), flush=True)
        logger.info(f"Emitted gesture: {gesture} (conf={confidence:.3f}, hand={hand})")
    except Exception as e:
        logger.error(f"Error emitting gesture event: {e}")


def main() -> None:
    logger.info("Starting gesture recognition...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            sys.exit(1)
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        detector = htm.HandDetector(detection_con=0.6, track_con=0.5, max_hands=1)

        # State tracking
        history = collections.deque(maxlen=HISTORY_LENGTH)
        recent_positions = collections.deque()
        last_state = None
        last_state_time = 0
        last_emit_time = 0
        previous_time = 0

        logger.info("Gesture recognition started successfully")

        while True:
            success, img = cap.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                continue

            img = cv2.flip(img, 1)
            detector.find_hands(img, draw=True)
            hands = detector.get_hands(img, draw_dots=False)

            current_time_ms = int(time.time() * 1000)

            # Process detected hands
            gesture_label, gesture_confidence, center_x, hand_label = None, 0.0, None, "Right"
            if hands:
                hand = hands[0]
                hand_label = hand.label
                gesture_label, gesture_confidence, center_x = classify_gesture(hand.lm_list, hand_label)

                if gesture_label:
                    debug_fingers_up, _ = fingers_up(hand.lm_list, hand_label)
                    cv2.putText(img, f"UP={debug_fingers_up} {gesture_label}", 
                              (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

            # Update history
            history.append((gesture_label, gesture_confidence, current_time_ms))

            # Track recent positions for swipe detection
            if center_x is not None:
                recent_positions.append((current_time_ms, center_x))

            # Clean old positions
            while recent_positions and current_time_ms - recent_positions[0][0] > SWIPE_WINDOW_MS:
                recent_positions.popleft()

            # Check for stable gesture
            is_stable = False
            average_confidence = 0.0
            if len(history) == history.maxlen:
                labels = [label for (label, _, _) in history]
                confidences = [conf for (label, conf, _) in history if label]
                is_stable = (labels.count(labels[-1]) == len(labels) and labels[-1] is not None)
                average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

            # Gesture recognition logic
            if gesture_label == "fist":
                last_state, last_state_time = "fist", current_time_ms

            # Play/pause
            if (gesture_label == "open_palm" and last_state == "fist" and 
                (current_time_ms - last_state_time) <= FIST_TO_OPEN_MS):
                
                if (gesture_confidence >= (CONFIDENCE_THRESHOLD - 0.05) and 
                    (current_time_ms - last_emit_time) >= COOLDOWN_MS):
                    
                    emit_gesture_event("play_pause", gesture_confidence, hand_label)
                    last_emit_time = current_time_ms
                last_state = None

            # Next/prev
            if is_stable and history[-1][0] == "two_fingers" and len(recent_positions) >= 2:
                delta_x = recent_positions[-1][1] - recent_positions[0][1]
                if (abs(delta_x) >= SWIPE_PIXELS and 
                    average_confidence >= (CONFIDENCE_THRESHOLD - 0.1)):
                    
                    if current_time_ms - last_emit_time >= COOLDOWN_MS:
                        gesture = "prev" if delta_x > 0 else "next"
                        emit_gesture_event(gesture, average_confidence, hand_label)
                        last_emit_time = current_time_ms
                        recent_positions.clear()

            # HUD
            current_time = time.time()
            fps = 1 / max(1e-6, (current_time - previous_time))
            previous_time = current_time
            
            hud_text = f"{int(fps)} FPS | {gesture_label or '-'} | conf={average_confidence:.2f}"
            cv2.putText(img, hud_text, (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

            cv2.imshow("Gesture Recognizer", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        sys.exit(1)
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()