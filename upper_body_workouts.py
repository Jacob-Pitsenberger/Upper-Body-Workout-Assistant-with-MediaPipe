"""
Author: Jacob Pitsenberger
Program: upper_body_workouts.py
Version: 1.0
Project: Upper Body Workout Assistant with MediaPipe
Date: 7/21/2023
Purpose: This Python module utilizes the Mediapipe library to perform pose recognition and count repetitions
         for various upper body workouts. The application captures real-time video from the webcam, tracks
         the user's body movements, and counts repetitions for specific exercises.
Uses: N/A
"""
import dataclasses
from typing import Tuple
import mediapipe as mp
import cv2
import numpy as np

def calculate_angle(a: list, b: list, c: list) -> float:
    """
    Calculate the angle (in degrees) between three points 'a', 'b', and 'c' in a 2D plane.

    Args:
        a (list): The first (start) point as a tuple (x, y).
        b (list): The second (mid) point as a tuple (x, y).
        c (list): The third (end) point as a tuple (x, y).

    Returns:
        float: The calculated angle in degrees (range: 0 to 180).

    Note:
        The function calculates the angle by converting the points into numpy arrays and then using the arctan2
        function to find the angle between them in radians. It then converts the angle to degrees and ensures it
        falls within the range of 0 to 180 degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    deg = np.round(deg)
    if deg > 180.0:
        deg = 360 - deg

    return deg


class UpperBodyWorkouts:
    """
    UpperBodyWorkouts class performs real-time tracking of upper body workouts using the Mediapipe library.
    It detects pose landmarks from the webcam feed and tracks exercises such as bicep curls, shoulder press,
    dumbbell raises, and triceps extensions.

    Attributes:
        BICEP_CURL_ANGLE_RANGE (list): List containing the angle range for bicep curl exercise in the format [down, up].
        SHOULDER_PRESS_ANGLE_RANGE (list): List containing the angle range for shoulder press exercise in the format [down, up].
        DUMBBELL_RAISES_ANGLE_RANGE (list): List containing the angle range for dumbbell raises exercise in the format [down, up].
        TRICEPS_EXTENSIONS_ANGLE_RANGE (list): List containing the angle range for triceps extensions exercise in the format [down, up].
        FONT_SIZE (float): Font size for text display on the output window.
        TEXT_COLOR (tuple): RGB color tuple for text display on the output window.
        INFO_BOX_COLOR (tuple): RGB color tuple for drawing the info box on the output window.
        TEXT_THICKNESS (int): Thickness of text display on the output window.
        WINDOW_WIDTH (int): Width of the output window.
        WINDOW_HEIGHT (int): Height of the output window.
    """
    BICEP_CURL_ANGLE_RANGE = [160, 30]
    SHOULDER_PRESS_ANGLE_RANGE = [60, 140]
    DUMBBELL_RAISES_ANGLE_RANGE = [10, 80]
    TRICEPS_EXTENSIONS_ANGLE_RANGE = [10, 40]

    FONT_SIZE = 0.5
    TEXT_COLOR = (255, 255, 255)
    INFO_BOX_COLOR = (138, 72, 48)
    TEXT_THICKNESS = 1

    WORKOUT_TEXT_POSITION = (15, 22)
    WORKOUT_POSITION = (100, 22)
    COUNTER_TEXT_POSITION = (15, 64)
    COUNTER_NUMBER_POSITION = (90, 70)
    STAGE_TEXT_POSITION = (15, 106)
    STAGE_POSITION = (95, 110)

    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480

    def __init__(self) -> None:
        """
        Initializes the UpperBodyWorkouts class with counters, stage, Mediapipe Pose model, and video capture.
        """
        self.counter = 0
        self.stage = None
        self.workout = None
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(0)

    def _detect_pose_landmarks(self, image: np.ndarray) -> dataclasses:
        """
        Detects pose landmarks using the Mediapipe Pose model on the provided image.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList: Pose landmarks detected in the image.
        """
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image)
            return results.pose_landmarks

    def _update_counters(self, angle: float, workout: str) -> None:
        """
        Updates the exercise counter and stage based on the provided angle and workout.

        Args:
            angle (float): Angle measurement of the workout.
            workout (str): Name of the workout being performed.
        """
        if workout == "bicep curls":
            self.workout = "bicep curls"
            if angle > self.BICEP_CURL_ANGLE_RANGE[0]:
                self.stage = "down"
            if angle < self.BICEP_CURL_ANGLE_RANGE[1] and self.stage == "down":
                self.stage = "up"
                self.counter += 1
        elif workout == "shoulder press":
            self.workout = "Shoulder Press"
            if angle < self.SHOULDER_PRESS_ANGLE_RANGE[0]:
                self.stage = "down"
            if angle > self.SHOULDER_PRESS_ANGLE_RANGE[1] and self.stage == "down":
                self.stage = "up"
                self.counter += 1
        elif workout == "dumbbell raises":
            self.workout = "dumbbell raises"
            if angle < self.DUMBBELL_RAISES_ANGLE_RANGE[0]:
                self.stage = "down"
            if angle > self.DUMBBELL_RAISES_ANGLE_RANGE[1] and self.stage == "down":
                self.stage = "up"
                self.counter += 1
        elif workout == "triceps extensions":
            self.workout = "triceps extensions"
            if angle < self.TRICEPS_EXTENSIONS_ANGLE_RANGE[0]:
                self.stage = "down"
            if angle > self.TRICEPS_EXTENSIONS_ANGLE_RANGE[1] and self.stage == "down":
                self.stage = "up"
                self.counter += 1

    def _display_info_box(self, image: np.ndarray) -> None:
        """
        Draws an info box on the output window to display workout, counter, and stage information.

        Args:
            image (numpy.ndarray): Input image in BGR format.
        """
        cv2.rectangle(image, (0, 0), (240, 120), self.INFO_BOX_COLOR, -1)
        cv2.putText(image, 'WORKOUT: ', self.WORKOUT_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, str(self.workout), self.WORKOUT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, 'REPS: ', self.COUNTER_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, str(self.counter), self.COUNTER_NUMBER_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    2 * self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, 'STAGE: ', self.STAGE_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, self.stage, self.STAGE_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    2 * self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)

    def bicep_curls(self, landmarks: dataclasses) -> Tuple[float, Tuple[int, int], str]:
        """
        Measures the angle and position of the right arm for bicep curls.

        Args:
            landmarks: Pose landmarks extracted by the Mediapipe Pose model.

        Returns:
            tuple: Tuple containing angle, angle position, and workout name for bicep curls.
        """
        workout = "bicep curls"
        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        angle_position = tuple(np.multiply(r_elbow, [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]).astype(int).round(2))
        return angle, angle_position, workout

    def shoulder_press(self, landmarks: dataclasses) -> Tuple[float, Tuple[int, int], str]:
        """
        Measures the angle and position of the right arm for shoulder press.

        Args:
            landmarks: Pose landmarks extracted by the Mediapipe Pose model.

        Returns:
            tuple: Tuple containing angle, angle position, and workout name for shoulder press.
        """
        workout = "shoulder press"
        r_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        angle_position = tuple(np.multiply(r_shoulder, [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]).astype(int).round(2))
        return angle, angle_position, workout

    def dumbbell_raises(self, landmarks: dataclasses) -> Tuple[float, Tuple[int, int], str]:
        """
        Measures the angle and position of the right arm for dumbbell raises.

        Args:
            landmarks: Pose landmarks extracted by the Mediapipe Pose model.

        Returns:
            tuple: Tuple containing angle, angle position, and workout name for dumbbell raises.
        """
        workout = "dumbbell raises"
        r_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        angle_position = tuple(np.multiply(r_shoulder, [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]).astype(int).round(2))
        return angle, angle_position, workout

    def triceps_extensions(self, landmarks: dataclasses) -> Tuple[float, Tuple[int, int], str]:
        """
        Measures the angle and position of the right arm for triceps extensions.

        Args:
            landmarks: Pose landmarks extracted by the Mediapipe Pose model.

        Returns:
            tuple: Tuple containing angle, angle position, and workout name for triceps extensions.
        """
        workout = "triceps extensions"
        r_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        angle_position = tuple(np.multiply(r_shoulder, [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]).astype(int).round(2))
        return angle, angle_position, workout

    def process_video(self, workout_method: str) -> None:
        """
        Main method to process the webcam video feed and track upper body workouts in real-time.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            pose_landmarks = self._detect_pose_landmarks(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if pose_landmarks:
                    landmarks = pose_landmarks.landmark

                    # angle, position, workout = self.bicep_curls(landmarks)
                    # angle, position, workout = self.shoulder_press(landmarks)
                    # angle, position, workout = self.dumbbell_raises(landmarks)
                    # angle, position, workout = self.triceps_extensions(landmarks)
                    if workout_method == 'bicep_curls':
                        angle, position, workout = self.bicep_curls(landmarks)
                    elif workout_method == 'shoulder_press':
                        angle, position, workout = self.shoulder_press(landmarks)
                    elif workout_method == 'dumbbell_raises':
                        angle, position, workout = self.dumbbell_raises(landmarks)
                    elif workout_method == 'triceps_extensions':
                        angle, position, workout = self.triceps_extensions(landmarks)
                    else:
                        raise ValueError(f"Invalid workout method: {workout_method}")

                    self._update_counters(angle, workout)

                    # for displaying angle to screen.
                    cv2.putText(image, str(angle), position,
                                cv2.FONT_HERSHEY_COMPLEX, self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)

                    # Draw pose landmarks and connections
                    mp.solutions.drawing_utils.draw_landmarks(image, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                              mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0),
                                                                                                     thickness=2,
                                                                                                     circle_radius=2),
                                                              mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0),
                                                                                                     thickness=2,
                                                                                                     circle_radius=2))

            except Exception as e:
                print(f"Error occurred during pose detection: {e}")

            self._display_info_box(image)
            cv2.imshow('output', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self) -> None:
        """
        Releases the video capture when the object is deleted.
        """
        if self.cap.isOpened():
            self.cap.release()
