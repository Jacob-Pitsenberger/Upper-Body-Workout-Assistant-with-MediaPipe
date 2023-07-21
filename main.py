"""
Author: Jacob Pitsenberger
Program: main.py
Version: 1.0
Project: Upper Body Workout Assistant with MediaPipe
Date: 7/21/2023
Purpose: This program contains the main method for running the Upper Body Workout Assistant with MediaPipe program.
Uses: upper_body_workout.py
"""

from upper_body_workouts import UpperBodyWorkouts

def main() -> None:
    """
    Main function to create an instance of the UpperBodyWorkouts class and start the video processing
    for multiple upper body workout methods.

    Workouts List:
        - "bicep_curls": Track the bicep curls exercise.
        - "shoulder_press": Track the shoulder press exercise.
        - "dumbbell_raises": Track the dumbbell raises exercise.
        - "triceps_extensions": Track the triceps extensions exercise.

    Note: To change the workout method, simply modify the index of the 'workouts' list.

    Example:
        To track 'shoulder_press':
        ```
        workouts = ["bicep_curls", "shoulder_press", "dumbbell_raises", "triceps_extensions"]
        curl_counter = UpperBodyWorkouts()
        curl_counter.process_video(workouts[1])
        ```
    """

    workouts = ["bicep_curls", "shoulder_press", "dumbbell_raises", "triceps_extensions"]
    curl_counter = UpperBodyWorkouts()
    curl_counter.process_video(workouts[0])


if __name__ == "__main__":
    main()
