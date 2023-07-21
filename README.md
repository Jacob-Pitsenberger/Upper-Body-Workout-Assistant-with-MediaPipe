# Upper Body Workouts Pose Recognition

This is a Python project that uses the Mediapipe library to perform pose recognition and count repetitions for various upper body workouts. The application utilizes a webcam to capture real-time video and tracks the user's body movements to count repetitions for specific exercises.

## Requirements

Before running the application, make sure you have the following installed:

- Python 3.x
- OpenCV
- Mediapipe

You can install the required libraries using `pip`:

```bash
pip install opencv-python mediapipe
```

## Usage

1. Clone the repository or download the project files.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the main.py script to start the application:

```bash
python main.py
```

The application will open a new window displaying the webcam feed with pose landmarks and the tracked exercise. It will also show the count of repetitions and the current stage (up or down) of the exercise.

By default, the application will track "bicep curls." To track other exercises, modify the workouts list in the main.py file:

```bash
workouts = ["bicep_curls", "shoulder_press", "dumbbell_raises", "triceps_extensions"]
```
Replace the workout name at the desired index to track a different exercise.

To exit the application, press the 'q' key.

## Supported Upper Body Workouts
Bicep Curls: Tracks the number of bicep curls performed.

Shoulder Press: Tracks the number of shoulder press movements.

Dumbbell Raises: Tracks the number of dumbbell raises.

Triceps Extensions: Tracks the number of triceps extensions.

## How It Works
The application uses the Mediapipe library to detect and track the user's upper body pose. It then calculates the angles between relevant keypoints to determine the exercise movement. The program updates the repetition count based on the specific exercise angle ranges.

## Known Issues
Certain lighting conditions or body positions may affect pose recognition accuracy.
Contributing
If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

## Authors

- [Jacob Pitsenberger](https://github.com/Jacob-Pitsenberger)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

