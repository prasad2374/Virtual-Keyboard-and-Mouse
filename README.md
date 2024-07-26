# Eye Tracking Virtual Keyboard and Typing Application 🖥️👁️⌨️

Welcome to the Eye Tracking Virtual Keyboard and Typing Application! This project leverages computer vision and eye-tracking technologies to provide a virtual keyboard and eye typing system. 

## Features ✨

- **Virtual Keyboard**: Use hand gestures to type on a virtual keyboard.
- **Eye Typing**: Use eye movements and blinks to type text.
- **Graphical User Interface**: Easy-to-use interface to start and stop the modes.

## Technologies Used 🛠️

- **OpenCV**: Open Source Computer Vision Library for image processing and computer vision tasks.
- **Mediapipe**: A framework for building multimodal (e.g., video, audio, etc.) machine learning pipelines.
- **NumPy**: A fundamental package for scientific computing with Python.
- **Pygame**: A set of Python modules designed for writing video games.
- **Tkinter**: Python's de-facto standard GUI package.
- **Pygame**: Used for creating the virtual keyboard interface and handling visual feedback.
- **PyAutoGUI**: Used for simulating keyboard presses.

## Requirements 📋

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

## Installation 🚀

1. **Clone the repository**:
    ```sh
    git clone https://github.com/d-hackmt/Virtual-Keyboard-and-Mouse.git
    ```

2. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

## Libraries and Their Purpose 📚

1. **OpenCV (`opencv-python`)**:
    - For image capture and processing.
    - Used to access the webcam and handle the video feed.

2. **Mediapipe (`mediapipe`)**:
    - Provides machine learning solutions for detecting facial landmarks.
    - Used for tracking eye and hand movements.

3. **NumPy (`numpy`)**:
    - A package for scientific computing.
    - Used for array operations and mathematical calculations.

4. **Pygame (`pygame`)**:
    - A library for creating games and multimedia applications.
    - Used to create the visual interface for the virtual keyboard.

5. **Tkinter (`tkinter`)**:
    - The standard GUI toolkit for Python.
    - Used to create the main interface for starting and stopping the application modes.

6. **PyAutoGUI (`pyautogui`)**:
    - A library for programmatically controlling the mouse and keyboard.
    - Used to simulate keypress events based on eye or hand inputs.

## How to Run the Application ▶️

1. **Start the Tkinter GUI**:
    ```sh
    python appfull.py
    ```

2. **Use the GUI to start and stop the virtual keyboard or eye typing modes**.

## Usage 📝

- **Start Virtual Keyboard**: This will open a window where you can use hand gestures to type.
- **Start Eye Typing**: This will open a window where you can use your eye movements and blinks to type.
- **Stop Virtual Keyboard**: Stops the virtual keyboard mode.
- **Stop Eye Typing**: Stops the eye typing mode.
- **Terminate Program**: Closes the entire application.

## Contributing 🤝

Feel free to fork this repository and contribute by submitting a pull request. Any improvements, bug fixes, or feature additions are welcome.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Thanks to the creators of OpenCV, Mediapipe, NumPy, Pygame, Tkinter, and PyAutoGUI for their amazing libraries.

---

Happy coding! 😃
