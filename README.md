# DeepFake-Buster

This repository contains a deepfake detection application built using Streamlit, TensorFlow Lite, and OpenCV. The app allows users to upload video files, preprocesses them, and predicts whether the video is real or a deepfake using a pre-trained model.

## Project Structure

```bash
.
├── Model Training/
├── __pycache__/
├── LICENSE
├── README.md
├── app.py
├── main.ipynb
├── main.py
└── deepfake_detection_model.tflite
```

- **Model Training/**: Directory for training-related scripts and files.
- **__pycache__/**: Compiled Python files.
- **LICENSE**: License information for the project.
- **README.md**: This file, providing an overview of the project.
- **app.py**: The main Streamlit application file.
- **main.ipynb**: Jupyter Notebook for experimentation and model testing.
- **main.py**: Python script for running the detection outside of Streamlit.
- **deepfake_detection_model.tflite**: The pre-trained TensorFlow Lite model used for inference.

## Installation

To run this application, you'll need to install the required dependencies:

```bash
pip install streamlit opencv-python-headless numpy Pillow tensorflow
```

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/deepfake-detection-app.git
    cd deepfake-detection-app
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Upload a video file in `.mp4` format to check whether it's real or a deepfake.

## Features

- **Video Upload**: Upload a video file in `.mp4` format.
- **Frame Extraction**: Extracts and preprocesses frames from the video.
- **Deepfake Detection**: Uses a TensorFlow Lite model to predict if the video is real or a deepfake.
- **Logs**: Displays logs and detection results in real-time.

## Dependencies

- **Streamlit**: Web framework for interactive applications.
- **OpenCV**: Library for video processing.
- **NumPy**: Library for numerical operations.
- **TensorFlow Lite**: Deep learning model framework for inference.
- **Pillow**: Python Imaging Library for image processing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## Acknowledgments

This project utilizes various open-source libraries and pre-trained models. Special thanks to the contributors of these projects.
