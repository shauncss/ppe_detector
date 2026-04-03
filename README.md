# PPE Detector

This project is a computer vision pipeline for detecting Personal Protective Equipment (PPE) on worksites, utilizing the Ultralytics YOLO framework. 

## Project Structure

* **`train.py`**: Initializes the YOLO model and begins the training process on the specified dataset.
* **`resume.py`**: A utility script to resume an interrupted training session from the last saved checkpoint.
* **`test.py`**: Runs inference on test media to evaluate the trained model's performance.
* **`test_image.jpg`**: A sample inference image featuring construction workers wearing safety vests and hard hats.

## Requirements

Ensure you have the necessary dependencies installed before running the scripts:
```bash
pip install ultralytics
