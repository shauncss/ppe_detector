from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Path to your best model weights (update 'train' to 'train2' etc. if needed)
    weights_path = 'runs/detect/train2/weights/best.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find model weights at {weights_path}")
        print("Make sure your training has finished and the path is correct.")
        exit()

    # Load the trained model
    model = YOLO(weights_path)

    # 2. Run inference on a static image (if it exists)
    if os.path.exists('test_image.jpg'):
        print("Testing Image...")
        model.predict(source='test_image.jpg', save=True, conf=0.5)
    else:
        print("No 'test_image.jpg' found. Skipping image test.")

    # 3. Run inference on a video file (if it exists)
    if os.path.exists('test_video.mp4'):
        print("Testing Video...")
        model.predict(source='test_video.mp4', save=True, conf=0.5)
    else:
        print("No 'test_video.mp4' found. Skipping video test.")
        
    print("Inference complete! Check the runs/detect/predict folder for results.")