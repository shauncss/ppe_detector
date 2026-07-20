from ultralytics import YOLO
import os

if __name__ == '__main__':
    weights_path = 'runs/detect/train3/weights/best.pt' # change train number
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find model weights at {weights_path}")
        exit()

    model = YOLO(weights_path)

    if os.path.exists('assets/test_image.jpg'):
        print("Testing Image...")
        model.predict(source='assets/test_image.jpg', save=True, conf=0.3)
    else:
        print("No 'test_image.jpg' found. Skipping image test.")

    if os.path.exists('assets/test_video.mp4'):
        print("Testing Video...")
        model.predict(source='assets/test_video.mp4', save=True, conf=0.3)
    else:
        print("No 'test_video.mp4' found. Skipping video test.")

    print("Done !!")