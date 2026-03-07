from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26l.pt')

    model.train(
        data="dataset3/data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,
        workers=16,
        cache='disk',
        device=0
    )