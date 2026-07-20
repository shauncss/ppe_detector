from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('weights/yolo26l.pt')

    model.train(
        data="data/dataset3/data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,
        workers=16,
        cache='disk',
        device=0
    )