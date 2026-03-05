from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26m.pt') 

    model.train(
        data="roboflow_dataset/data.yaml", 
        epochs=50,        
        patience=10,      
        imgsz=416,        
        batch=-1,         
        workers=8,        
        cache=True,       
        device=0 
    )