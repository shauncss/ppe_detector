from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26l.pt') 

    model.train(
        data="dataset2/data.yaml", 
        epochs=50,        
        patience=10,      
        imgsz=1024,        
        batch=-1,         
        workers=8,        
        cache='disk',       
        device=0 
    )