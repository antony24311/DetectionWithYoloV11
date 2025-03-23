from ultralytics import YOLO
import torch
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    model = YOLO('yolo11n.pt')
    results = model.train(
                            data=r'C:\Users\anton\Desktop\新增資料夾 (4)\dataset.yaml',
                            epochs=500,
                            batch=8,
                            imgsz=512,
                            multi_scale=True,
                            augment=True,                           
                            mosaic=1.0,
                            scale=0.5,
                            device=0
                        )
    print("训练完成！")

if __name__ == '__main__':
    main()