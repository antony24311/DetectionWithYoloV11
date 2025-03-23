import os
import cv2
from ultralytics import YOLO

# 影像資料夾路徑
input_folder = "C:/Users/anton/Desktop/新增資料夾 (4)/image"
output_folder = "C:/Users/anton/Desktop/新增資料夾 (4)/predict"
bbox_folder = "C:/Users/anton/Desktop/新增資料夾 (4)/bbox"
os.makedirs(output_folder, exist_ok=True)  # 存儲切割後影像
os.makedirs(bbox_folder, exist_ok=True)  # 存儲 BBox 資料

# 載入 YOLOv11 訓練好的模型
model = YOLO("best.pt")  # 使用最好的模型

# 讀取資料夾內的所有影像並進行預測
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # 執行 YOLO 預測
    results = model(image_path)

    # 儲存預測的 BBox 資訊
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 獲取 BBox 座標
            bboxes.append([x1, y1, x2, y2])

    # 儲存 BBox 資料到文本檔
    bbox_file = os.path.join(bbox_folder, image_name.replace('.jpg', '.txt'))  # 設定輸出文件名稱
    with open(bbox_file, 'w') as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

    # 儲存預測的結果影像
    result_image = results[0].plot()  # 繪製結果影像
    result_image_path = os.path.join(output_folder, image_name)
    cv2.imwrite(result_image_path, result_image)  # 儲存影像

    print(f"儲存結果：{image_name}")

print("YOLO 預測完成並儲存結果！")
