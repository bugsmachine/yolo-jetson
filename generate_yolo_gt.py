import os
import glob
from ultralytics import YOLO

# 配置
# 目标文件夹列表
FOLDERS = ['MOT17-10-SDP', 'MOT17-11-FRCNN', 'MOT17-13-SDP']
# YOLO模型路径，会自动下载
MODEL_PATH = 'yolo11x.pt'
# 置信度阈值
CONF_THRES = 0.5
# 输出文件名
OUTPUT_FILENAME = 'gt_yolo.txt'

def generate_gt():
    # 获取当前工作目录
    base_dir = os.getcwd()
    print(f"当前工作目录: {base_dir}")

    # 加载模型
    print(f"正在加载模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("模型加载完成。")

    for folder in FOLDERS:
        folder_path = os.path.join(base_dir, folder)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder} 不存在，跳过。")
            continue
            
        print(f"正在处理文件夹: {folder}")
        
        # 查找视频文件 (.mp4)
        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        if not video_files:
            print(f"  在 {folder} 中未找到 .mp4 文件，跳过。")
            continue
            
        # 使用第一个找到的视频文件
        video_path = video_files[0]
        print(f"  找到视频文件: {os.path.basename(video_path)}")
        
        output_path = os.path.join(folder_path, OUTPUT_FILENAME)
        print(f"  开始推理，结果将保存到: {output_path}")
        
        try:
            # 运行推理
            # stream=True 用于处理长视频以节省内存
            results = model(source=video_path, stream=True, conf=CONF_THRES, verbose=False)
            
            with open(output_path, 'w') as f:
                frame_count = 0
                for result in results:
                    frame_id = frame_count + 1 # MOT格式帧号从1开始
                    
                    # 获取检测框 (xywh 格式: x_center, y_center, width, height) -> 错了，MOT格式通常是 left, top, width, height
                    # YOLO result.boxes.xywh 是中心点坐标和宽高
                    # YOLO result.boxes.xyxy 是左上角和右下角坐标
                    # MOT GT 格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    # 这里我们需要将 xyxy 转为 ltwh
                    
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # 获取 xyxy 坐标
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cls = box.cls[0]
                        
                        # 转换为 MOT 格式 (left, top, width, height)
                        bb_left = x1
                        bb_top = y1
                        bb_width = x2 - x1
                        bb_height = y2 - y1
                        
                        # class 0 是人 (person)
                        # 如果需要过滤特定类别，可以在这里添加 check
                        # COCO数据集 class 0 是 person
                        if int(cls) == 0:
                            # 写入文件
                            # frame_id, track_id(-1), bb_left, bb_top, bb_width, bb_height, not_ignored(1), class_id(1), visibility(1.0)
                            # 注意：mot_evaluator expects int for column 6 and 7
                            line = f"{frame_id},-1,{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1,1,1.0\n"
                            f.write(line)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                         print(f"  已处理 {frame_count} 帧...", end='\r')
                         
                print(f"\n  文件夹 {folder} 处理完成。总帧数: {frame_count}")
                
        except Exception as e:
            print(f"  处理 {folder} 时发生错误: {e}")

    print("\n所有任务完成。")

if __name__ == "__main__":
    generate_gt()
