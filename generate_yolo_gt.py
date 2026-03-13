import os
import glob
from ultralytics import YOLO


FOLDERS = ['MOT17-10-SDP', 'MOT17-11-FRCNN', 'MOT17-13-SDP']


MODEL_PATH = 'yolo11x.pt'


CONF_THRES = 0.5


OUTPUT_FILENAME = 'gt_yolo.txt'


def generate_gt():
    base_dir = os.getcwd()
    print(f"当前工作目录: {base_dir}")

    print(f"正在加载模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("模型加载完成。")

    for folder in FOLDERS:
        folder_path = os.path.join(base_dir, folder)

        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder} 不存在，跳过。")
            continue

        print(f"正在处理文件夹: {folder}")

        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        if not video_files:
            print(f"  在 {folder} 中未找到 .mp4 文件，跳过。")
            continue

        video_path = video_files[0]
        print(f"  找到视频文件: {os.path.basename(video_path)}")

        output_path = os.path.join(folder_path, OUTPUT_FILENAME)
        print(f"  开始推理，结果将保存到: {output_path}")

        try:
            results = model(source=video_path, stream=True, conf=CONF_THRES, verbose=False)

            with open(output_path, 'w') as f:
                frame_count = 0
                for result in results:
                    frame_id = frame_count + 1

                    boxes = result.boxes.cpu().numpy()

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cls = box.cls[0]

                        bb_left = x1
                        bb_top = y1
                        bb_width = x2 - x1
                        bb_height = y2 - y1

                        if int(cls) == 0:
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
