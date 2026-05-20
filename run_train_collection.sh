#!/bin/bash
# 训练集 label 重新生成脚本
# 使用新训练模式（selector 真实决策 + oracle YOLO）
# 估计耗时约 1.7 小时，建议夜间挂机跑
set -e
cd "$(dirname "$0")"

MEDIA=/home/tan/Desktop/all
ORG=$MEDIA/organized_dataset
CKPT=checkpoints/best.pth
LOG=run_train_collection.log

run_seq() {
    local prefix=$1
    local name=$2
    local label="${prefix}__${name}"
    local dataset_root="$ORG/$prefix/$name"
    local out_json="${name}-new-yolo.json"

    echo "" | tee -a "$LOG"
    echo "========================================== $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "[$label]" | tee -a "$LOG"
    echo "==========================================" | tee -a "$LOG"

    if [ ! -d "$dataset_root" ]; then
        echo "[SKIP] $dataset_root not found" | tee -a "$LOG"
        return
    fi

    python main.py "$dataset_root" \
        --media-root "$MEDIA" \
        --train-mode \
        --selector-checkpoint "$CKPT" 2>&1 | tee -a "$LOG"

    if [ -f "$out_json" ]; then
        mv "$out_json" "data/labels/${label}.json"
        echo "[OK] data/labels/${label}.json saved" | tee -a "$LOG"
    else
        echo "[WARN] $out_json not found" | tee -a "$LOG"
    fi
}

echo "训练集 label 重新生成开始: $(date)" | tee "$LOG"

run_seq drone M0101
run_seq drone M0206
run_seq drone M0209
run_seq drone M0301
run_seq drone M0401
run_seq drone M0402
run_seq drone M0604
run_seq drone M0802
run_seq drone M1001
run_seq drone M1002
run_seq drone M1003
run_seq drone M1007
run_seq drone M1101
run_seq drone M1201
run_seq drone M1301
run_seq drone M1302
run_seq drone M1304
run_seq drone M1305
run_seq drone uav0000071_03240_v
run_seq drone uav0000143_02250_v
run_seq drone uav0000182_00000_v
run_seq drone uav0000239_03720_v
run_seq drone uav0000239_12336_v
run_seq drone uav0000243_00001_v
run_seq drone uav0000244_01440_v
run_seq drone uav0000263_03289_v
run_seq drone uav0000264_02760_v
run_seq drone uav0000278_00001_v
run_seq drone uav0000279_00001_v
run_seq drone uav0000289_00001_v
run_seq drone uav0000295_02300_v
run_seq drone uav0000309_00000_v
run_seq drone uav0000315_00000_v
run_seq drone uav0000342_04692_v
run_seq drone uav0000352_05980_v
run_seq drone uav0000360_00001_v
run_seq static MVI_20033
run_seq static MVI_20035
run_seq static MVI_20061
run_seq static MVI_20062
run_seq static MVI_20063
run_seq static MVI_20064
run_seq static MVI_39051
run_seq static MVI_39211
run_seq static MVI_39311
run_seq static MVI_39371
run_seq static MVI_39401
run_seq static MVI_39501
run_seq static MVI_39511
run_seq static MVI_39761
run_seq static MVI_39771
run_seq static MVI_39801
run_seq static MVI_39811
run_seq static MVI_39821
run_seq static MVI_39851
run_seq static MVI_39861
run_seq static MVI_39931
run_seq static MVI_40131
run_seq static MVI_40141
run_seq static MVI_40152
run_seq static MVI_40161
run_seq static MVI_40162
run_seq static MVI_40171
run_seq static MVI_40181
run_seq static MVI_40201
run_seq static MVI_40204
run_seq static MVI_40212
run_seq static MVI_40243
run_seq static MVI_40244
run_seq static MVI_40701
run_seq static MVI_40711
run_seq static MVI_40714
run_seq static MVI_40742
run_seq static MVI_40743
run_seq static MVI_40751
run_seq static MVI_40763
run_seq static MVI_40771
run_seq static MVI_40772
run_seq static MVI_40773
run_seq static MVI_40774
run_seq static MVI_40775
run_seq static MVI_40851
run_seq static MVI_40852
run_seq static MVI_40853
run_seq static MVI_40854
run_seq static MVI_40863
run_seq static MVI_40864
run_seq static MVI_40891

echo "" | tee -a "$LOG"
echo "=========================================="  | tee -a "$LOG"
echo "全部完成: $(date)" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 跑完自动开始重训
echo "开始重新训练模型..." | tee -a "$LOG"
python -m src.train --config configs/default.yaml 2>&1 | tee -a "$LOG"
echo "训练完成: $(date)" | tee -a "$LOG"
