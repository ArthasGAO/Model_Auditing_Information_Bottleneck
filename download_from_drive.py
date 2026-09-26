import os
import shutil

# ==========================================
# 1. 配置路径
# ==========================================

# Google Drive 中模型所在目录
SOURCE_DIR = r"G:\My Drive\Model_Training\saved_models\vanilla"

# 本地模型保存目录
LOCAL_ROOT = r"./saved_models/vanilla/Negative_Model_Pool_0.0"

# ==========================================
# 2. 目标模型配置
# ==========================================

MODEL_PREFIX = "CIFAR-10_VGG16_25000"

SEED_START = 92
SEED_END = 122

OVERLAP_RATE = "0.0"


def sync_specific_folders():

    # 本地目标路径
    target_dir = LOCAL_ROOT
    os.makedirs(target_dir, exist_ok=True)

    print(f"开始扫描源目录: {SOURCE_DIR}")
    print(f"本地目标目录: {target_dir}")
    print(
        f"目标模型: {MODEL_PREFIX}_<seed>_{OVERLAP_RATE}, "
        f"seed = {SEED_START} ~ {SEED_END}"
    )
    print("-" * 60)

    # 检查源目录
    if not os.path.exists(SOURCE_DIR):
        print(
            "错误：找不到源路径，请确保 Google Drive "
            "桌面版已启动并成功挂载 G 盘。"
        )
        return

    copied_count = 0
    skipped_count = 0
    missing_count = 0

    # ==========================================
    # 3. 按 seed 精确构造目标文件夹
    # ==========================================
    for seed in range(SEED_START, SEED_END + 1):

        # 例如：
        # CIFAR-10_ResNet-18_25000_92_0.0
        folder_name = (
            f"{MODEL_PREFIX}_{seed}_{OVERLAP_RATE}"
        )

        source_item_path = os.path.join(
            SOURCE_DIR,
            folder_name
        )

        target_item_path = os.path.join(
            target_dir,
            folder_name
        )

        # --------------------------------------
        # 检查源模型是否存在
        # --------------------------------------
        if not os.path.isdir(source_item_path):
            print(f"[缺失] 源目录不存在: {folder_name}")
            missing_count += 1
            continue

        # --------------------------------------
        # 本地已经存在则跳过
        # --------------------------------------
        if os.path.exists(target_item_path):
            print(f"[跳过] 已存在: {folder_name}")
            skipped_count += 1
            continue

        # --------------------------------------
        # 拷贝整个模型目录
        # --------------------------------------
        print(f"[下载中] 正在拷贝: {folder_name} ...")

        shutil.copytree(
            source_item_path,
            target_item_path
        )

        copied_count += 1

    # ==========================================
    # 4. 输出统计
    # ==========================================
    print("-" * 60)

    total_expected = SEED_END - SEED_START + 1

    print("同步完成！")
    print(f"目标模型总数: {total_expected}")
    print(f"成功拷贝:     {copied_count}")
    print(f"本地已存在:   {skipped_count}")
    print(f"源目录缺失:   {missing_count}")


if __name__ == "__main__":
    sync_specific_folders()