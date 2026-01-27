"""
检查数据集文件数量
"""
from pathlib import Path

DATA_DIR = Path("KIT-ML")
MOTION_DIR = DATA_DIR / "new_joints_40_bin256"
TEXT_DIR = DATA_DIR / "texts"

# 统计文件数量
motion_files = list(MOTION_DIR.glob("*.npy"))
text_files = list(TEXT_DIR.glob("*.txt"))

print(f"Motion files (.npy): {len(motion_files)}")
print(f"Text files (.txt): {len(text_files)}")

# 统计交集
motion_ids = set(f.stem for f in motion_files)
text_ids = set(f.stem for f in text_files)

common = motion_ids & text_ids
only_motion = motion_ids - text_ids
only_text = text_ids - motion_ids

print(f"\nCommon IDs (both motion and text): {len(common)}")
print(f"Only motion (no text): {len(only_motion)}")
print(f"Only text (no motion): {len(only_text)}")

# 打印一些只有 motion 没有 text 的例子
if only_motion:
    print(f"\nExamples of motion without text (first 10):")
    for mid in sorted(only_motion)[:10]:
        print(f"  {mid}")

# 打印一些只有 text 没有 motion 的例子
if only_text:
    print(f"\nExamples of text without motion (first 10):")
    for tid in sorted(only_text)[:10]:
        print(f"  {tid}")

