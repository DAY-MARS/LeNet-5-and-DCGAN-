import os
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "my_digits"          # 直接覆盖原来的文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EACH = 500
IMAGE_SIZE = 32

# 优先用笔画较粗的字体，如果系统没有则自动剔除
FONTS = [
    "C:/Windows/Fonts/impact.ttf",    # 粗体，非常醒目
    "C:/Windows/Fonts/comic.ttf",     # 手写体，较粗
    "C:/Windows/Fonts/arialbd.ttf",   # Arial 粗体
    "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI 粗体
    "C:/Windows/Fonts/calibrib.ttf",  # Calibri 粗体
    "C:/Windows/Fonts/timesbd.ttf",   # Times New Roman 粗体
    "C:/Windows/Fonts/georgiab.ttf",  # Georgia 粗体
    "C:/Windows/Fonts/lucon.ttf",     # Lucida Console
]
available_fonts = [f for f in FONTS if os.path.exists(f)]
if not available_fonts:
    # 如果粗体都找不到，退回普通字体
    available_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/comic.ttf",
    ]
print(f"使用字体: {available_fonts}")

for digit in range(10):
    digit_dir = os.path.join(OUTPUT_DIR, str(digit))
    os.makedirs(digit_dir, exist_ok=True)

    for i in range(NUM_EACH):
        font_path = random.choice(available_fonts)
        # 字号进一步加大，确保笔画占更多像素
        font_size = random.randint(26, 32)
        font = ImageFont.truetype(font_path, font_size)

        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        draw = ImageDraw.Draw(img)

        # 小偏移，防止切边
        x = random.randint(0, 3)
        y = random.randint(0, 3)

        # 三次绘制，由浅到深，模拟“加粗 + 加深”
        draw.text((x, y), str(digit), font=font, fill=140)   # 浅灰底影
        draw.text((x, y), str(digit), font=font, fill=60)    # 中灰
        draw.text((x, y), str(digit), font=font, fill=0)     # 纯黑核心

        # 旋转角度进一步缩小，保持数字形状稳定
        angle = random.uniform(-6, 6)
        img = img.rotate(angle, fillcolor=255)

        img.save(os.path.join(digit_dir, f"{i:04d}.png"))

    print(f"✅ 数字 {digit} 已生成 {NUM_EACH} 张")

print(f"\n加粗加深数据集已保存在 '{OUTPUT_DIR}' 文件夹")