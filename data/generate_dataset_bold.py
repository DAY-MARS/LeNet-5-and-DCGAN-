import os
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "my_digits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EACH = 500
IMAGE_SIZE = 32

# 优先使用粗体字体
FONTS = [
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/lucon.ttf",
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
        font_size = random.randint(26, 32)  # 字号更大
        font = ImageFont.truetype(font_path, font_size)

        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        draw = ImageDraw.Draw(img)

        x = random.randint(0, 3)
        y = random.randint(0, 3)

        # 三层描边加粗
        draw.text((x, y), str(digit), font=font, fill=140)  # 浅灰底影
        draw.text((x, y), str(digit), font=font, fill=60)   # 中灰
        draw.text((x, y), str(digit), font=font, fill=0)    # 纯黑核心

        angle = random.uniform(-6, 6)
        img = img.rotate(angle, fillcolor=255)

        img.save(os.path.join(digit_dir, f"{i:04d}.png"))

    print(f"✅ 数字 {digit} 已生成 {NUM_EACH} 张")

print(f"\n加粗数据集已保存在 '{OUTPUT_DIR}' 文件夹")