import os

from PIL import Image

def resize_with_padding(image_path, target_size=(512, 512), background_color=(0, 0, 0)):
    """
    保持宽高比缩放图片，并用指定颜色填充至目标尺寸
    """
    image = Image.open(image_path)
    # 计算缩放比例
    width, height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / width, target_height / height)
    new_size = (int(width * ratio), int(height * ratio))

    # 缩放图片
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # 创建目标尺寸的画布并居中粘贴缩放后的图片
    padded_image = Image.new("RGB", target_size, background_color)
    offset = (
        (target_width - new_size[0]) // 2,
        (target_height - new_size[1]) // 2
    )
    padded_image.paste(resized_image, offset)

    padded_image.save(image_path)

def main():
    for filename in os.listdir("liutao"):
        resize_with_padding(f"liutao/{filename}")

if __name__ == '__main__':
    main()
