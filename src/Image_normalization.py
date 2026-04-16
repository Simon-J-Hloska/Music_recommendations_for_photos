import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image, ImageOps


class ImageNormalizer:
    def __init__(self, input_dir, output_dir, size=224):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.size = size

    def resize_with_padding(self, img):
        img = ImageOps.exif_transpose(img)
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        bg_color = img.resize((1, 1)).getpixel((0, 0))
        new_img = Image.new("RGB", (self.size, self.size), bg_color)
        x = (self.size - img.width) // 2
        y = (self.size - img.height) // 2
        new_img.paste(img, (x, y))
        return new_img

    def process_one(self, filename):
        try:
            path = os.path.join(self.input_dir, filename)
            img = Image.open(path).convert("RGB")
            img = self.resize_with_padding(img)
            save_path = os.path.join(self.output_dir, filename)
            img.save(save_path, "JPEG", quality=90, optimize=True)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def check_if_resized_already(self, list_of_image_names):
        found_image_names = []
        for img_name in list_of_image_names:
            if (Path(self.output_dir) / img_name).exists():
                found_image_names.append(img_name)
        return found_image_names

    def run(self):
        valid_ext = (".jpg", ".jpeg", ".png", ".webp")
        files = [f for f in os.listdir(self.input_dir)if f.lower().endswith(valid_ext)]
        resized_already_images = self.check_if_resized_already(files)
        for img_name in resized_already_images:
            files.remove(img_name)
            print(f"skipped resizing {img_name}, already resized")
        print(f"Resizing {len(files)} images to {self.size}x{self.size}...")
        with Pool(cpu_count()) as p:
            p.map(self.process_one, files)
        print("Done resizing.")