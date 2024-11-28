import os
import cv2
import random
from ultralytics import YOLO


def select_random_filenames(path, count=10, seed=None):
    try:
        if seed is not None:
            random.seed(seed)

        # Get a list of all files in the specified directory
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
        # Check if there are enough files
        if len(all_files) < count:
            print(f"Only {len(all_files)} files available. Selecting all of them.")
            return all_files
        
        # Select random files
        random_files = random.sample(all_files, count)
        return random_files
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():


    # model = YOLO("yolov8x.pt") 
    model = YOLO("visdrones/baseline/weights/best.pt") 

    # img_root = "/home/yaz/mlc/baseline/datasets/VisDrone/VisDrone2019-DET-test-dev/images"
    # images = select_random_filenames(img_root, count=15, seed=42)

    # model([os.path.join(img_root, img) for img in images])
    results = model("image.png", save=True)
    # out_root = results[0].save_dir
    out_root = results[0].save_dir

    for res in results:
        img_name = res.path.split("/")[-1]
        img = res.plot(font_size=14, pil=True)
        cv2.imwrite(os.path.join(out_root, img_name), img)


if __name__ == "__main__":
    main()