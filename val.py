import wandb
import os
import cv2
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


def main():
    # model = YOLO("yolov8x.pt") 
    model = YOLO("visdrones/baseline/weights/best.pt") 

    # wandb.init(project="visdrones", name="test-dev-inference", job_type="inference")
    add_wandb_callback(model, enable_model_checkpointing=False)

    # img_root = "/home/yaz/mlc/baseline/datasets/VisDrone/VisDrone2019-DET-test-dev/images/*27.jpg"
    img_root = "/home/yaz/mlc/baseline/datasets/VisDrone/VisDrone2019-DET-test-dev/test-dev.yaml"

    model.val(data=img_root, project="visdrones", name="test-dev-inference")
    # out_root = results[0].save_dir

    # for res in results:
    #     img_name = res.path.split("/")[-1]
    #     img = res.plot(font_size=14, pil=True)
    #     cv2.imwrite(os.path.join(out_root, img_name), img)

    wandb.finish()
    

if __name__ == "__main__":
    main()
