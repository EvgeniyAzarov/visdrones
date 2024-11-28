import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback



def main():
    # model = YOLO("yolov8x.pt") 
    model = YOLO("visdrones/baseline/weights/best.pt") 

    add_wandb_callback(model, enable_model_checkpointing=True)
    model.train(
        data="VisDrone.yaml", 
        epochs=100, 
        batch=16,
        imgsz=640, 
        project="visdrones", 
        name="baseline",
        device=7,
        resume=True,
    )

    wandb.finish()

if __name__ == "__main__":
    main()