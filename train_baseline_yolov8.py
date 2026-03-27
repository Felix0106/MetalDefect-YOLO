import argparse
from pathlib import Path
from ultralytics import YOLO  # 直接在这里导入 YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 baseline on NEU-DET.")
    parser.add_argument(
        "--data",
        type=str,
        default="E:/Graduation_Project/datasets/NEU-DET/NEU-DET/data.yaml",
        help="Path to dataset yaml.",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model name or path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="Device id, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Windows建议设为0防止报错")
    parser.add_argument("--project", type=str, default="E:/Graduation_Project/runs", help="Output root directory.")
    parser.add_argument("--name", type=str, default="neu_yolov8n_baseline", help="Run name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX after training.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 删除了原来报错的 YOLO = import_yolo() 这行代码
    
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers, # 已经默认改为 0
        project=args.project,
        name=args.name,
        seed=args.seed,
        cos_lr=True,
        close_mosaic=10,
        pretrained=True,
    )
    print(f"Training finished. Results: {train_results.save_dir}")

    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    best_model = YOLO(str(best_pt))
    #metrics = best_model.val(data=str(data_path), split="val", device=args.device)
    metrics = best_model.val(data=str(data_path), split="val", device=args.device, project=args.project, name=args.name + "_val")
    print("Validation finished.")
    print(f"Precision(B): {metrics.box.mp:.4f}")
    print(f"Recall(B):    {metrics.box.mr:.4f}")
    print(f"mAP50(B):     {metrics.box.map50:.4f}")
    print(f"mAP50-95(B):  {metrics.box.map:.4f}")

    if args.export_onnx:
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)
        print(f"ONNX exported to: {onnx_path}")

if __name__ == "__main__":
    main()