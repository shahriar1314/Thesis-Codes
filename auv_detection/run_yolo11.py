# run_yolo11.py
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='yolo11s.pt')
    p.add_argument('--data',    type=str, default='real_data_obb/data.yaml')
    p.add_argument('--source',  type=str, default='real_data_obb/images/val')
    p.add_argument('--conf',    type=float, default=0.25)
    p.add_argument('--device',  type=str,   default='0')  # 'cpu' or GPU index
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        data=args.data,
        conf=args.conf,
        device=args.device,
        save=True,    # writes images with boxes to runs/predict/exp
        show=False
    )
    print(f"✅ Saved predictions to {results.save_dir}")

if __name__ == '__main__':
    main()
