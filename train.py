import argparse
from attention_module import create_attention_model
from ultralytics import YOLO
from prepare_datasets import prepare_all_datasets


def create_attention_model(pretrained_size='m'):
    model = YOLO(f'yolov8{pretrained_size}.pt')
    return add_attention_to_model(model)

def train_with_transfer_learning(
    pretrained_size='m',
    custom_dataset='/content/VOC.yaml',
    epochs=100,
    batch_size=16,
    imgsz=640,
    freeze_backbone=True
):
    model = create_attention_model(pretrained_size)
    train_args = {
        'data': custom_dataset,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'name': f'yolo_attention_{pretrained_size}',
        'patience': 30,
        'cos_lr': True,
        'warmup_epochs': 2.0,
        'lr0': 0.01,
        'lrf': 0.001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'close_mosaic': 10,
        'augment': True,
        'mixup': 0.1,
        'copy_paste': 0.1
    }
    results = model.train(**train_args)
    return model, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model with attention.')
    parser.add_argument('--pretrained_size', type=str, default='m', help='Pretrained model size (e.g., m, l, x)')
    parser.add_argument('--custom_dataset', type=str, default='/content/VOC.yaml', help='Path to custom dataset yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')

    args = parser.parse_args()

    # Call the training function with the arguments from the command line
    
def main():
    prepare_all_datasets()
    model, results = train_with_transfer_learning(
        pretrained_size=args.pretrained_size,
        custom_dataset=args.custom_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz
    )
    predictions=model.val(data="VOC.yaml")
