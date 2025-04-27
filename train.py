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

model, results = train_with_transfer_learning()
