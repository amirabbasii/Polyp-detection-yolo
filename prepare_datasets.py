import os
import cv2
import json
import shutil
from skimage import io
from skimage.measure import label, regionprops, find_contours
from ultralytics import YOLO

# Utility functions
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border

def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])
    return bboxes

def create_data_directories():
    if not os.path.exists('data'):
        for folder in ['images', 'labels']:
            for split in ['train', 'val','test']:
                os.makedirs(f'data/{folder}/{split}')

# Kvasir-SEG dataset preparation
def prepare_kvasir_seg():
    f = open('/content/Kvasir-SEG/kavsir_bboxes.json')
    json_file = json.load(f)
    
    create_data_directories()

    for i, name in enumerate(json_file.keys()):
        img_width = json_file[name]['width']
        img_height = json_file[name]['height']
        
        ans = ""
        for bbox in json_file[name]['bbox']:
            w = bbox['xmax'] - bbox['xmin']
            h = bbox['ymax'] - bbox['ymin']
            x_center = bbox['xmin'] + w / 2
            y_center = bbox['ymin'] + h / 2

            x_center = x_center / img_width
            y_center = y_center / img_height
            w = w / img_width
            h = h / img_height

            ans += f'0 {x_center} {y_center} {w} {h}\n'

        if i < 900:
            section = "train"
        elif i < 900 + 90:
            section = "val"
        else:
            section = "test"

        source_image_path = f'/content/Kvasir-SEG/images/{name}.jpg'
        shutil.copy(source_image_path, f'data/images/{section}/Kavir_{name}.jpg')

        with open(f'data/labels/{section}/Kavir_{name}.txt', 'w') as f:
            f.write(ans)

# CVC-ClinicDB dataset preparation
def prepare_cvc_clinicdb():
    images = os.listdir("/content/CVC-ClinicDB/Original")
    create_data_directories()

    for i, name in enumerate(images):
        image = io.imread(f'/content/CVC-ClinicDB/Original/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'/content/CVC-ClinicDB/Ground Truth/{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w = x2 - x
            h = y2 - y

            x = x + w / 2
            y = y + h / 2
            x = x / image.shape[1]
            y = y / image.shape[0]
            w = w / image.shape[1]
            h = h / image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 550:
            section = "train"
        elif i < 550 + 55:
            section = "val"
        else:
            section = "test"

        cv2.imwrite(f'data/images/{section}/CVC-ClinicDB_{name.replace("tif", "jpg")}', image)
        with open(f'data/labels/{section}/CVC-ClinicDB_{name.replace(".tif", ".txt")}', 'w') as f:
            f.write(ans)

# ETIS dataset preparation
def prepare_etis():
    create_data_directories()

    for i, name in enumerate(os.listdir("/content/ETIS-LaribPolypDB/ETIS-LaribPolypDB")):
        image = io.imread(f'/content/ETIS-LaribPolypDB/ETIS-LaribPolypDB/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'/content/ETIS-LaribPolypDB/Ground Truth/p{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w = x2 - x
            h = y2 - y

            x = x + w / 2
            y = y + h / 2
            x = x / image.shape[1]
            y = y / image.shape[0]
            w = w / image.shape[1]
            h = h / image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 100:
            section = "train"
        elif i < 100 + 10:
            section = "val"
        else:
            section = "test"

        cv2.imwrite(f'data/images/{section}/ETIS_{name.replace("tif", "jpg")}', image)
        with open(f'data/labels/{section}/ETIS_{name.replace(".tif", ".txt")}', 'w') as f:
            f.write(ans)

# CVC-ColonDB dataset preparation
def prepare_cvc_colondb():
    path = '/content/CVC-ColonDB'  # Assuming the dataset has been downloaded
    create_data_directories()

    for i, name in enumerate(os.listdir(f'{path}/CVC-ColonDB/images')):
        image = io.imread(f'{path}/CVC-ColonDB/images/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'{path}/CVC-ColonDB/masks/{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w = x2 - x
            h = y2 - y

            x = x + w / 2
            y = y + h / 2
            x = x / image.shape[1]
            y = y / image.shape[0]
            w = w / image.shape[1]
            h = h / image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 300:
            section = "train"
        elif i < 300 + 30:
            section = "val"
        else:
            section = "test"

        cv2.imwrite(f'data/images/{section}/CVC-ColonDB_{name.replace("tif", "jpg")}', image)
        with open(f'data/labels/{section}/CVC-ColonDB_{name.replace(".tif", ".txt")}', 'w') as f:
            f.write(ans)
def save_yaml():
    conf = "train: /content/data/images/train\n\nval: /content/data/images/test\n\ntest: /content/data/images/test\n\nnc: 1\n\nnames: ['x']"
    with open("VOC.yaml", "w") as f:
        f.write(conf)
