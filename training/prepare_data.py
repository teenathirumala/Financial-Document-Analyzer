import os
import json
from PIL import Image
from datasets import Dataset

def parse_sroie_label_file(label_file):
    words, boxes, labels = [], [], []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            word = parts[0]
            x0, y0, x1, y1 = map(int, parts[1:5])
            label = parts[-1]
            words.append(word)
            boxes.append([x0, y0, x1, y1])
            labels.append(label)
    return words, boxes, labels

def create_json_from_sroie(data_dir):
    images = os.listdir(os.path.join(data_dir, 'train'))
    examples = []
    for image_file in images:
        if not image_file.endswith('.jpg') and not image_file.endswith('.png'):
            continue
        image_path = os.path.join(data_dir, 'train', image_file)
        label_path = os.path.join(data_dir, 'train_labels', image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        if not os.path.exists(label_path):
            continue
        words, boxes, ner_tags = parse_sroie_label_file(label_path)
        examples.append({
            'id': image_file,
            'image_path': image_path,
            'words': words,
            'bboxes': boxes,
            'ner_tags': ner_tags
        })
    with open('training/sroie_train.json', 'w') as f:
        json.dump(examples, f, indent=2)
    print("✅ Created training/sroie_train.json")

if __name__ == '__main__':
    create_json_from_sroie('data/SROIE2019')

