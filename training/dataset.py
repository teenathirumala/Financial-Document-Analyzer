import json
from datasets import Dataset
from transformers import LayoutLMv3Processor
from PIL import Image

LABEL_LIST = ['O', 'B-ADDR', 'B-COMPANY', 'B-DATE', 'B-TOTAL', 'I-ADDR', 'I-COMPANY', 'I-DATE', 'I-TOTAL']
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

class SROIEDataset:
    def __init__(self, json_path, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def to_huggingface_dataset(self):
        def transform(example):
            image = Image.open(example["image_path"]).convert("RGB")
            encoding = self.processor(image, example["words"], boxes=example["bboxes"], truncation=True, padding="max_length", return_tensors="pt")
            encoding = {k: v[0] for k, v in encoding.items()}
            labels = [LABEL_TO_ID.get(l, 0) for l in example["ner_tags"]]
            labels += [0] * (512 - len(labels))  # pad to max length
            encoding["labels"] = labels[:512]
            return encoding

        dataset = Dataset.from_list(self.data)
        return dataset.map(transform)

