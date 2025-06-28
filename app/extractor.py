import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
import torch
from PIL import Image
import os

def load_model():
    # processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    # model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained("models/layoutlmv3-finetuned-sroie")
    processor = LayoutLMv3Processor.from_pretrained("models/layoutlmv3-finetuned-sroie")

    return processor, model

def extract_from_pdf(pdf_path, processor, model):
    label_map = {
        0: "O", 1: "B-VENDOR", 2: "I-VENDOR", 3: "B-TOTAL", 4: "I-TOTAL"
    }
    images = convert_from_path(pdf_path)
    results = []
    for i, img in enumerate(images):
        img.save(f"temp_page_{i}.jpg")
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words, boxes = [], []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 50:
                words.append(ocr_data['text'][i])
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                boxes.append([x, y, x + w, y + h])
        if not words:
            continue
        encoding = processor(img, words, boxes=boxes, return_tensors="pt")
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = processor.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        labels = [label_map[label_id] for label_id in predictions[0].tolist()]
        results.append(list(zip(words, labels)))
    return results
