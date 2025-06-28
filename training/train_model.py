from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, TrainingArguments, Trainer
from dataset import SROIEDataset, LABEL_TO_ID
import os

MODEL_NAME = "microsoft/layoutlmv3-base"

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
dataset_loader = SROIEDataset("training/sroie_train.json", processor)
dataset = dataset_loader.to_huggingface_dataset()

model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_TO_ID))

training_args = TrainingArguments(
    output_dir="models/layoutlmv3-finetuned-sroie",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_dir="logs",
    save_strategy="epoch",
    logging_steps=10,
    # evaluation_strategy="no",
    # fp16=True if torch.cuda.is_available() else False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("models/layoutlmv3-finetuned-sroie")
processor.save_pretrained("models/layoutlmv3-finetuned-sroie")

print("Finetuned model saved to models/layoutlmv3-finetuned-sroie")
