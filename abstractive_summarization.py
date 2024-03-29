# -*- coding: utf-8 -*-
"""Abstractive Summarization

### Load Input Doc (Extractive Summary)
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

import nltk
nltk.download('punkt')

import evaluate
rouge_score = evaluate.load("rouge")

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
from nltk.tokenize import sent_tokenize

# Load the Extracted Dataset
extract_dataset = load_dataset('maulinnasari/dataset_ext_50_mn')

# Preprocess the 'document' field to join the paragraphs into a single paragraph
extract_dataset = extract_dataset.map(lambda example: {'document': ' '.join(example['document']), 'summary': example['summary']})

"""### Abstractive Summarization"""

model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

max_input_length = 1024
max_target_length = 128

def tokenize_function(data):
    model_inputs = tokenizer(
        data['document'],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        data['summary'],
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = extract_dataset.map(tokenize_function, batched=True)

def show_samples(dataset, split_name, num_samples=3, seed=42):
    sample = dataset[split_name].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Document: {example['document']}'")
        print(f"'>> Summary: {example['summary']}'")

show_samples(tokenized_datasets, split_name="train")

batch_size = 4
num_train_epochs = 10
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=0.00001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(
    extract_dataset["train"].column_names
)
tokenized_datasets.column_names

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("ext_50")
