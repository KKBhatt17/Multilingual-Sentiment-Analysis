# imports
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

training_texts=[]
training_labels=[]
for i in range(0,45000):
  training_texts.append(indic_df.iloc[i]['text'])
  training_labels.append(indic_df.iloc[i]['sentiment'])

train_texts, val_texts, train_labels, val_labels = train_test_split(training_texts, training_labels, test_size=0.2, shuffle=True)

# train_texts=train_texts[0:30000]
# val_texts=val_texts[0:4000]
# train_labels=train_labels[0:30000]
# val_labels=val_labels[0:4000]


class TweetDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
  
  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item
  
  def __len__(self):
    return len(self.labels)


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
#model_name = "distilbert-base-uncased-finetuned-sst-2-english"

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)
# test_dataset = TweetDataset(test_encodings, test_labels)


## define metric
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/DLNLP_Project/finetuned_models/mbart/results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    eval_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
)


model = AutoModelForSequenceClassification.from_pretrained(model_name)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics=compute_metrics
)

# torch.cuda.empty_cache()

print(trainer.train())

# save model
save_directory = "models/mbart"