# imports
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments


# load saved model
save_directory = "models/mbart"

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

# loding a pretrained model (own or huggingface's)
ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
ft_tokenizer = AutoTokenizer.from_pretrained(save_directory)

classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
res = classifier(["मुझे यह नजारा बहुत पसंद है"])
print(res)