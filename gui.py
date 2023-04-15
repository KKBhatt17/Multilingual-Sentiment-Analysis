## This file contains code for connecting the model in the background to the GUI which is web interface (Anvil).



## Dependencies
# pip install transformers
# pip install anvil-uplink
# pip install datasets


import anvil.server
from transformers import pipeline
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# The key given here changes every session of the wen interface.
key = "L2AIJBPJKPLA6JQVXY67GMD2-A4M53V53WWH5NATO"
anvil.server.connect(key)


# @anvil.server.callable
# def predict_sentiment(input_text):
#   save_directory = "/content/drive/MyDrive/DLNLP_Project/finetuned_models/distilbert-base-uncased-finetuned-sst-2-english-1"
#   ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
#   ft_tokenizer = AutoTokenizer.from_pretrained(save_directory)
#   classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
#   result = classifier([input_text])
#   sentiment = result[0]['label']
#   score = result[0]['score']
#   print(input_text)
#   print(sentiment)
#   #sentiment = "Positive"
#   return sentiment, score



@anvil.server.callable
def predict_sentiment(input_text,selected_model):
  if selected_model=="DistilBERT":
    save_directory = "/content/drive/MyDrive/DLNLP_Project/finetuned_models/distilbert-base-uncased-finetuned-sst-2-english-1"
    ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    ft_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
    result = classifier([input_text])
    sentiment = result[0]['label']
    score = result[0]['score']
    print(input_text)
    print(sentiment)
    return sentiment, score
  
  elif selected_model=="mBERT":
    save_directory = "/content/drive/MyDrive/DLNLP_Project/finetuned_models/mbert"
    ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    ft_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    classifier = pipeline("sentiment-analysis", model=ft_model, tokenizer=ft_tokenizer)
    result = classifier([input_text])
    star_value = result[0]['label']
    score = result[0]['score']
    if star_value=='1 star':
      sentiment='NEGATIVE'
    else:
      sentiment='POSITIVE'
    print(input_text)
    print(sentiment)
    return sentiment, score

# This line keeps the background process running and waiting for any input given from the GUI.
anvil.server.wait_forever()