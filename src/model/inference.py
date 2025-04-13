import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelInference:
  def __init__(self, model='meta-llama/Llama-3.2-1B'):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    self.model = AutoModelForCausalLM.from_pretrained(model, token=os.environ.get('HF_TOKEN'))
    self.model.eval()

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # Use GPU if it's available
    self.device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    self.model.to(self.device)

  def chat(self, message, max_new_tokens=5):
    inputs = self.tokenizer(message, return_tensors='pt').to(self.device)
    outputs = self.model.generate(
      **inputs, 
      max_new_tokens=max_new_tokens,
      pad_token_id=self.tokenizer.eos_token_id
    )
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(message):].strip().lower()
  
  def get_sentiment(self, text: str) -> int:
    try:
      nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
      nltk.download('vader_lexicon')
    text = text.lower().strip()
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['pos'] > scores['neg']:
      return 1
    elif scores['pos'] < scores['neg']:
      return 0
    else:
      return 1 if scores['compound'] >= 0.05 else 0
    