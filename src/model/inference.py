import os

import nltk
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelInference:
  def __init__(self, model='meta-llama/Llama-3.2-3B-Instruct'):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.classifier = pipeline(
      "zero-shot-classification",
      model="facebook/bart-large-mnli"
      )
    self.model = AutoModelForCausalLM.from_pretrained(model, token=os.environ.get('HF_TOKEN'))
    self.model.eval()

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # Use GPU if it's available
    self.device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else None
    self.model.to(self.device)

  def chat(self, message, max_new_tokens=100):
    inputs = self.tokenizer(message, return_tensors='pt').to(self.device)
    outputs = self.model.generate(
      **inputs, 
      max_new_tokens=max_new_tokens,
      pad_token_id=self.tokenizer.eos_token_id
    )
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(message):].strip().lower()
  
  def evaluate_response(self, response, max_new_tokens=100):
    if not response:
      return{
        'Yes': None,
        'No': None,
        'Bestlabel': None
    }
    candidate_labels = ['Yes', 'No']
    classification = self.classifier(response, candidate_labels)
    # Create a mapping from label to score.
    label_scores = dict(zip(classification['labels'], classification['scores']))
    
    # Select the best label based on the highest score.
    best_label = max(label_scores, key=label_scores.get)
    best_label_value = 1 if best_label == 'Yes' else 0
    
    return {
        'Yes': round(label_scores.get('Yes'), 4),
        'No': round(label_scores.get('No'), 4),
        'Bestlabel': best_label_value
    }
  
    