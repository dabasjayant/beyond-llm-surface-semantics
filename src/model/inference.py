from transformers import pipeline
from transformers import Pipeline, AutoTokenizer, AutoModelForCausalLM

class ModelInference:
  def __init__(self, model='meta-llama/Llama-3.2-1B'):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModelForCausalLM.from_pretrained(model)
    # self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
    # self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
    self.mode.eval()

  def chat(self, message):
    inputs = self.tokenizer(message, return_tensors='pt')
    # Generate
    generate_ids = self.model.generate(inputs.input_ids)
    return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    