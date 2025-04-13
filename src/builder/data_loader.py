import json
from typing import Dict

class DataLoader:
  def __init__(self, domain_path: Dict[str, str]):
    self.domain_path = domain_path
    self.data = []
    self.load_data()

  def set_data(self, data):
    self.data = data

  def get_data(self):
    return self.data

  def load_data(self):
    all_data = {}

    for domain, path in self.domain_path.items():
      try:
        with open(path, 'r') as file:
          data = json.load(file)
          all_data[domain] = data
      except FileNotFoundError:
        print(f'Error: File not found: {self.file_path}')
      except json.JSONDecodeError:
        print(f'Error: Invalid JSON format in: {self.file_path}')

    self.set_data(all_data)