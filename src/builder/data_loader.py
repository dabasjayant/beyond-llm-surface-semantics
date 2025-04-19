import json
from typing import Dict, Optional
import random

class DataLoader:
  def __init__(self, domain_path: Dict[str, str], *, seed: Optional[int]=None):
    self.domain_path = domain_path
    self.seed = seed
    self.data = {}
    self.load_data()

  def set_data(self, data):
    self.data = data

  def get_data(self):
    return self.data

  def load_data(self, shuffle: bool=True):
    all_data = {}

    for domain, path in self.domain_path.items():
      try:
        with open(path, 'r') as file:
          data = json.load(file)

          assert isinstance(data, list)
          if shuffle:
            if self.seed is None:
              random.shuffle(data)
            else:
              rng = random.Random(self.seed)
              rng.shuffle(data)

          all_data[domain] = data
      except FileNotFoundError:
        print(f'Error: File not found: {self.file_path}')
      except json.JSONDecodeError:
        print(f'Error: Invalid JSON format in: {self.file_path}')

    self.set_data(all_data)


if __name__ == "__main__":
    # Example 1: global control only
    random.seed(99)
    loader_global = DataLoader({"healthcare": "data/healthcare.json"})

    # load without shuffle
    loader_global.load_data(shuffle=False)
    orig_global = loader_global.get_data()["healthcare"]

    # then shuffle
    loader_global.load_data(shuffle=True)
    data_a = loader_global.get_data()["healthcare"]

    # check that shuffle actually changed the order
    if data_a == orig_global:
        print("⚠️ Global shuffle did NOT change the order.")
    else:
        print("✅ Global shuffle changed the order.")

    # Example 2: per‑loader reproducibility
    loader_local = DataLoader({"healthcare": "data/healthcare.json"}, seed=42)

    # load without shuffle
    loader_local.load_data(shuffle=False)
    orig_local = loader_local.get_data()["healthcare"]

    # shuffle twice
    loader_local.load_data(shuffle=True)
    data_b1 = loader_local.get_data()["healthcare"]
    loader_local.load_data(shuffle=True)
    data_b2 = loader_local.get_data()["healthcare"]

    # reproducibility check
    assert data_b1 == data_b2, "❌ Per‑loader shuffles with same seed differ!"
    print("✅ Per‑loader shuffle reproducibility confirmed.")

    # check that the per‑loader shuffle actually changed the order
    if data_b1 == orig_local:
        print("⚠️ Per‑loader shuffle did NOT change the order.")
    else:
        print("✅ Per‑loader shuffle changed the order.")