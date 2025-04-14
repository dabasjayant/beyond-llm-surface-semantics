import csv
import json
import os
from collections import defaultdict
from typing import Dict

from src.builder.data_loader import DataLoader
from src.builder.prompt_generator import PromptGenerator

from src.model.inference import ModelInference

class TurnController:
  def __init__(self, domain_path: Dict[str, str]):
    self.data_loader = DataLoader(domain_path)
    self.prompt_generator = PromptGenerator()

    self.model = ModelInference()

    # Load data
    self.data = self.data_loader.get_data()
    self.fieldnames = ['domain', 'scenario', 'round', 'classifier_output', 'predicted', 'actual', 'response']
    self.results = []

    self.output_path = 'outputs/output.csv'

  def run_scenario(self):
    # For each domain:
    for domain, scenarios in self.data.items():
      print(f'Running domain: {domain} with {len(scenarios)} scenarios...')
      # For each scenario
      for i, scenario in enumerate(scenarios):
        prompt = self.prompt_generator.generate_prompt(scenario)
        
        # Run base prompt
        response = self.model.chat(prompt)
        output = self.model.evaluate_response(response)
        self.results.append({
          'domain': domain,
          'scenario': i+1,
          'round': 1,
          'classifier_output': dict(list(output.items())[:2]),
          'predicted': output['Bestlabel'],
          'actual': 1 if scenario['label'] else 0,
          'response': response
        })
        
        self.prompt_generator.create_history(prompt, response)

        # For each update
        for j, update in enumerate(scenario['updates']):
          prompt = self.prompt_generator.generate_prompt(update, is_update=True)
          
          # Run update prompt
          response = self.model.chat(prompt)
          output = self.model.evaluate_response(response)
          self.results.append({
            'domain': domain,
            'scenario': i+1,
            'round': j+2,
            'classifier_output': dict(list(output.items())[:2]),
            'predicted': output['Bestlabel'],
            'actual': 1 if update['label'] else 0,
            'response': response
          })
          
          self.prompt_generator.update_history(prompt, response)
        self.prompt_generator.delete_history()

    # Write to file
    print(f"Writing outputs to '{self.output_path}...'")
    self.write_results(format='csv', filepath=self.output_path)

    # Run evaluation on outputs file
    print('Evaluating results:')
    print('Done.')

  
  def write_results(self, format='csv', filepath='outputs/output.csv'):
    # Validate file extension using format
    filepath = self._sanitize_filename(format, filepath)
    if os.path.exists(filepath):
        os.remove(filepath)

    if format == 'csv':
      with open(filepath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=self.fieldnames)
        writer.writeheader()
        writer.writerows(self.results)
    elif format == 'json':
      transformed_data = self._transform_json(self.results)
      with open(filepath, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    else:
      raise ValueError(f"{format} format is not supported. Please choose either 'csv' or 'json'")
    
  def _transform_json(self, data_list):
    result = defaultdict(lambda: defaultdict(list))
    for entry in data_list:
        result[entry['domain']][entry['scenario']].append({
            'round': entry['round'],
            'predicted': entry['predicted'],
            'actual': entry['actual']
        })
    return {d: dict(scenarios) for d, scenarios in result.items()}
  
  def _sanitize_filename(self, format: str, filename: str):
    format = format.lower()
    if format not in {'csv', 'json'}:
        raise ValueError("Unsupported format. Expected 'csv' or 'json'.")
    
    # Remove any existing extension
    base = filename.rsplit('.', 1)[0]
    
    # Append correct extension
    return f'{base}.{format}'