from typing import Dict

from src.builder.data_loader import DataLoader
from src.builder.prompt_generator import PromptGenerator

from src.model.inference import ModelInference

class TurnController:
  def __init__(self, domain_path: Dict[str, str]):
    self.data_loader = DataLoader(domain_path)
    self.prompt_generator = PromptGenerator()

    # Load data
    self.data = self.data_loader.get_data()

  def run_scenario(self):
    # For each domain:
    for domain, scenarios in self.data.items():
      # For each scenario
      for i, scenario in enumerate(scenarios):
        prompt = self.prompt_generator.generate_prompt(scenario)
        
        # Run base prompt

        output = 'Test model base output'
        self.prompt_generator.create_history(prompt, output)

        # For each update
        for j, update in enumerate(scenario['updates']):
          prompt = self.prompt_generator.generate_prompt(update, is_update=True)
          
          # Run update prompt

          output = 'Test model updated output'          
          self.prompt_generator.update_history(prompt, output)

      self.prompt_generator.delete_history()
      # Write to file
      # Run evaluation on outputs file