from dotenv import load_dotenv

from src.turn_controller import TurnController
from src.evaluation import EvaluationEngine

def main():
  load_dotenv(dotenv_path='.env')

  domain_path = {
    'behavior': 'data/behavior.json',
    'healthcare': 'data/healthcare.json',
    'sports': 'data/politics.json',
    'science': 'data/science.json',
    'politics': 'data/politics.json',
  }

  controller = TurnController(domain_path)

  controller.run_scenario()

  output_path = 'outputs/output.csv'
  result_path = 'outputs/loss_by_scenario.csv'
  evaluation_engine = controller.get_loss(output_path, result_path)

if __name__ == "__main__":
  main()