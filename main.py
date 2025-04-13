from dotenv import load_dotenv

from src.turn_controller import TurnController

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

if __name__ == "__main__":
  main()