from dotenv import load_dotenv
import random
import numpy as np
import torch

from src.turn_controller import TurnController
from src.evaluation import EvaluationEngine

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed all the usual RNGs for reproducibility.

    :param seed: the master seed
    :param deterministic: if True, force CUDA/CuDNN into deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # these two ensure deterministic CUDA convolution (at cost of performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
  load_dotenv(dotenv_path='.env')

  SEED = 42
  set_global_seed(SEED)

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