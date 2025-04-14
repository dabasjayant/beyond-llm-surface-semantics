import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast

# Define the DepthSensitiveLoss class (from Evaluate.py)
class EvaluationEngine(nn.Module):
  def __init__(self, alpha=0.5, epsilon=1e-6):
    """
    alpha: weight between WBCE and CWL (0 = only CWCE, 1 = only WBCE)
    """
    super().__init__()
    self.alpha = alpha
    self.epsilon = epsilon

  def weighted_bce(self, y_pred, y_true, depth_weights):
    # log-likelihood terms
    log_preds = torch.log(y_pred + self.epsilon)
    log_one_minus = torch.log(1 - y_pred + self.epsilon)
    bce = - (y_true * log_preds + (1 - y_true) * log_one_minus)
    weighted_bce = depth_weights * bce  # shape (B, N)
    return weighted_bce.mean()

  def continuity_weighted_loss(self, y_pred, y_true):
    """
    Rewards longer correct subsequences.
    Normalized by N (total rounds).
    """
    B, N = y_true.shape       
    assert B == 1
    
    y_pred_bin = (y_pred > 0.5).float()
    correct = (y_pred_bin == y_true).float()

    # Longest contiguous subsequence of correct predictions
    streak = 0
    max_streak = np.float64(0)

    for val in correct[0]:
      if val == 1:
        streak += 1
        max_streak = max(max_streak, streak)
      else:
        streak = 0

    cwl = 1 - (max_streak / N)  # Loss = 1 - (reward ratio)
    return cwl

  def forward(self, y_pred, y_true, depth_weights):
    wbce = self.weighted_bce(y_pred, y_true, depth_weights)
    cwl = self.continuity_weighted_loss(y_pred, y_true)
    return self.alpha * wbce + (1 - self.alpha) * cwl


  # Function to extract the predicted probability from the classifier output string
  def extract_probability(self, row):
    # Parse the classifier output (a string dictionary) to a real dictionary.
    prob_dict = ast.literal_eval(row['classifier_output'])
    # Assume that y_true uses 1 (or "Yes") to indicate the positive class.
    # Adjust these conditions if your ground truth format differs.
    if row['actual'] in [1, 'Yes', 'yes']:
      return prob_dict['Yes']
    else:
      return prob_dict['No']


  # ---- Helper function to compute loss per scenario ----
  def compute_loss_for_scenario(self, df_scenario, loss_fn, beta=1.0):
    """
    df_scenario: DataFrame for one scenario. Must have columns: 'round', 'y_true', and 'classifier_output'
    loss_fn: Instance of DepthSensitiveLoss.
    beta: exponent for depth-based weighting.
    """
    # Sort the rounds (assuming 'round' indicates the order)
    df_scenario = df_scenario.sort_values('round')
    N = len(df_scenario)
    if N == 0:
      return np.nan

    # Create a new column for predicted probability by parsing the classifier output
    df_scenario = df_scenario.copy()  # To avoid SettingWithCopyWarning
    df_scenario['predicted_probability'] = df_scenario.apply(extract_probability, axis=1)
    
    # Convert ground truth and predicted probability columns to tensors
    y_true = torch.tensor(df_scenario['actual'].values, dtype=torch.float32).unsqueeze(0)  # shape (1, N)
    y_pred = torch.tensor(df_scenario['predicted_probability'].values, dtype=torch.float32).unsqueeze(0)  # shape (1, N)

    # Compute depth weights: for round i use ( (i+1)/N )^beta (1-indexed rounds)
    weights = [((i + 1) / N) ** beta for i in range(N)]
    depth_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)  # shape (1, N)

    # Return the computed loss value
    loss = loss_fn(y_pred, y_true, depth_weights)
    return loss.item()


  def get_loss(self, result_path, output_path):
    # Load the CSV file (update the file path if needed)
    df = pd.read_csv(result_path)

    # Assumptions for the CSV columns:
    # 'domain'             : domain of the scenario.
    # 'scenario'           : identifier for the scenario.
    # 'round'              : round number (integer for proper ordering).
    # 'y_true'             : ground truth label (should be 1/"Yes" for positive, 0/"No" for negative).
    # 'classifier_output'  : a string representing a dictionary with keys "Yes" and "No".

    # Instantiate the depth-sensitive loss function
    loss_fn = DepthSensitiveLoss(alpha=0.5)
    beta = 1.0  # Adjust beta for depth weighting if necessary

    results = []  # To store loss per scenario

    # Group the results by domain and scenario
    for (domain, scenario), group in df.groupby(['domain', 'scenario']):
      loss_value = compute_loss_for_scenario(group, loss_fn, beta=beta)
      results.append({
        'domain': domain,
        'scenario': scenario,
        'loss': loss_value,
        'num_rounds': len(group)
      })

    # Create a DataFrame for the results and print/save as needed
    results_df = pd.DataFrame(results)
    print(results_df)
    # Optionally, save the results to a CSV file:
    results_df.to_csv(output_path, index=False)

