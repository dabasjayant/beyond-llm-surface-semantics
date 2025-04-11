class PromptGenerator:
  def __init__(self):
    self.history = ''

  def create_history(self, prompt, output):
    self.history = f'Our conversation history: \n\n[User]: \n{prompt} \n\n[Your answer]: \n{output} \n\n\n'

  def update_history(self, prompt, output):
    self.history = self.history + f'[User]: \n{prompt} \n\n[Your answer]: \n{output} \n\n\n'

  def get_history(self):
    return self.history

  def delete_history(self):
    self.history = ''

  def generate_prompt(self, data, is_update=False):
    prompt = 'Updated information:\n\n' if is_update else 'Answer the following question based on the given facts, rules, and preferences. Provide a true or false answer (binary cross-entropy).\n\n'

    # Add facts to the prompt
    prompt += 'Facts:\n'
    for fact in data['facts']:
      prompt += f'- {fact}\n'

    # Add rules to the prompt
    prompt += '\nRules:\n'
    for rule in data['rules']:
      prompt += f'- {rule}\n'

    # Add preferences to the prompt
    prompt += '\nPreferences:\n'
    for preference in data['preferences']:
      prompt += f'- {preference}\n'

    # Add the question
    prompt += f"\nQuestion: {data['question']}\n"

    return prompt