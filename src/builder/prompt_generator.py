class PromptGenerator:
  def __init__(self):
    self.history = ''

  def create_history(self, prompt, output):
    self.history = f'Our conversation history: \n\n[User]: \n{prompt} \n\n{output} \n\n\n'

  def update_history(self, prompt, output):
    self.history = f'{prompt} \n\n{output} \n\n\n'

  def get_history(self):
    return self.history

  def delete_history(self):
    self.history = ''

  def generate_prompt(self, data, is_update=False):
    prompt = self.history + '\n\nUpdated information:\n\n' if is_update else ''

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

    prompt += '\nAnswer the below question as "yes" or "no" based on the given facts, rules, and preferences. Output should always start with "Answer: yes" or "Answer: no".\n'
    prompt += '\nExample: output should always be in the format "Answer: yes" or "Answer: no"\n'

    # Add the question
    prompt += f"\nQuestion: Is this correct (yes or no)? {data['question']}\n\n"

    return prompt