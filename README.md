# beyond-llm-surface-semantics
Beyond Surface Semantics: Evaluating LLMs on Complex Causal Reasoning


## Dataset schema

```
{
  "facts": [],
  "rules": ["Rule 1: This is first rule", "Rule 2: This is second rule"],
  "preferences": [],
  "question": "",
  "label": boolean,
  "rounds": [
    {
      "facts": [],
      "rules": ["Rule 2: Updated second rule", "Rule 3: Another rule"],
      "preferences": ["", ""],
      "question": "",
      "label": boolean
    },
    {
      "facts": [],
      "rules": [],
      "preferences": [],
      "question": "Updated goal",
      "label": boolean
    }
  ]
}
```
