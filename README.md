# Beyond Surface Semantics: <br/>Evaluating LLMs on Complex Causal Reasoning

This project introduces the Real-time Adaptive Causal Thinking (ReACT) Benchmark, a dynamic evaluation framework for assessing LLMs on real-time adaptive causal reasoning across multi-round, evolving scenarios.

## Introduction
Recent advancements in large language models (LLMs) show strong performance on various structured reasoning tasks, often rivaling human capabilities. However, most evaluations assume relatively fixed or consistent datasets, overlooking how causal relationships can shift or conflict in real-world scenarios.

To address this gap, our study systematically assesses LLMs under a Real-Time Adaptation Challenge of Memory Consistency. Unlike existing benchmarks focusing on static rules or single-step causal inferences, our approach involves multi-layered, evolving sequences of events with changing rules, shifting preferences, and sometimes contradictory information. This setup tests whether LLMs can retain prior context and adapt their reasoning to new, potentially conflicting details over time—an important step toward real-world utility, where conditions are rarely static or unambiguous.

## Domains
1. **behavior**: Human Behavior and Social Norms <br/>
Why: Common sense, social reasoning, evolving norms. <br/>
Tests: Fuzzy causality, belief updating, implicit rules.

2. **healthcare**: Medicine and Diagnosis <br/>
Why: High-stakes, rule-based, evolving with new information (symptoms, allergies, side effects). <br/>
Tests: Contradictions, probabilistic updates, belief revision.

3. **politics**: Political Strategies <br/>
Why: Involves conflicting rules, dynamic updates, and trade-offs. <br/>
Tests: Rule conflict resolution, context-sensitive preferences.

4. **science**: Science and Technology <br/>
Why: Involves multi-variable, probabilistic, and time-delayed causal relationships. <br/>
Tests: Systemic reasoning, belief updates under uncertainty, and intervention effects.

5. **sports**: Game Rules and Strategy <br/>
Why: Controlled environment with rule changes and turn-based logic. <br/>
Tests: Rule injection, memory over turns, multi-step planning.

## Rounds
Rounds are used to update the initial scenario for an adaptive evaluation. <br/>

## Depth
We use a minimum of 3 to a maximum of 7 update rounds to create samples for complex reasoning tasks while keeping the data generalized by using varying depth.

## Dataset schema
As part of the experimental design, we introduced a dynamic, variable-length causal evolution framework that simulates real-world complexity through escalating task difficulty and contextual shifts across dialogue rounds.
```
{
  "behavior" : [
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
        ...
      ]
    },
    {
      "facts": [],
      "rules": [],
      "preferences": [],
      "question": "",
      "label": boolean,
      "updates": [
        {
          "facts": [],
          "rules": [],
          "preferences": [],
          "question": "",
          "label": boolean
        },
        ...
      ]
    }
  ],
  ...
}
```

## Dataset Generation

- **Synthetic Examples**
- **Model-Based Generation (Few-Shot Method)**

### Synthetic Examples
To begin, we manually design five carefully evaluated scenarios for each domain. These initial scenarios serve as demonstrations for generating the remaining samples. At this stage, we focus on refining the examples and addressing any issues to ensure a smooth generation process later on.

### Model-Based Generation
**Model Used:** `Claude 3.7 Sonnet`

The prompts used in this step can be found in the `data_prompt.txt` file within this repository.

Key considerations during the generation process:
- **Model Exclusion:** To eliminate any potential bias between the model and our ReACT benchmark, the model is excluded from the experiments.
- **Hypothetical Scenario Generation:** We instruct the model to avoid using any public information or common knowledge, ensuring that the generated scenarios remain free of biases embedded during the model's training.
- **Manual Evaluation:** After generation, we manually review the scenarios to validate the labels and confirm the causal relationships are accurate.