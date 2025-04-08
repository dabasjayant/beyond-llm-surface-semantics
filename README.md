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
      "rounds": [
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
