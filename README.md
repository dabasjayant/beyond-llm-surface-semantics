# Beyond Surface Semantics: <br/>Evaluating LLMs on Complex Causal Reasoning

This project introduces the Real-time Adaptive Causal Thinking (ReACT) Benchmark, a dynamic evaluation framework for assessing LLMs on real-time adaptive causal reasoning across multi-round, evolving scenarios.

## Introduction
Recent advancements in large language models (LLMs) show strong performance on various structured reasoning tasks, often rivaling human capabilities. However, most evaluations assume relatively fixed or consistent datasets, overlooking how causal relationships can shift or conflict in real-world scenarios.

To address this gap, our study systematically assesses LLMs under a Real-Time Adaptation Challenge of Memory Consistency. Unlike existing benchmarks focusing on static rules or single-step causal inferences, our approach involves multi-layered, evolving sequences of events with changing rules, shifting preferences, and sometimes contradictory information. This setup tests whether LLMs can retain prior context and adapt their reasoning to new, potentially conflicting details over time—an important step toward real-world utility, where conditions are rarely static or unambiguous.

## Dataset schema
As part of the experimental design, we introduced a dynamic, variable-length causal evolution framework that simulates real-world complexity through escalating task difficulty and contextual shifts across dialogue rounds.
```
{
  "history" : [
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
  "medicine" : [ ... ],
  "policy": [ ... ],
  "science": [ ... ],
  "sports": [ ... ]
}
```
