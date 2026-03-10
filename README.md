# Explainable AI (INFOMXAI) - Project 1

Authors: Andrei Medesan, Janan Jahed, Joly-Eline Himpers, Stefan Durlanescu


This project implements explainability-by-design algorithms for AI decision-making systems, enabling agents to make socially appropriate decisions and explain them in natural language to non-expert users. The project is divided into two parts, namely four code assignments implementing the core explainability-by-design pipeline and a natural language explanation generator that builds on top of the former part.

--- 

## Part 1 — Assignments

All assignments use **Python 3** and the [`anytree`](https://pypi.org/project/anytree/) library. Agent behaviour is represented as goal trees, i.e., tree structures where inner nodes are goals (of type `SEQ`, `AND`, or `OR`) and leaf nodes are actions (`ACT`). Each node may carry attributes such as preconditions, postconditions, costs, and links to dependent nodes.

- **assignment1.py**: enumerates all possible execution traces an agent could follow, by traversing the goal tree from a given starting node according to its semantics.
- **assignment2.py**: annotates a goal tree based on whether nodes violate a given social norm. 
- **assignment3.py**: determines the optimal execution trace for an agent to follow, taking into account the agent's initial beliefs and the goal to achieve (via pre/postconditions)
- **assignment4.py**: generates a formal explanation for why a specific action was selected as part of the trace chosen in assignment 3. The explanation is a list of typed explanatory factors:

| Factor | Description |
|--------|-------------|
| `P` | preconditions that made an action possible |
| `C` | why a specific alternative was chosen at an OR node |
| `V` | value comparison between chosen and unchosen alternatives |
| `N` | an alternative was rejected due to a norm violation |
| `F` | an alternative was rejected due to unsatisfied preconditions |
| `L` | a link between an action and a dependent node |
| `D` | the goal node that the action contributes to |
| `U` | the user's preference ordering |


## Part 2 — Natural Language Explanation Generator

This builds on Assignment 4 to translate formal explanations into plain English, targeting non-expert end-users with no knowledge of the agent's internal mechanisms. We used a hybrid approach that takes the generated structured factors from assignment 4 and through a rule-based implementation, each factor type maps to a sentence template. Finally, these explanations are further enhanced by a prompted Gemini synthesis that rewrites everything into a coherent and understandable text for the user. The generated explanations can be found in the file `src/explanations_output.txt`. The code for natural language explanations can be found in `src/part2.py`.

---

## Getting Started

Install dependencies:
```bash
pip install anytree
```

Or use Anaconda to create a virtual environment:
```bash
cd XAI
conda env create -f environment.yml
conda activate XAI_env
```

---

## References

Winikoff, M., Sidorenko, G., Dignum, V., & Dignum, F. (2021). *Why bad coffee? Explaining BDI agent behaviour with valuings.* Artificial Intelligence, 300, 103554. https://doi.org/10.1016/j.artint.2021.103554