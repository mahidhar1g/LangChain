# CHAINING: A GUIDE

This document provides an overview of **chaining tasks** for sequential, parallel, and conditional execution, as illustrated in the accompanying visuals.

## Chains Overview

Chaining is the process of linking tasks to achieve a desired outcome. Tasks are executed in specific ways depending on the chaining type:

### Task Examples
1. **Task 1**: Creating a prompt template with placeholders added via `invoke()`.
2. **Task 2**: Passing the finalized prompt to a Language Learning Model (LLM).

The chaining process can link multiple tasks:
- Task 1 leads to Task 2.
- Task 2 connects to Task 3, and so on.

---

## Types of Chaining

### 1. Sequential (Extended) Chaining
- Tasks are performed **one by one** in a straight, linear order.
- Example:
  1. Task A completes.
  2. Task B begins only after Task A finishes.

### 2. Parallel Chaining
- Tasks are executed **simultaneously**, without dependencies.
- Example:
  - Task 1: Brewing coffee.
  - Task 2: Toasting bread.
  - Task 3: Frying an egg.
  - All tasks run independently and converge at serving breakfast.

### 3. Conditional Chaining
- Tasks are executed based on a specific **condition**.
- Example:
  - A user query triggers a condition:
    - **If** it’s a technical issue: Execute the "Technical Issue Chain".
    - **Else if** it’s a billing query: Execute the "Billing Chain".
    - **Else**: Execute the "FAQ Chain".


---

## Summary
Chaining methods enable structured and efficient task execution. By selecting the appropriate chaining type, workflows can be tailored to meet complex requirements:
- Sequential Chaining for dependent tasks.
- Parallel Chaining for independent, simultaneous tasks.
- Conditional Chaining for dynamic branching based on conditions.

---

Feel free to adapt this README to your specific chaining implementation!
