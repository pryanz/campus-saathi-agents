# Formal Problem Formulation: Campus Saathi

## 1. Initial State ($s_0$)

- **Location:** Entry gate (configurable).
- **Completed Steps:** Empty set ($\emptyset$).
- **Time:** $t = 0$.

## 2. Actions ($A(s)$)

- For a given state, the action space consists of choosing the next uncompleted reporting step room to visit and navigating via the shortest path on the campus graph.

## 3. Transition Model ($T(s, a)$)

- Moves the agent from the current room to the target room via graph edges, adds edge travel time, and accounts for FIFO queue waiting time and service time at the destination counter.

## 4. Goal Test ($G(s)$)

- Returns `True` if all 10 reporting steps (6 academic, 4 hostel) are marked as completed.

## 5. Path Cost Function ($c$)

- Sum of edge travel times, service times, and queue waiting times incurred across all state transitions until the goal state is reached.
