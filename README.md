#  AIDRA — Adaptive Intelligent Disaster Response Agent

> A full-featured AI simulation for disaster triage and rescue operations, built with Python & Tkinter.  
> **CCP AIC-201 Final Project **

---

##  Overview

AIDRA simulates a real-time disaster response scenario on a procedurally generated grid map. Two ambulances are dispatched to rescue victims of varying severity, navigating through obstacles, fires, and blocked roads — all while classical AI algorithms plan optimal routes and ML models predict victim survival.

---

##  Features

| Module | What it does |
|---|---|
| **Search Algorithms** | BFS, DFS, Greedy Best-First, A\*, Hill Climbing — all benchmarked live |
| **Algorithm Selector** | Manual, All (Compare), or Context-Aware auto-selection |
| **Simulated Annealing** | Optimises the global rescue order before operations begin |
| **CSP Allocator** | Assigns ambulances to victims using backtracking + MRV heuristic |
| **Fuzzy Logic** | Computes real-time urgency scores to drive dispatch decisions |
| **kNN Model** | Predicts victim survival probability from learned historical data |
| **Naive Bayes** | Bernoulli NB baseline for survival classification |
| **Dynamic Events** | Random road blocks, fire spread, and new fire spawns mid-mission |
| **KPI Dashboard** | Live metrics: rescued/dead, avg rescue time, risk exposure, resource utilisation |
| **Animated Grid** | Real-time Tkinter visualisation of ambulance trails and victim states |

---

##  AI Modules In Detail

###  Search Algorithms
- **BFS** — Shortest unweighted path, good for low-risk areas
- **DFS** — Fast exploration, non-optimal
- **Greedy** — Heuristic-driven, maximises speed
- **A\*** — Optimal cost + risk weighted path
- **Hill Climbing** — Local greedy, may get stuck

**Context-Aware mode** automatically picks the right algorithm per rescue:
- On-fire cell → Greedy (max speed)
- Critical victim, close → A\*
- Critical victim, far → Greedy
- Minor in low-risk area → BFS
- Default → A\*

**All (Compare) mode** runs all 5 algorithms and selects the best using a multi-objective score:
> 35% path length + 30% risk + 20% nodes expanded + 15% time

---

###  Simulated Annealing
Optimises the global rescue order before the mission starts using multi-objective cost:
- Heuristic distance between consecutive victims
- Urgency penalty (critical victims scheduled late cost more)
- Local risk at each victim's position

---

###  CSP Resource Allocation
**Variables:** Each unrescued victim  
**Domains:** Ambulance IDs {0, 1}  
**Constraints:**
- Max 2 victims per ambulance simultaneously
- Total medic kits used ≤ remaining kits (hard constraint)

**Solver:** Backtracking + MRV (Minimum Remaining Values) + Forward Checking

---

###  Fuzzy Logic Urgency
Three fuzzy rules combine severity, survival probability, and area risk into a `[0, 1]` urgency score:
- `> 0.7` → **DISPATCH NOW**
- `0.4–0.7` → Dispatch with caution
- `< 0.4` → Routine dispatch

---

###  ML Models

**Dataset:** 20,000 synthetic records with Gaussian-distributed features, correlated variables (high risk → longer delay), feature interactions, ~5% missing data (median imputed), and probabilistic sigmoid labels.

| Model | Type | Training |
|---|---|---|
| **kNN** | Instance-based | Stores training split; selects best k via cross-validation |
| **Naive Bayes** | Bernoulli NB | Frequency priors + Laplace smoothing; no iterative training |

Both models predict **survival probability** for each victim at placement time.

---

##  Grid Environment

- **12 × 14** procedurally generated grid
- 20 random obstacles placed at startup
- Gaussian risk values per cell (`μ=0.35, σ=0.25`)
- 2 medical centres at fixed corners
- Ambulance base at `(0, 0)`

Dynamic events during mission:
- **30% chance** per rescue: random road block + fire spread → replanning triggered
- **15% chance** per rescue: new fire spawns → victim risks updated

---

##  Requirements

```
Python 3.8+
tkinter   (included in standard library)
```

No external dependencies needed.

---

##  Running

```bash
python AIDRAFinal.py
```

---

## 🎮 How to Use

1. **Select Algorithm** from the dropdown (BFS / DFS / Greedy / A\* / Hill Climbing / All (Compare) / Context-Aware)
2. **Select Victim Count** (5–15)
3. Click **▶ RUN SIMULATION**
4. Watch the animated grid — ambulances navigate to victims and then to medical centres
5. Read real-time decisions and trade-off analysis in the Decision Log panel
6. View final KPI report when mission completes
7. Click **↺ RESET** to run again with a new map

---

##  KPIs Tracked

- Victims rescued / dead
- Average rescue time (steps)
- Total risk exposure
- Medic kits used / remaining (hard cap: 10)
- Resource utilisation rate
- Total replanning events
- Per-algorithm usage count
- Per-victim survival delta (initial → final probability)

---
## Demo Video
https://www.linkedin.com/posts/kashf-noor-55b520310_artificialintelligence-disastermanagement-ugcPost-7459284843030474753-6C2s?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE8ZYRcB67Ktcr7NgVl6Qk0V-trStqEnZl8

---

##  Course

**CCP AIC-201** — Artificial Intelligence  
Final Project — v6
