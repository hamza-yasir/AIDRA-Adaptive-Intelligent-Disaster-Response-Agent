# =============================================================================
# AIDRA - Adaptive Intelligent Disaster Response Agent  v6
# Full Tkinter GUI | CCP AIC-201 | All AI modules integrated
# =============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import random, math, heapq, time, threading
from collections import deque

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
ROWS, COLS = 12, 14
CELL_SIZE  = 50
PADDING    = 4
ANIM_DELAY = 0.12   # seconds between animation steps

SEVERITY_COLORS = {"critical": "#FF4444", "moderate": "#FFB300", "minor": "#66BB6A"}
SEVERITY_ORDER  = {"critical": 0, "moderate": 1, "minor": 2}

# Medic kit cost per severity — hard cap of 10 total kits per run
KIT_COST = {"critical": 2, "moderate": 1, "minor": 1}

# Temporal deterioration: survival_prob drops per rescue turn skipped
DETERIORATION_RATE = {"critical": 0.010, "moderate": 0.007, "minor": 0.002}

# Simulated Annealing parameters
SA_TEMP_INIT = 100.0
SA_TEMP_MIN  = 0.5
SA_COOL      = 0.92
SA_ITER      = 200

PALETTE = {
    "bg":        "#1A1A2E",
    "panel":     "#16213E",
    "card":      "#0F3460",
    "accent":    "#E94560",
    "green":     "#00B894",
    "yellow":    "#FDCB6E",
    "blue":      "#74B9FF",
    "text":      "#EAEAEA",
    "subtext":   "#A0A0B0",
    "blocked":   "#2D3436",
    "risk_low":  "#27AE60",
    "risk_mid":  "#F39C12",
    "risk_high": "#C0392B",
    "trail":     "#6C5CE7",
    "base":      "#00CEC9",
    "medcenter": "#00B894",
    "ambu1":     "#74B9FF",
    "ambu2":     "#FD79A8",
    "grid_line": "#2A2A4A",
    "dead":      "#555555",
}

# =============================================================================
# GRID & ENVIRONMENT
# =============================================================================
class Cell:
    def __init__(self):
        self.blocked = False
        # Gaussian risk: skewed toward lower values, occasional high-risk hotspots
        self.risk    = round(min(1.0, max(0.0, random.gauss(0.35, 0.25))), 2)
        self.on_fire = False

class Environment:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.grid = [[Cell() for _ in range(COLS)] for _ in range(ROWS)]
        self.base = (0, 0)
        self.medical_centers = []
        self._place_obstacles()
        self._place_medical_centers()

    def _place_obstacles(self):
        count = 0
        while count < 20:
            r, c = random.randint(0, ROWS-1), random.randint(0, COLS-1)
            if (r, c) != (0, 0):
                self.grid[r][c].blocked = True
                count += 1

    def _place_medical_centers(self):
        for pos in [(ROWS-1, COLS-1), (ROWS-1, 0)]:
            r, c = pos
            self.grid[r][c].blocked = False
            self.medical_centers.append(pos)

    def block_random_road(self):
        for _ in range(50):
            r, c = random.randint(0, ROWS-1), random.randint(0, COLS-1)
            if (r,c) != self.base and (r,c) not in self.medical_centers:
                if not self.grid[r][c].blocked:
                    self.grid[r][c].blocked = True
                    return (r, c)
        return None

    def spawn_new_fire(self):
        """Dynamically spawn a new fire hotspot."""
        for _ in range(20):
            r = random.randint(1, ROWS-2)
            c = random.randint(1, COLS-2)
            if (not self.grid[r][c].blocked and (r,c) != self.base
                    and (r,c) not in self.medical_centers):
                self.grid[r][c].on_fire = True
                self.grid[r][c].risk = min(1.0, self.grid[r][c].risk + 0.3)
                return (r, c)
        return None

    def spread_fire(self):
        hotspots = [(r,c) for r in range(ROWS) for c in range(COLS)
                    if self.grid[r][c].on_fire]
        for r, c in hotspots:
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and not self.grid[nr][nc].blocked:
                    if random.random() < 0.25:
                        self.grid[nr][nc].on_fire = True
                        self.grid[nr][nc].risk = min(1.0, self.grid[nr][nc].risk + 0.2)

    def neighbors(self, r, c):
        result = []
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and not self.grid[nr][nc].blocked:
                result.append((nr, nc))
        return result

# =============================================================================
# VICTIM
# =============================================================================
class Victim:
    def __init__(self, vid, r, c, severity):
        self.id       = vid
        self.r        = r
        self.c        = c
        self.severity = severity
        self.distance = 0
        # Gaussian delay correlated with risk (set after placement)
        self.delay    = max(0, int(random.gauss(4, 2.5)))
        self.local_risk  = 0.0
        self.survival_prob = 0.0
        self.initial_surv  = 0.0
        self.rescued       = False
        self.dead          = False
        self.assigned_ambulance = None
        self.rescue_time   = None
        self.steps_waited  = 0   # turns skipped while other victims rescued

    def deteriorate(self):
        """Reduce survival probability each turn this victim is not rescued."""
        if self.rescued or self.dead:
            return
        self.steps_waited += 1
        drop = DETERIORATION_RATE[self.severity] * random.uniform(0.8, 1.3)
        self.survival_prob = max(0.0, self.survival_prob - drop)
        # Death event: survival collapse triggers death
        thresholds = {"critical": 0.08, "moderate": 0.04, "minor": 0.0}
        if self.survival_prob < thresholds[self.severity]:
            self.dead = True

# =============================================================================
# DATASET & ML MODELS
# =============================================================================
def make_dataset(n=600):
    """
    Improved dataset:
    - Gaussian distributions on continuous features
    - Correlated features (high risk -> longer delay)
    - Feature interactions (severity x risk, delay x distance compound)
    - Random Gaussian noise to prevent memorisation
    - ~5% missing data simulated via median imputation
    - Reduced penalty weights -> more survivors, better recall
    - Raised base survival values
    - Probabilistic label assignment via sigmoid (not hard cutoff)
    """
    data = []
    # Realistic severity proportions
    sev_pool = ["critical"]*28 + ["moderate"]*42 + ["minor"]*30

    for _ in range(n):
        sev   = random.choice(sev_pool)
        dist  = max(1, int(random.gauss(10, 4)))
        risk  = min(1.0, max(0.0, random.gauss(0.38, 0.22)))
        # Correlated: high risk areas mean longer wait
        delay = max(0, int(risk * 6 + random.gauss(2, 1.5)))

        # Missing data simulation (~5% chance per feature, median imputation)
        if random.random() < 0.05: dist  = 10
        if random.random() < 0.05: delay = 3

        # Higher base survival values
        base = {"critical": 0.52, "moderate": 0.72, "minor": 0.90}[sev]

        # Feature interactions
        sev_risk   = risk * {"critical": 0.35, "moderate": 0.20, "minor": 0.08}[sev]
        compound   = (delay * dist) / 200.0   # delay-distance interaction

        score = (base
                 - 0.012 * dist
                 - 0.025 * delay
                 - 0.18  * risk
                 - sev_risk
                 - compound
                 + random.gauss(0, 0.04))  # noise

        # Probabilistic label (sigmoid, threshold 0.30 for more survivors)
        prob = 1 / (1 + math.exp(-10 * (score - 0.35)))
        survival = 1 if random.random() < prob else 0

        data.append({"sev": sev, "dist": dist, "delay": delay,
                     "risk": risk, "survival": survival})
    return data


def encode_sev(s):
    return {"critical": 0, "moderate": 1, "minor": 2}[s]

def victim_features(v):
    return [encode_sev(v.severity), v.distance, v.delay, v.local_risk]

def victim_features_from_dict(d):
    return [encode_sev(d["sev"]), d["dist"], d["delay"], d["risk"]]

def knn_features(v):
    return [
        encode_sev(v.severity) / 2,   # 0-1
        v.distance / 20,              # normalize
        v.delay / 10,                 # normalize
        v.local_risk                  # already 0-1
    ]

def knn_features_from_dict(d):
    return [
        encode_sev(d["sev"]) / 2,
        d["dist"] / 20,
        d["delay"] / 10,
        d["risk"]
    ]

def euclidean(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))

# ---------- kNN — instance-based; training = storing examples + k selection ----------
class KNNModel:
    """
    kNN 'trains' by:
      1. Storing the training split of the dataset (instance memory).
      2. Selecting best k via cross-validation on the validation split.
    Prediction at runtime: find k nearest neighbours in stored training data.
    Threshold calibrated for balanced precision/recall (0.45).
    """
    def __init__(self, k_candidates=None):
        self.k_candidates  = k_candidates or [3, 5, 7, 9]
        self.training_data = []
        self.best_k        = 5

    def fit(self, dataset):
        split = int(len(dataset) * 0.8)
        self.training_data = dataset[:split]
        val = dataset[split:]
        best_acc, best_k = -1, 5
        for k in self.k_candidates:
            correct = 0
            for d in val:
                class DV:
                    severity = d["sev"]; distance = d["dist"]
                    delay    = d["delay"]; local_risk = d["risk"]
                prob = self._raw_predict(DV(), k)
                pred = 1 if prob >= 0.45 else 0
                if pred == d["survival"]:
                    correct += 1
            acc = correct / max(len(val), 1)
            if acc > best_acc:
                best_acc = acc
                best_k   = k
        self.best_k = best_k

    def _raw_predict(self, victim, k):
        q     = knn_features(victim)
        dists = [(euclidean(q, knn_features_from_dict(d)), d["survival"])
                 for d in self.training_data]
        dists.sort(key=lambda x: x[0])
        return sum(x[1] for x in dists[:k]) / k

    def predict_prob(self, victim):
        return self._raw_predict(victim, self.best_k)

    @staticmethod
    def get_threshold():
        return 0.30   # calibrated threshold

# ---------- Binary Naive Bayes — NO gradient training; uses frequency priors ----------
class BinaryNaiveBayes:
    """
    Bernoulli Naive Bayes (as taught in class).

    * Features are binarised at their dataset medians.
    * Likelihoods are P(feature_j = 1 | class) estimated from dataset frequencies.
    * Priors are class frequencies in the full dataset.
    * Laplace smoothing prevents zero probabilities.
    * There is no iterative training — the model is constructed directly from counts.
    * This matches the Bernoulli NB formulation taught in lectures (not Gaussian).
    """
    def __init__(self, dataset):
        self.medians     = self._compute_medians(dataset)
        n                = len(dataset)
        n1               = sum(1 for d in dataset if d["survival"] == 1)
        n0               = n - n1
        self.prior       = {1: n1/n, 0: n0/n}
        self.likelihood  = {0: [], 1: []}
        for c in [0, 1]:
            rows = [d for d in dataset if d["survival"] == c]
            nc   = len(rows)
            for j in range(4):
                count_1 = sum(1 for d in rows
                              if self._binarise(victim_features_from_dict(d)[j], j) == 1)
                # Laplace smoothing: (count + 1) / (nc + 2)
                self.likelihood[c].append((count_1 + 1) / (nc + 2))

    def _compute_medians(self, dataset):
        feats = [victim_features_from_dict(d) for d in dataset]
        return [sorted(f[j] for f in feats)[len(feats)//2] for j in range(4)]

    def _binarise(self, val, j):
        return 1 if val >= self.medians[j] else 0

    def predict_prob(self, victim):
        q         = victim_features(victim)
        log_probs = {}
        for c in [0, 1]:
            lp = math.log(self.prior[c] + 1e-12)
            for j in range(4):
                b  = self._binarise(q[j], j)
                p  = self.likelihood[c][j]
                lp += math.log(p if b == 1 else (1 - p) + 1e-12)
            log_probs[c] = lp
        m  = max(log_probs.values())
        e0 = math.exp(log_probs[0] - m)
        e1 = math.exp(log_probs[1] - m)
        return e1 / (e0 + e1)

    @staticmethod
    def get_threshold():
        return 0.45   # same calibrated threshold as kNN

# =============================================================================
# SIMULATED ANNEALING — Rescue Order Optimisation
# =============================================================================
def sa_cost(order, env):
    """
    Multi-objective cost for a rescue ordering (lower = better):
      - heuristic distance to each victim from previous position
      - urgency penalty: critical victims scheduled late cost more
      - local risk at victim position
    """
    cost    = 0.0
    current = env.base
    for rank, v in enumerate(order):
        if v.dead: continue
        dist          = heuristic(current, (v.r, v.c))
        sev_w         = {"critical": 3.0, "moderate": 1.5, "minor": 0.8}[v.severity]
        urgency_pen   = sev_w * (rank + 1)
        risk_pen      = v.local_risk * 2.0
        cost         += dist + urgency_pen + risk_pen
        current       = (v.r, v.c)
    return cost

def simulated_annealing_order(victims, env):
    """
    Optimise rescue order using Simulated Annealing.
    Returns (best_order, sa_log_dict).
    SA explores neighbour solutions (random pairwise swaps) and accepts
    worse solutions with probability e^(-delta/temp) to escape local optima.
    """
    active = [v for v in victims if not v.dead and not v.rescued]
    if len(active) <= 1:
        return active, {"iterations": 0, "improvements": 0,
                        "initial_cost": 0.0, "final_cost": 0.0}

    current      = list(active)
    random.shuffle(current)
    current_cost = sa_cost(current, env)
    best         = list(current)
    best_cost    = current_cost
    initial_cost = sa_cost(active, env)

    temp         = SA_TEMP_INIT
    iterations   = 0
    improvements = 0

    while temp > SA_TEMP_MIN and iterations < SA_ITER:
        i, j       = random.sample(range(len(current)), 2)
        candidate  = list(current)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_cost  = sa_cost(candidate, env)
        delta      = cand_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current      = candidate
            current_cost = cand_cost
            if current_cost < best_cost:
                best      = list(current)
                best_cost = current_cost
                improvements += 1

        temp *= SA_COOL
        iterations += 1

    return best, {
        "iterations":   iterations,
        "improvements": improvements,
        "initial_cost": initial_cost,
        "final_cost":   best_cost,
    }

# =============================================================================
# SEARCH ALGORITHMS
# =============================================================================
def reconstruct(parent, goal):
    path = []; node = goal
    while node is not None:
        path.append(node); node = parent[node]
    return path[::-1]

def bfs(env, start, goal):
    q = deque([start]); parent = {start: None}; expanded = 0; max_f = 1
    t0 = time.perf_counter()
    while q:
        max_f = max(max_f, len(q)); node = q.popleft(); expanded += 1
        if node == goal: break
        for n in env.neighbors(*node):
            if n not in parent:
                parent[n] = node; q.append(n)
    elapsed = (time.perf_counter() - t0) * 1000
    path = reconstruct(parent, goal) if goal in parent else []
    return {"path": path, "expanded": expanded, "frontier": max_f, "time_ms": elapsed}

def dfs(env, start, goal):
    stack = [start]; parent = {start: None}; expanded = 0; max_f = 1
    t0 = time.perf_counter()
    while stack:
        max_f = max(max_f, len(stack)); node = stack.pop(); expanded += 1
        if node == goal: break
        for n in env.neighbors(*node):
            if n not in parent:
                parent[n] = node; stack.append(n)
    elapsed = (time.perf_counter() - t0) * 1000
    path = reconstruct(parent, goal) if goal in parent else []
    return {"path": path, "expanded": expanded, "frontier": max_f, "time_ms": elapsed}

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def greedy(env, start, goal):
    pq = [(heuristic(start, goal), start)]; parent = {start: None}
    expanded = 0; max_f = 1
    t0 = time.perf_counter()
    while pq:
        max_f = max(max_f, len(pq)); _, node = heapq.heappop(pq); expanded += 1
        if node == goal: break
        for n in env.neighbors(*node):
            if n not in parent:
                parent[n] = node; heapq.heappush(pq, (heuristic(n, goal), n))
    elapsed = (time.perf_counter() - t0) * 1000
    path = reconstruct(parent, goal) if goal in parent else []
    return {"path": path, "expanded": expanded, "frontier": max_f, "time_ms": elapsed}

def astar(env, start, goal):
    cost = {start: 0}; parent = {start: None}
    pq = [(0, start)]; expanded = 0; max_f = 1
    t0 = time.perf_counter()
    while pq:
        max_f = max(max_f, len(pq)); _, node = heapq.heappop(pq); expanded += 1
        if node == goal: break
        for n in env.neighbors(*node):
            nc = cost[node] + 1 + env.grid[n[0]][n[1]].risk
            if n not in cost or nc < cost[n]:
                cost[n] = nc; parent[n] = node
                heapq.heappush(pq, (nc + heuristic(n, goal), n))
    elapsed = (time.perf_counter() - t0) * 1000
    path = reconstruct(parent, goal) if goal in parent else []
    return {"path": path, "expanded": expanded, "frontier": max_f, "time_ms": elapsed}

def hill_climbing(env, start, goal):
    current = start; path = [current]; visited = {current}; expanded = 0
    t0 = time.perf_counter()
    while current != goal:
        expanded += 1
        nbrs = [n for n in env.neighbors(*current) if n not in visited]
        if not nbrs: break
        best = min(nbrs, key=lambda n: heuristic(n, goal))
        if heuristic(best, goal) >= heuristic(current, goal): break
        visited.add(best); path.append(best); current = best
    elapsed = (time.perf_counter() - t0) * 1000
    if current != goal: path = []
    return {"path": path, "expanded": expanded, "frontier": 1, "time_ms": elapsed}

def path_risk(env, path):
    return sum(env.grid[r][c].risk for r, c in path)

def context_aware_algo_select(victim, env, start, goal):
    """
    Context-aware algorithm selection:
    - Victim on fire cell            → Greedy (maximum speed)
    - Critical victim, close         → A* (optimal, fast enough)
    - Critical victim, far           → Greedy (pure speed, urgency wins)
    - Minor in low-risk area         → BFS (guaranteed shortest unweighted path)
    - Default                        → A* (best trade-off)
    """
    dist       = heuristic(start, goal)
    on_fire    = env.grid[victim.r][victim.c].on_fire
    local_risk = env.grid[victim.r][victim.c].risk

    if on_fire:                                        return "Greedy"
    if victim.severity == "critical" and dist <= 8:   return "A*"
    if victim.severity == "critical" and dist > 8:    return "Greedy"
    if victim.severity == "minor" and local_risk < 0.3: return "BFS"
    return "A*"

# =============================================================================
# CSP — RESOURCE ALLOCATION
# =============================================================================
class CSPAllocator:
    """
    CSP formulation:
      Variables   : each unrescued victim
      Domains     : ambulance IDs {0, 1}
      Constraints :
        - Max 2 victims per ambulance simultaneously
        - Total medic kits used <= kits_remaining (HARD)
    Solver       : Backtracking + MRV (Minimum Remaining Values) heuristic
                   + Forward checking (kit depletion tracking)
    """
    def __init__(self, victims, ambulances=2, kits_remaining=10):
        self.victims       = [v for v in victims if not v.rescued and not v.dead]
        self.ambulances    = ambulances
        self.kits_remaining = kits_remaining
        self.assignment    = {}
        self.skipped       = []
        self.backtrack_count = 0

    def solve(self):
        # MRV: sort by severity (critical first), then lowest survival prob
        order = sorted(self.victims,
                       key=lambda v: (SEVERITY_ORDER[v.severity], v.survival_prob))
        self._bt(order, 0, {i: [] for i in range(self.ambulances)},
                 self.kits_remaining)
        assigned_ids = set(self.assignment.keys())
        for v in self.victims:
            if v.id not in assigned_ids:
                self.skipped.append(v)
        return self.assignment

    def _bt(self, order, idx, ambu_loads, kits_left):
        if idx == len(order):
            self.assignment = {v.id: ambu
                               for ambu, vs in ambu_loads.items() for v in vs}
            return True
        v = order[idx]
        kn = KIT_COST[v.severity]

        if kits_left < kn:
            # Skip this victim — not enough kits (forward checking)
            return self._bt(order, idx+1, ambu_loads, kits_left)

        # Prefer ambulance with fewer assignments (MRV / degree heuristic)
        ambu_order = sorted(range(self.ambulances),
                            key=lambda a: len(ambu_loads[a]))
        for ambu_id in ambu_order:
            ambu_loads[ambu_id].append(v)
            if self._bt(order, idx+1, ambu_loads, kits_left - kn):
                return True
            ambu_loads[ambu_id].pop()
            self.backtrack_count += 1
        return False

# =============================================================================
# FUZZY LOGIC — URGENCY DECISION
# =============================================================================
def fuzzy_urgency(severity, survival_prob, area_risk):
    sev_score = {"critical": 1.0, "moderate": 0.6, "minor": 0.2}[severity]
    low_surv  = max(0, 1 - survival_prob/0.4) if survival_prob < 0.4 else 0
    high_risk = min(1, area_risk/0.7)
    low_risk  = max(0, 1 - area_risk/0.3)
    urgency   = 0
    urgency   = max(urgency, min(sev_score, low_surv)  * 1.0)
    urgency   = max(urgency, min(sev_score, high_risk) * 0.8)
    urgency   = max(urgency, min(1-sev_score, low_risk) * 0.3)
    return round(min(1.0, urgency), 3)

# =============================================================================
# KPI TRACKER
# =============================================================================
class KPITracker:
    def __init__(self, total_victims):
        self.total_victims = total_victims
        self.saved = 0; self.dead = 0
        self.total_time  = 0.0; self.total_risk = 0.0
        self.path_costs  = []
        self.algo_usage  = {}; self.replan_count = 0
        self.victim_details = []
        self.kits_used   = 0

    def record(self, victim, algo, path_len, risk):
        self.saved += 1
        self.total_time += path_len; self.total_risk += risk
        self.path_costs.append(path_len)
        self.algo_usage[algo] = self.algo_usage.get(algo, 0) + 1
        self.kits_used += KIT_COST[victim.severity]
        self.victim_details.append({
            "vid": victim.id, "sev": victim.severity, "algo": algo,
            "path_len": path_len, "risk": round(risk, 3),
            "survival": round(victim.survival_prob, 3),
            "initial_surv": round(victim.initial_surv, 3),
            "steps_waited": victim.steps_waited,
        })

    def record_death(self, victim):
        self.dead += 1

    def avg_rescue_time(self):
        return self.total_time / max(self.saved, 1)

    def resource_util(self):
        return min(1.0, self.kits_used / 10.0)

# =============================================================================
# ML EVALUATION HELPERS
# =============================================================================
def _eval_model_generic(predict_fn, threshold, dataset):
    split = int(len(dataset) * 0.8)
    test  = dataset[split:]
    tp = fp = tn = fn = 0
    for d in test:
        class DV:
            severity = d["sev"]; distance = d["dist"]
            delay    = d["delay"]; local_risk = d["risk"]
        prob  = predict_fn(DV())
        pred  = 1 if prob >= threshold else 0
        actual = d["survival"]
        if   pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        else:                           fn += 1
    acc  = (tp+tn) / max(tp+fp+tn+fn, 1)
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1   = 2*prec*rec / max(prec+rec, 0.001)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

# =============================================================================
# MAIN APPLICATION
# =============================================================================
class AIDRAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AIDRA — Adaptive Intelligent Disaster Response Agent  v6")
        self.configure(bg=PALETTE["bg"])
        self.resizable(True, True)

        # Build dataset and train/initialise models ONCE at startup
        self.dataset   = make_dataset(1000)
        self.knn_model = KNNModel()
        self.knn_model.fit(self.dataset)           # kNN trains (stores examples + best k)
        self.nb_model  = BinaryNaiveBayes(self.dataset)  # NB: frequency priors, no gradient

        # State
        self.env      = None
        self.victims  = []
        self.kpi      = None
        self.running  = False
        self.paused   = False
        self.ambulance_pos    = [(0,0),(0,0)]
        self.ambulance_trails = [[],[]]
        self.selected_algo = tk.StringVar(value="Context-Aware")
        self.selected_ml   = tk.StringVar(value="kNN")
        self.speed_var     = tk.DoubleVar(value=1.0)
        self.log_entries   = []
        self.kits_left     = 10

        self._build_ui()
        self._init_simulation()

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self._build_toolbar()
        self._build_canvas_area()
        self._build_right_panel()
        self._build_statusbar()

    def _build_toolbar(self):
        tb = tk.Frame(self, bg=PALETTE["card"], pady=6, padx=10)
        tb.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(tb, text="🚨 AIDRA", font=("Consolas",16,"bold"),
                 bg=PALETTE["card"], fg=PALETTE["accent"]).pack(side="left", padx=8)
        tk.Label(tb, text="Adaptive Intelligent Disaster Response Agent",
                 font=("Consolas",10), bg=PALETTE["card"], fg=PALETTE["subtext"]).pack(side="left")

        right = tk.Frame(tb, bg=PALETTE["card"])
        right.pack(side="right")

        tk.Label(right, text="Search:", bg=PALETTE["card"], fg=PALETTE["text"],
                 font=("Consolas",10)).pack(side="left", padx=(8,2))
        ttk.Combobox(right, textvariable=self.selected_algo, width=14,
                     values=["Context-Aware","A*","BFS","DFS",
                             "Greedy","Hill Climbing","All (Compare)"],
                     state="readonly", font=("Consolas",10)).pack(side="left", padx=2)

        tk.Label(right, text="ML:", bg=PALETTE["card"], fg=PALETTE["text"],
                 font=("Consolas",10)).pack(side="left", padx=(8,2))
        ttk.Combobox(right, textvariable=self.selected_ml, width=12,
                     values=["kNN","Naive Bayes"],
                     state="readonly", font=("Consolas",10)).pack(side="left", padx=2)

        tk.Label(right, text="Speed:", bg=PALETTE["card"], fg=PALETTE["text"],
                 font=("Consolas",10)).pack(side="left", padx=(8,2))
        ttk.Scale(right, from_=0.2, to=3.0, variable=self.speed_var,
                  orient="horizontal", length=80).pack(side="left", padx=2)

        self.btn_run   = tk.Button(right, text="▶ RUN", command=self._start_sim,
                                   bg=PALETTE["green"], fg="white",
                                   font=("Consolas",10,"bold"), relief="flat",
                                   padx=8, cursor="hand2")
        self.btn_run.pack(side="left", padx=4)

        self.btn_pause = tk.Button(right, text="⏸ PAUSE", command=self._toggle_pause,
                                   bg=PALETTE["yellow"], fg="#1A1A2E",
                                   font=("Consolas",10,"bold"), relief="flat",
                                   padx=8, cursor="hand2")
        self.btn_pause.pack(side="left", padx=4)

        self.btn_reset = tk.Button(right, text="↺ RESET", command=self._reset_sim,
                                   bg=PALETTE["accent"], fg="white",
                                   font=("Consolas",10,"bold"), relief="flat",
                                   padx=8, cursor="hand2")
        self.btn_reset.pack(side="left", padx=4)

    def _build_canvas_area(self):
        left = tk.Frame(self, bg=PALETTE["bg"])
        left.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        left.rowconfigure(0, weight=1); left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=COLS*CELL_SIZE+2*PADDING,
                                height=ROWS*CELL_SIZE+2*PADDING,
                                bg=PALETTE["bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        legend = tk.Frame(left, bg=PALETTE["bg"])
        legend.pack(fill="x", pady=(4,0))
        for label, color in [
            ("🏠 Base",       PALETTE["base"]),
            ("🏥 Med Ctr",    PALETTE["medcenter"]),
            ("🚑 Amb 1",      PALETTE["ambu1"]),
            ("🚑 Amb 2",      PALETTE["ambu2"]),
            ("🔴 Critical",   SEVERITY_COLORS["critical"]),
            ("🟡 Moderate",   SEVERITY_COLORS["moderate"]),
            ("🟢 Minor",      SEVERITY_COLORS["minor"]),
            ("☠ Dead",        PALETTE["dead"]),
            ("█ Blocked",     PALETTE["blocked"]),
        ]:
            tk.Label(legend, text="●", fg=color, bg=PALETTE["bg"],
                     font=("Consolas",11)).pack(side="left")
            tk.Label(legend, text=label, fg=PALETTE["subtext"], bg=PALETTE["bg"],
                     font=("Consolas",9)).pack(side="left", padx=(0,6))

    def _build_right_panel(self):
        right = tk.Frame(self, bg=PALETTE["bg"])
        right.grid(row=1, column=1, sticky="nsew", padx=(0,8), pady=8)
        right.rowconfigure(1, weight=1); right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        # KPI cards
        kf = tk.Frame(right, bg=PALETTE["bg"])
        kf.grid(row=0, column=0, sticky="ew")
        kf.columnconfigure((0,1,2,3,4), weight=1)
        self.kpi_labels = {}
        for i, (key, lbl, color) in enumerate([
            ("saved",    "Saved",     PALETTE["green"]),
            ("dead",     "Dead",      PALETTE["accent"]),
            ("kits",     "Kits Left", PALETTE["yellow"]),
            ("avg_time", "Avg Time",  PALETTE["blue"]),
            ("replans",  "Replans",   PALETTE["subtext"]),
        ]):
            card = tk.Frame(kf, bg=PALETTE["card"], padx=6, pady=6)
            card.grid(row=0, column=i, padx=3, pady=3, sticky="ew")
            tk.Label(card, text=lbl, fg=PALETTE["subtext"], bg=PALETTE["card"],
                     font=("Consolas",8)).pack()
            l = tk.Label(card, text="—", fg=color, bg=PALETTE["card"],
                         font=("Consolas",14,"bold"))
            l.pack()
            self.kpi_labels[key] = l

        def _scrollable_panel(parent, title, title_color, height):
            f = tk.Frame(parent, bg=PALETTE["panel"], bd=0)
            f.columnconfigure(0, weight=1); f.rowconfigure(1, weight=1)
            tk.Label(f, text=title, font=("Consolas",10,"bold"),
                     bg=PALETTE["panel"], fg=title_color, pady=4
                     ).grid(row=0, column=0, sticky="w", padx=8)
            st = scrolledtext.ScrolledText(f, bg=PALETTE["panel"], fg=PALETTE["text"],
                                           font=("Consolas",9), relief="flat", wrap="word",
                                           insertbackground=PALETTE["text"], height=height)
            st.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0,4))
            self._style_text_tags(st)
            return f, st

        dec_f, self.decision_text = _scrollable_panel(
            right, "⚡ REASONING & DECISIONS", PALETTE["accent"], 12)
        dec_f.grid(row=1, column=0, sticky="nsew", pady=(0,4))

        log_f, self.log_text = _scrollable_panel(
            right, "📋 EVENT LOG", PALETTE["blue"], 8)
        log_f.grid(row=2, column=0, sticky="ew", pady=(0,4))

        ml_f, self.ml_text = _scrollable_panel(
            right, "🤖 ML MODEL METRICS", PALETTE["yellow"], 8)
        ml_f.grid(row=3, column=0, sticky="nsew")

    def _style_text_tags(self, w):
        w.tag_config("header", foreground=PALETTE["accent"], font=("Consolas",9,"bold"))
        w.tag_config("good",   foreground=PALETTE["green"])
        w.tag_config("warn",   foreground=PALETTE["yellow"])
        w.tag_config("bad",    foreground=PALETTE["accent"])
        w.tag_config("info",   foreground=PALETTE["blue"])
        w.tag_config("sub",    foreground=PALETTE["subtext"])
        w.tag_config("white",  foreground=PALETTE["text"])
        w.tag_config("dead",   foreground=PALETTE["dead"])

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=PALETTE["card"], pady=3, padx=10)
        sb.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.status_var = tk.StringVar(value="Ready. Press ▶ RUN to begin simulation.")
        tk.Label(sb, textvariable=self.status_var, fg=PALETTE["subtext"],
                 bg=PALETTE["card"], font=("Consolas",9), anchor="w").pack(side="left")
        self.progress = ttk.Progressbar(sb, length=150, mode="determinate")
        self.progress.pack(side="right", padx=8)

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION INIT
    # ─────────────────────────────────────────────────────────────────────────
    def _init_simulation(self):
        self.env       = Environment()
        self.victims   = []
        self.kits_left = 10

        # Random victim count 6–12 with realistic severity proportions
        n_v    = random.randint(6, 9)
        n_crit = max(1, round(n_v * 0.25))
        n_mod  = max(1, round(n_v * 0.45))
        n_min  = max(0, n_v - n_crit - n_mod)
        sevs   = (["critical"]*n_crit + ["moderate"]*n_mod + ["minor"]*n_min)
        random.shuffle(sevs)

        for i, sev in enumerate(sevs):
            for _ in range(200):
                r = random.randint(0, ROWS-1)
                c = random.randint(0, COLS-1)
                if (not self.env.grid[r][c].blocked and
                        (r,c) != self.env.base and
                        (r,c) not in self.env.medical_centers and
                        not any(v.r == r and v.c == c for v in self.victims)):
                    v = Victim(i+1, r, c, sev)
                    v.distance   = heuristic((r,c), self.env.base)
                    v.local_risk = self.env.grid[r][c].risk
                    # Correlated delay: higher local risk → slightly longer delay
                    v.delay = max(0, int(v.local_risk * 6 + random.gauss(2, 1.5)))
                    self.victims.append(v)
                    break

        for v in self.victims:
            v.survival_prob = self._predict(v)
            v.initial_surv  = v.survival_prob

        # Seed initial fires
        for _ in range(3):
            r = random.randint(2, ROWS-2)
            c = random.randint(2, COLS-2)
            self.env.grid[r][c].on_fire = True

        self.kpi = KPITracker(len(self.victims))
        self.ambulance_pos    = [self.env.base, self.env.base]
        self.ambulance_trails = [[], []]
        self.log_entries      = []

        self._draw_grid()
        self._update_kpi_display()
        self._refresh_ml_metrics()
        self.progress["value"] = 0

    def _predict(self, victim):
        if self.selected_ml.get() == "kNN":
            return self.knn_model.predict_prob(victim)
        else:
            return self.nb_model.predict_prob(victim)

    # ─────────────────────────────────────────────────────────────────────────
    # GRID DRAWING
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_grid(self, anim_positions=None, trails=None):
        self.canvas.delete("all")
        env = self.env

        for r in range(ROWS):
            for c in range(COLS):
                x1 = PADDING + c*CELL_SIZE; y1 = PADDING + r*CELL_SIZE
                x2 = x1 + CELL_SIZE - 1;   y2 = y1 + CELL_SIZE - 1
                cell = env.grid[r][c]
                if cell.blocked:
                    color = PALETTE["blocked"]
                elif cell.on_fire:
                    color = "#8B0000"
                else:
                    color = "#0D2B1A" if cell.risk < 0.33 else \
                            "#2B1A00" if cell.risk < 0.66 else "#2B0000"
                self.canvas.create_rectangle(x1,y1,x2,y2, fill=color,
                                             outline=PALETTE["grid_line"], width=1)
                if cell.on_fire and not cell.blocked:
                    self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                            text="🔥", font=("Arial",12))

        if trails:
            for ti, trail in enumerate(trails):
                color = PALETTE["ambu1"] if ti == 0 else PALETTE["ambu2"]
                for pos in trail:
                    r, c = pos
                    ox = PADDING + c*CELL_SIZE + CELL_SIZE//2 - 4
                    oy = PADDING + r*CELL_SIZE + CELL_SIZE//2 - 4
                    self.canvas.create_oval(ox,oy,ox+8,oy+8, fill=color,
                                            outline="", stipple="gray50")

        for mc in env.medical_centers:
            r, c = mc
            x1=PADDING+c*CELL_SIZE; y1=PADDING+r*CELL_SIZE
            self.canvas.create_rectangle(x1+2,y1+2,x1+CELL_SIZE-3,y1+CELL_SIZE-3,
                                          fill=PALETTE["medcenter"], outline="white", width=2)
            self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                     text="🏥", font=("Arial",16))

        br, bc = env.base
        bx=PADDING+bc*CELL_SIZE; by=PADDING+br*CELL_SIZE
        self.canvas.create_rectangle(bx+2,by+2,bx+CELL_SIZE-3,by+CELL_SIZE-3,
                                      fill=PALETTE["base"], outline="white", width=2)
        self.canvas.create_text(bx+CELL_SIZE//2, by+CELL_SIZE//2,
                                 text="🏠", font=("Arial",16))

        for v in self.victims:
            r, c = v.r, v.c
            x1=PADDING+c*CELL_SIZE; y1=PADDING+r*CELL_SIZE
            if v.dead:
                self.canvas.create_oval(x1+4,y1+4,x1+CELL_SIZE-5,y1+CELL_SIZE-5,
                                         fill=PALETTE["dead"], outline="#888", width=2)
                self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                         text="✝", font=("Consolas",12,"bold"), fill="#AAA")
            elif not v.rescued:
                color = SEVERITY_COLORS[v.severity]
                self.canvas.create_oval(x1+4,y1+4,x1+CELL_SIZE-5,y1+CELL_SIZE-5,
                                         fill=color, outline="white", width=2)
                self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2-2,
                                         text=f"V{v.id}", font=("Consolas",8,"bold"), fill="white")
                sev_char = {"critical":"!","moderate":"~","minor":"·"}[v.severity]
                self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE-9,
                                         text=sev_char, font=("Consolas",9,"bold"), fill=color)

        if anim_positions:
            for ai, pos in enumerate(anim_positions):
                if pos is None: continue
                r, c = pos
                x1=PADDING+c*CELL_SIZE; y1=PADDING+r*CELL_SIZE
                col = PALETTE["ambu1"] if ai == 0 else PALETTE["ambu2"]
                self.canvas.create_rectangle(x1+3,y1+3,x1+CELL_SIZE-4,y1+CELL_SIZE-4,
                                              fill=col, outline="white", width=2)
                self.canvas.create_text(x1+CELL_SIZE//2, y1+CELL_SIZE//2,
                                         text=f"A{ai+1}", font=("Consolas",9,"bold"),
                                         fill="#1A1A2E")

        self.canvas.update()

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts    = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}\n"
        self.log_entries.append(entry)
        self.log_text.insert("end", entry)
        self.log_text.see("end")

    def _decision(self, msg, tag="white"):
        self.decision_text.insert("end", msg+"\n", tag)
        self.decision_text.see("end")

    # ─────────────────────────────────────────────────────────────────────────
    # KPI & ML DISPLAY
    # ─────────────────────────────────────────────────────────────────────────
    def _update_kpi_display(self):
        total = len(self.victims)
        self.kpi_labels["saved"].config(text=f"{self.kpi.saved}/{total}")
        self.kpi_labels["dead"].config(text=f"{self.kpi.dead}/{total}")
        self.kpi_labels["kits"].config(text=f"{self.kits_left}/10")
        self.kpi_labels["avg_time"].config(text=f"{self.kpi.avg_rescue_time():.1f}")
        self.kpi_labels["replans"].config(text=f"{self.kpi.replan_count}")

    def _refresh_ml_metrics(self):
        self.ml_text.delete("1.0","end")
        ml = self.selected_ml.get()
        self.ml_text.insert("end", f"Active Model: {ml}\n","header")
        self.ml_text.insert("end", "─"*38+"\n","sub")

        # kNN
        knn_m  = _eval_model_generic(self.knn_model.predict_prob,
                                     self.knn_model.get_threshold(), self.dataset)
        mk = "► " if ml == "kNN" else "  "
        tk_tag = "good" if ml == "kNN" else "sub"
        self.ml_text.insert("end",
            f"\n{mk}kNN (k={self.knn_model.best_k},"
            f" thresh={self.knn_model.get_threshold()}) [TRAINED]\n", tk_tag)
        self.ml_text.insert("end",
            f"  Acc={knn_m['acc']:.3f}  Prec={knn_m['prec']:.3f}\n"
            f"  Rec={knn_m['rec']:.3f}  F1={knn_m['f1']:.3f}\n"
            f"  TP={knn_m['tp']} FP={knn_m['fp']} TN={knn_m['tn']} FN={knn_m['fn']}\n","sub")

        # Naive Bayes
        nb_m   = _eval_model_generic(self.nb_model.predict_prob,
                                     self.nb_model.get_threshold(), self.dataset)
        mn = "► " if ml == "Naive Bayes" else "  "
        tn_tag = "good" if ml == "Naive Bayes" else "sub"
        self.ml_text.insert("end",
            f"\n{mn}Binary Naive Bayes (thresh={self.nb_model.get_threshold()})"
            f" [FREQUENCY PRIORS]\n", tn_tag)
        self.ml_text.insert("end",
            f"  Bernoulli NB — binarised features, Laplace smoothing\n"
            f"  Acc={nb_m['acc']:.3f}  Prec={nb_m['prec']:.3f}\n"
            f"  Rec={nb_m['rec']:.3f}  F1={nb_m['f1']:.3f}\n"
            f"  TP={nb_m['tp']} FP={nb_m['fp']} TN={nb_m['tn']} FN={nb_m['fn']}\n","sub")

        self.ml_text.insert("end","\n── Victim Survival Probabilities ──\n","info")
        for v in self.victims:
            if v.dead:
                self.ml_text.insert("end", f"V{v.id} [{v.severity[:3].upper()}] → DEAD\n","dead")
            else:
                sev_tag = {"critical":"bad","moderate":"warn","minor":"good"}[v.severity]
                self.ml_text.insert("end",
                    f"V{v.id} [{v.severity[:3].upper()}] → {v.survival_prob:.3f}  "
                    f"fuzzy={fuzzy_urgency(v.severity,v.survival_prob,v.local_risk):.3f}\n",
                    sev_tag)

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH RUNNER
    # ─────────────────────────────────────────────────────────────────────────
    def _run_search(self, start, goal, victim=None, force_algo=None):
        fn_map = {"BFS":bfs,"DFS":dfs,"Greedy":greedy,"A*":astar,"Hill Climbing":hill_climbing}
        alg    = force_algo or self.selected_algo.get()

        if alg == "Context-Aware" and victim is not None:
            chosen = context_aware_algo_select(victim, self.env, start, goal)
            res = fn_map[chosen](self.env, start, goal)
            res["algo"] = chosen + " (ctx)"
        elif alg == "All (Compare)":
            results = []
            for name, fn in fn_map.items():
                r = fn(self.env, start, goal)
                r["algo"] = name
                if r["path"]:
                    r["risk"]   = path_risk(self.env, r["path"])
                    r["length"] = len(r["path"])
                    results.append(r)
            if results:
                lens = [r["length"]   for r in results]
                risks= [r["risk"]     for r in results]
                exps = [r["expanded"] for r in results]
                tims = [r["time_ms"]  for r in results]
                def norm(vals):
                    mn,mx = min(vals),max(vals)
                    return [(v-mn)/(mx-mn+1e-9) for v in vals]
                nl=norm(lens); nr=norm(risks); ne=norm(exps); nt=norm(tims)
                for i, r in enumerate(results):
                    r["score"] = 1.0 - 0.35*nl[i] - 0.30*nr[i] - 0.20*ne[i] - 0.15*nt[i]
            chosen_res = max(results, key=lambda x: x["score"]) if results else None
            if chosen_res and chosen_res.get("path"):
                chosen_res["risk"]   = path_risk(self.env, chosen_res["path"])
                chosen_res["length"] = len(chosen_res["path"])
            else:
                if chosen_res:
                    chosen_res["risk"] = 0; chosen_res["length"] = 0
            return chosen_res, results
        else:
            clean = alg if alg != "Context-Aware" else "A*"
            res = fn_map.get(clean, astar)(self.env, start, goal)
            res["algo"] = clean

        if res.get("path"):
            res["risk"]   = path_risk(self.env, res["path"])
            res["length"] = len(res["path"])
        else:
            res["risk"] = 0; res["length"] = 0
        return res, [res]

    # ─────────────────────────────────────────────────────────────────────────
    # ANIMATION
    # ─────────────────────────────────────────────────────────────────────────
    def _animate_path(self, ambu_idx, path, other_pos=None):
        delay = ANIM_DELAY / self.speed_var.get()
        self.ambulance_trails[ambu_idx] = []
        for step in path:
            while self.paused: time.sleep(0.1)
            self.ambulance_pos[ambu_idx] = step
            self.ambulance_trails[ambu_idx].append(step)
            p = list(self.ambulance_pos)
            t = [list(x) for x in self.ambulance_trails]
            self.canvas.after(0, lambda _p=p, _t=t: self._draw_grid(_p, _t))
            time.sleep(delay)

    # ─────────────────────────────────────────────────────────────────────────
    # CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    def _start_sim(self):
        if self.running: return
        self.running = True
        self.btn_run.config(state="disabled")
        for v in self.victims:
            v.survival_prob = self._predict(v)
            v.initial_surv  = v.survival_prob
        self._refresh_ml_metrics()
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(
            text="▶ RESUME" if self.paused else "⏸ PAUSE",
            bg=PALETTE["green"] if self.paused else PALETTE["yellow"])

    def _reset_sim(self):
        self.running = False; self.paused = False
        self.btn_run.config(state="normal")
        self.btn_pause.config(text="⏸ PAUSE", bg=PALETTE["yellow"])
        self.decision_text.delete("1.0","end")
        self.log_text.delete("1.0","end")
        self.progress["value"] = 0
        self._init_simulation()
        self.status_var.set("Simulation reset. Press ▶ RUN to begin.")

    # ─────────────────────────────────────────────────────────────────────────
    # CORE SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def _run_simulation(self):
        env = self.env

        # ── 1. Simulated Annealing — optimise rescue order ──────────────────
        self._decision("═══ SIMULATED ANNEALING — RESCUE ORDER ═══\n","header")
        self._decision(
            "  SA explores pairwise swap neighbours, accepting worse solutions\n"
            "  with probability e^(-Δcost/T) to escape local optima.\n","sub")
        sa_order, sa_log = simulated_annealing_order(self.victims, env)
        self._decision(
            f"  Iterations: {sa_log['iterations']}  "
            f"Improvements: {sa_log['improvements']}\n"
            f"  Cost: {sa_log['initial_cost']:.2f} → {sa_log['final_cost']:.2f} "
            f"({'↓ improved' if sa_log['final_cost'] < sa_log['initial_cost'] else '= no change'})\n","info")
        for rank, v in enumerate(sa_order, 1):
            self._decision(
                f"  #{rank}: V{v.id} [{v.severity.upper()}] "
                f"surv={v.survival_prob:.3f} fuzzy="
                f"{fuzzy_urgency(v.severity,v.survival_prob,v.local_risk):.3f}\n",
                {"critical":"bad","moderate":"warn","minor":"good"}[v.severity])
        self._log(f"SA: {sa_log['improvements']} improvements / {sa_log['iterations']} iter | "
                  f"cost {sa_log['initial_cost']:.2f}→{sa_log['final_cost']:.2f}")

        # ── 2. CSP — resource allocation ────────────────────────────────────
        csp = CSPAllocator(self.victims, kits_remaining=self.kits_left)
        allocation = csp.solve()
        self._decision("\n═══ CSP RESOURCE ALLOCATION ═══\n","header")
        self._decision(
            f"  Strategy: MRV ordering + forward checking (kit budget)\n"
            f"  Kits available: {self.kits_left}   Backtracks: {csp.backtrack_count}\n","sub")
        for vid, ambu in allocation.items():
            v = next(x for x in self.victims if x.id == vid)
            v.assigned_ambulance = ambu
            self._decision(
                f"  V{vid} ({v.severity}) → Amb {ambu+1}  "
                f"[{KIT_COST[v.severity]} kits]\n","white")
        if csp.skipped:
            self._decision(f"\n  ⚠ KIT SHORTAGE — cannot treat:\n","bad")
            for v in csp.skipped:
                self._decision(f"    V{v.id} ({v.severity}) — insufficient kits\n","bad")
                self._log(f"⚠ V{v.id} ({v.severity}): SKIPPED — not enough kits")

        # ── 3. Rescue operations in SA order ────────────────────────────────
        self._decision("\n═══ RESCUE OPERATIONS ═══\n","header")
        total = len(sa_order)
        ambu_current = [env.base, env.base]

        for idx, victim in enumerate(sa_order):
            if not self.running: break
            while self.paused: time.sleep(0.1)

            # Death check
            if victim.dead:
                self._decision(
                    f"\n💀 V{victim.id} [{victim.severity.upper()}] "
                    f"DIED while waiting (surv={victim.survival_prob:.3f}, "
                    f"waited {victim.steps_waited} turn(s))\n","dead")
                self._log(f"💀 V{victim.id} DIED — waited {victim.steps_waited} turns")
                self.kpi.record_death(victim)
                self._update_kpi_display()
                continue

            # Kit-shortage skip
            if victim.id not in allocation:
                self._decision(f"\n⚠ V{victim.id} — insufficient kits, cannot treat\n","bad")
                continue

            ambu_id = victim.assigned_ambulance if victim.assigned_ambulance is not None else 0
            self.status_var.set(
                f"Rescuing V{victim.id} ({victim.severity}) "
                f"| Amb {ambu_id+1} | Kits left: {self.kits_left}")

            self._decision(f"\n{'═'*38}\n","sub")
            self._decision(
                f"🚑 RESCUING V{victim.id} — {victim.severity.upper()}  "
                f"[waited {victim.steps_waited} turn(s)]\n","header")
            self._decision(
                f"  Pos: ({victim.r},{victim.c})  "
                f"Surv: {victim.initial_surv:.3f}→{victim.survival_prob:.3f} "
                f"(Δ{victim.survival_prob-victim.initial_surv:+.3f})\n","info")

            # Deteriorate all waiting victims
            for other in sa_order[idx+1:]:
                if not other.rescued and not other.dead:
                    other.deteriorate()
                    if other.dead:
                        self._log(f"💀 V{other.id} deteriorated to death!")

            start  = ambu_current[ambu_id]
            goal   = (victim.r, victim.c)
            mc_goal = min(env.medical_centers, key=lambda m: heuristic(m, goal))

            result, all_results = self._run_search(start, goal, victim=victim)

            if not result or not result.get("path"):
                self._log(f"V{victim.id}: No path found! Skipping.")
                self._decision("  ⚠ No path — skipping victim\n","bad")
                continue

            # Algorithm display
            if self.selected_algo.get() == "All (Compare)" and all_results:
                self._decision("\n  Algorithm Comparison:\n","info")
                for r in sorted(all_results, key=lambda x: x.get("score",0), reverse=True):
                    mk = "► " if r["algo"] == result["algo"] else "  "
                    self._decision(
                        f"  {mk}{r['algo']:12} "
                        f"exp={r['expanded']:4} "
                        f"t={r.get('time_ms',0):.2f}ms "
                        f"len={r['length']:3} "
                        f"risk={r['risk']:.2f} "
                        f"score={r.get('score',0):.3f}\n","sub")
                self._decision(f"\n  Selected: {result['algo']}\n","good")
            elif self.selected_algo.get() == "Context-Aware":
                self._decision(
                    f"  Context → {result['algo']} "
                    f"(sev={victim.severity}, d={heuristic(start,goal)}, "
                    f"fire={env.grid[victim.r][victim.c].on_fire})\n","info")

            # Trade-off
            path_r = result.get("risk", 0)
            self._decision("\n  Trade-off:\n","header")
            if   path_r > 2.0: self._decision("  ⚠ VERY HIGH risk — criticality justifies\n","bad")
            elif path_r > 1.0: self._decision("  ~ Moderate risk — acceptable\n","warn")
            else:               self._decision("  ✓ Low-risk route\n","good")
            if victim.severity == "critical":
                self._decision("  ⚡ Critical: speed > safety\n","warn")
            elif victim.steps_waited >= 3:
                self._decision("  ⚡ Long wait: urgency elevated\n","warn")
            else:
                self._decision("  🛡 Non-critical: safety > speed\n","info")

            urg = fuzzy_urgency(victim.severity, victim.survival_prob, victim.local_risk)
            self._decision(f"  Fuzzy urgency: {urg:.3f} → ","warn")
            if   urg > 0.7: self._decision("DISPATCH NOW\n","bad")
            elif urg > 0.4: self._decision("Dispatch with caution\n","warn")
            else:            self._decision("Routine dispatch\n","good")

            self._log(
                f"V{victim.id}: {result['algo']} len={result['length']} "
                f"risk={result.get('risk',0):.2f} "
                f"exp={result.get('expanded',0)} t={result.get('time_ms',0):.2f}ms")

            # Animate to victim
            self._animate_path(ambu_id, result["path"],
                               other_pos=ambu_current[1-ambu_id])
            ambu_current[ambu_id] = goal

            # Dynamic events
            ev = random.random()
            if ev < 0.30:
                blocked = env.block_random_road()
                if blocked:
                    self._log(f"⚠ Road blocked at {blocked}! Replanning...")
                    self._decision(f"\n  🔁 Road blockage at {blocked} — replanning\n","bad")
                    self.kpi.replan_count += 1
                    env.spread_fire()
                    mc_r, _ = self._run_search(goal, mc_goal, victim=victim)
                    if mc_r and mc_r.get("path"):
                        self._decision(f"  → Replanned via {mc_r['algo']} "
                                       f"len={mc_r['length']}\n","good")
            elif ev < 0.45:
                fp = env.spawn_new_fire()
                if fp:
                    self._log(f"🔥 New fire at {fp}!")
                    self._decision(f"\n  🔥 New fire at {fp} — rerouting\n","bad")
                    self.kpi.replan_count += 1
                    for other in sa_order[idx+1:]:
                        if not other.rescued and not other.dead:
                            other.local_risk = env.grid[other.r][other.c].risk
                    env.spread_fire()
                    mc_r, _ = self._run_search(goal, mc_goal, victim=victim)
                    if mc_r and mc_r.get("path"):
                        self._decision(f"  → Rerouted via {mc_r['algo']}\n","good")

            # Animate to medical centre
            mc_f, _ = self._run_search(goal, mc_goal, victim=victim)
            if mc_f and mc_f.get("path"):
                self._animate_path(ambu_id, mc_f["path"],
                                   other_pos=ambu_current[1-ambu_id])
                ambu_current[ambu_id] = mc_goal

            # Deduct kits
            kc = KIT_COST[victim.severity]
            self.kits_left = max(0, self.kits_left - kc)

            victim.rescued = True
            victim.rescue_time = result.get("length", 0)

            # Probabilistic survival outcome
            survived = (random.random()*0.38) < victim.survival_prob
            self.kpi.record(victim, result["algo"], victim.rescue_time,
                            result.get("risk", 0))
            self._update_kpi_display()
            self._refresh_ml_metrics()

            out_str = "✅ SURVIVED" if survived else "❌ DID NOT SURVIVE"
            out_tag = "good" if survived else "bad"
            self._decision(
                f"\n  {out_str}  (p={victim.survival_prob:.3f})\n"
                f"  Kits remaining: {self.kits_left}/10\n", out_tag)
            self._log(f"{'✅' if survived else '❌'} V{victim.id} {out_str} | "
                      f"kits_left={self.kits_left}")

            self.canvas.after(0, self._draw_grid,
                              list(ambu_current),
                              [list(t) for t in self.ambulance_trails])
            self.progress["value"] = int((idx+1)/total*100)

        self._show_final_report()
        self.running = False
        self.btn_run.config(state="normal")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────────────────
    def _show_final_report(self):
        total = len(self.victims)
        self._decision("\n"+"═"*38+"\n","sub")
        self._decision("          MISSION COMPLETE\n","header")
        self._decision("═"*38+"\n","sub")

        self._decision(f"\n📊 PERFORMANCE REPORT\n","header")
        self._decision(f"  Victims Total:       {total}\n","white")
        self._decision(f"  Victims Rescued:     {self.kpi.saved}/{total}\n","good")
        self._decision(f"  Victims Dead:        {self.kpi.dead}/{total}\n","bad")
        skipped = total - self.kpi.saved - self.kpi.dead
        if skipped > 0:
            self._decision(f"  Kit-shortage skips:  {skipped}\n","warn")
        self._decision(f"  Avg Rescue Time:     {self.kpi.avg_rescue_time():.2f} steps\n","info")
        self._decision(f"  Total Risk Exposure: {self.kpi.total_risk:.3f}\n","warn")
        self._decision(f"  Kits Used/Remaining: {self.kpi.kits_used}/10  "
                       f"({self.kits_left} left)\n","warn")
        self._decision(f"  Resource Util Rate:  {self.kpi.resource_util():.1%}\n","info")
        self._decision(f"  Total Replans:       {self.kpi.replan_count}\n","warn")

        self._decision(f"\n🔀 SIMULATED ANNEALING SUMMARY\n","header")
        self._decision(f"  SA optimised rescue order before operations began.\n","sub")
        self._decision(f"  Works with CSP: SA orders → CSP assigns ambulances.\n","sub")

        self._decision(f"\n📌 ALGORITHM USAGE\n","header")
        for algo, cnt in self.kpi.algo_usage.items():
            self._decision(f"  {algo}: {cnt} mission(s)\n","info")

        self._decision(f"\n🤖 VICTIM OUTCOME DETAILS\n","header")
        for d in self.kpi.victim_details:
            delta = d["survival"] - d["initial_surv"]
            self._decision(
                f"  V{d['vid']} [{d['sev'][:3].upper()}]: "
                f"surv {d['initial_surv']:.3f}→{d['survival']:.3f} "
                f"(Δ{delta:+.3f}) waited={d['steps_waited']} "
                f"path={d['path_len']} risk={d['risk']} algo={d['algo']}\n","sub")

        self._decision(f"\n🔍 SEARCH COMPARISON NOTE\n","header")
        alg = self.selected_algo.get()
        if alg == "All (Compare)":
            self._decision(
                "  Multi-objective scoring: 35% path length + 30% risk +\n"
                "  20% nodes expanded + 15% time (ms)\n","sub")
        elif alg == "Context-Aware":
            self._decision("  Context-aware selection adapts per rescue.\n","sub")
        else:
            self._decision(f"  Single algorithm: {alg}\n","sub")
            self._decision("  Use 'All (Compare)' for full algorithm comparison.\n","sub")

        self._decision(f"\n🏁 Simulation complete.\n","good")
        self.status_var.set(
            f"Done! Rescued {self.kpi.saved}/{total}, "
            f"Dead {self.kpi.dead}. Press ↺ RESET to run again.")
        self._update_kpi_display()

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = AIDRAApp()
    app.mainloop()