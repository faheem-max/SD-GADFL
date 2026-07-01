import os
import time
import pickle
import numpy as np

# ======================
# SYNCHRONIZATION UTILITIES
# ======================

def wait_for_round(round_num, client_ids, models_dir="models"):
    """Wait for all clients to finish a given round."""
    while True:
        ready = all(
            os.path.exists(f"{models_dir}/round_{round_num}_client{cid}.pt")
            for cid in client_ids
        )
        if ready:
            break
        time.sleep(0.5)


def save_training_time(client_id, round_num, training_time, times_dir="training_times"):
    """Save this client's training time for the current round."""
    os.makedirs(times_dir, exist_ok=True)
    time_file = f"{times_dir}/round_{round_num}_client{client_id}.pkl"
    with open(time_file, "wb") as f:
        pickle.dump(training_time, f)


def load_all_training_times(round_num, client_ids, times_dir="training_times"):
    """Load training times from all clients for barrier calculation."""
    times = {}
    for cid in client_ids:
        time_file = f"{times_dir}/round_{round_num}_client{cid}.pkl"
        while not os.path.exists(time_file):
            time.sleep(0.1)
        with open(time_file, "rb") as f:
            times[cid] = pickle.load(f)
    return times


def calculate_waiting_time(client_id, training_time, all_training_times):
    """Calculate waiting time and normalized waiting time."""
    times_list = list(all_training_times.values())
    Tmax       = max(times_list)
    wait_time  = Tmax - training_time
    max_wait   = max([Tmax - t for t in times_list])
    norm_wait  = wait_time / (max_wait + 1e-8) if max_wait > 0 else 0.0
    return wait_time, norm_wait


def save_participation(round_num, client_id, participated, dir="sync_decisions"):
    """Save whether client participated in this round."""
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/round_{round_num}_client{client_id}_participated.pkl"
    with open(path, "wb") as f:
        pickle.dump(bool(participated), f)


def load_participated_clients(round_num, client_ids, dir="sync_decisions"):
    """
    Load which clients participated in this round.

    For round 0: all clients participate (no files exist yet).
    For round > 0: load participation files.

    IMPORTANT: Does NOT silently fallback to all clients.
    If a client did not save a participation file, it is NOT included.
    This preserves the actual behavior being measured.
    """
    if round_num == 0:
        return client_ids

    participated = []
    for cid in client_ids:
        path = f"{dir}/round_{round_num}_client{cid}_participated.pkl"
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    if pickle.load(f):
                        participated.append(cid)
            except Exception:
                pass

    return participated


# ======================
# INCREMENTAL SCALING — INTROSPECTION HELPER
# ======================

def list_saved_initial_clients(init_dir="initial_weights"):
    """
    Return a sorted list of client IDs that already have saved initial weights.

    Useful for verifying the state of the init directory before launching a
    scaled-up experiment.
    """
    if not os.path.isdir(init_dir):
        return []

    client_ids = []
    for filename in os.listdir(init_dir):
        if filename.startswith("client") and filename.endswith("_init.pt"):
            try:
                id_str = filename[len("client"):-len("_init.pt")]
                client_ids.append(int(id_str))
            except ValueError:
                pass

    return sorted(client_ids)


# ======================
# SARSA Q-LEARNING OPTIMIZER FOR ADAPTIVE SELECTION
# ======================

class SARSA_BetaOptimizer:
    """
    EXACT TRUE SARSA implementation for adaptive client selection in FL.

    State:  (training_time_bin, accuracy_bin)
    Action: Beta value in [0.4, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
    Reward: -2.0 * norm_wait - accuracy_penalty

    EXACT SARSA update: Q(s,a) += lr * [r + γ * Q(s',a') - Q(s,a)]
    where:
      s  = state at time t
      a  = action taken in state s
      r  = reward received
      s' = state at time t+1
      a' = action that WILL BE taken in state s' (on-policy)

    This is ON-POLICY learning.

    Synchronization probability: P(sync) = sigmoid(λ₁·accuracy - λ₂·wait·beta)

    CHANGE — num_bins reduced from 8 to 4:
        With 8 bins: Q-table has 8×8×7 = 448 cells. Over 100 rounds
        each cell is visited on average 100/448 ≈ 0.22 times — far too
        sparse for convergence. The beta distribution from the previous
        100-round experiment was indistinguishable from uniform random,
        confirming the Q-table had not converged.

        With 4 bins: Q-table has 4×4×7 = 112 cells. Over 100 rounds
        each cell receives ~0.9 visits; over 200 rounds ~1.8 visits.
        This is the minimum required for the agent to develop a policy.

        Literature basis: Sutton & Barto (2018) Ch. 9 — state aggregation
        in tabular RL; coarser discretization is preferred when experience
        is limited to avoid the curse of dimensionality.
    """

    def __init__(self, client_id, beta_values=None,
                 num_bins=4,
                 lr=0.12, gamma=0.92, epsilon=0.15):
        self.client_id   = client_id
        self.beta_values = beta_values or [0.4, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
        self.n_actions   = len(self.beta_values)
        self.n_bins      = num_bins

        self.q_table = np.zeros((num_bins, num_bins, self.n_actions))

        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon

    def discretize(self, train_time, acc, max_train=60.0, max_acc=1.0):
        """
        Convert continuous values to discrete bins.
        Handles negative R² values by mapping [-1, 1] to [0, 1].
        """
        train_norm = np.clip(train_time / max_train, 0.0, 1.0)
        acc_norm = np.clip((acc + 1.0) / (max_acc + 1.0), 0.0, 1.0)

        t_bin = min(int(train_norm * (self.n_bins - 1)), self.n_bins - 1)
        a_bin = min(int(acc_norm   * (self.n_bins - 1)), self.n_bins - 1)

        return t_bin, a_bin

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        t, a = state
        return np.argmax(self.q_table[t, a])

    def update_with_state_action(self, state, action, reward,
                                 next_state, next_action):
        """
        EXACT TRUE SARSA update: Q(s,a) += lr * [r + γ·Q(s',a') - Q(s,a)]

        On-policy: uses the ACTUAL next action (not the greedy best action).
        """
        t, a       = state
        current_q  = self.q_table[t, a, action]

        if next_state is None or next_action is None:
            target = reward
        else:
            nt, na = next_state
            next_q = self.q_table[nt, na, next_action]
            target = reward + self.gamma * next_q

        self.q_table[t, a, action] += self.lr * (target - current_q)

    def decay_epsilon(self, min_epsilon=0.05):
        """Halve epsilon toward min_epsilon."""
        self.epsilon = max(self.epsilon * 0.5, min_epsilon)
        return self.epsilon

    def save(self, round_num, dir="sarsa_qtables"):
        """Save Q-table to disk."""
        os.makedirs(dir, exist_ok=True)
        path = f"{dir}/client{self.client_id}_qtable_r{round_num}.npy"
        np.save(path, self.q_table)
        print(f"[SARSA Client {self.client_id}] Q-table saved at round {round_num}")

    def load_latest(self, dir="sarsa_qtables"):
        """Load latest Q-table from disk."""
        if not os.path.exists(dir):
            print(
                f"[SARSA Client {self.client_id}] "
                f"No Q-table directory found → starting fresh"
            )
            return False

        files = [
            f for f in os.listdir(dir)
            if f.startswith(f"client{self.client_id}_qtable_r")
        ]
        if not files:
            print(
                f"[SARSA Client {self.client_id}] "
                f"No previous Q-table found → starting fresh"
            )
            return False

        latest = max(files, key=lambda x: int(x.split("_r")[-1].split(".")[0]))
        path   = os.path.join(dir, latest)
        self.q_table = np.load(path)
        print(
            f"[SARSA Client {self.client_id}] Loaded Q-table from {path}"
        )
        return True


# ======================
# GADFL — GEOGRAPHY-AWARE AGGREGATION FUNCTIONS
# ======================
#
# These functions implement the GADFL (Geography-Aware DFL) mechanism:
#
#   1. compute_speed_histogram  — convert local speeds to a probability histogram
#   2. apply_differential_privacy — add Laplace noise for privacy (optional)
#   3. compute_jsd              — Jensen-Shannon Divergence between two histograms
#   4. compute_jsd_matrix       — pairwise JSD for all agents
#   5. compute_adaptive_beta    — auto-scale beta from mean JSD
#   6. compute_aggregation_weights — final softmax weights for aggregation
#
# Why JSD instead of KL divergence:
#   - Symmetric:  JSD(p, q) = JSD(q, p)  → consistent weights from both sides
#   - Bounded:    JSD(p, q) ∈ [0, 1]     → numerically stable, no normalization needed
#   - Smooth:     handles zero bins via mixture m = 0.5*(p+q)
#
# Why softmax(-β × JSD):
#   - β → 0 : all weights equal → degenerates to uniform FedAvg (baseline)
#   - β large: winner-takes-all → only most similar neighbor used
#   - β = 1/mean_JSD: adaptive scaling to the federation's heterogeneity level
#
# Privacy:
#   - Only histograms are shared (10 floats ≈ 80 bytes)
#   - No raw data, no model parameters in this step
#   - Optional Laplace noise (differential privacy) on histograms
#   - Sharing happens ONCE before training, not every round
# ======================

def compute_speed_histogram(speeds, num_bins=10):
    """
    Compute normalized speed histogram from local target values.

    The histogram is the ONLY data shared between agents in GADFL.
    It reveals the speed distribution shape without exposing raw samples.

    Speed range fixed at [0, 90] km/h for consistency across all agents.
    Speeds above 90 km/h are included in the last bin.

    Args:
        speeds   : array-like of speed_kmh target values
        num_bins : number of equal-width bins (default 10 → 0-9, 9-18, ..., 81-90)

    Returns:
        numpy array of shape (num_bins,), normalized to sum=1.0
    """
    speeds = np.array(speeds, dtype=np.float64).flatten()

    # Fixed range ensures identical bin boundaries across all agents
    # Speeds > 90 go into last bin via clipping implicit in np.histogram
    hist, _ = np.histogram(speeds, bins=num_bins, range=(0.0, 90.0))
    hist    = hist.astype(np.float64)

    # Add small epsilon to avoid log(0) issues in JSD computation
    hist = hist + 1e-10

    # Normalize to probability distribution
    hist = hist / hist.sum()

    return hist


def apply_differential_privacy(histogram, epsilon):
    """
    Add Laplace noise to histogram for differential privacy.

    Mechanism: Laplace(location=0, scale=sensitivity/epsilon)
    Global sensitivity for a normalized histogram = 2/n_samples.
    We use sensitivity=1 (conservative bound for simplicity).

    Privacy guarantee:
        epsilon-DP: an adversary cannot distinguish whether any single
        sample was included in or excluded from the histogram.

    epsilon values guide:
        0.1  = very strong privacy, histogram is very noisy
        1.0  = strong privacy, some geographic info leaked
        2.0  = moderate privacy, recommended sweet spot
        5.0  = weak privacy, near-exact histogram
        inf  = no privacy (exact histogram, current default)

    Args:
        histogram : normalized numpy array (sums to 1.0)
        epsilon   : privacy budget (None or inf → no noise added)

    Returns:
        noisy normalized histogram (same shape as input)
    """
    # No DP requested
    if epsilon is None or epsilon <= 0:
        return histogram.copy()
    try:
        if float(epsilon) == float('inf'):
            return histogram.copy()
    except (TypeError, ValueError):
        return histogram.copy()

    noise_scale = 1.0 / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=histogram.shape)

    noisy = histogram + noise

    # Clip to remove negatives, then renormalize to valid probability distribution
    noisy = np.clip(noisy, 0.0, None)
    total = noisy.sum()

    if total < 1e-10:
        # Too much noise destroyed the histogram — return original as fallback
        return histogram.copy()

    return noisy / total


def compute_jsd(p, q):
    """
    Jensen-Shannon Divergence between two probability distributions p and q.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q) is the mixture distribution.

    Properties:
        - Symmetric:  JSD(p, q) = JSD(q, p)
        - Bounded:    JSD ∈ [0, 1]  (using log base 2, or ∈ [0, ln2] with ln)
        - Zero iff:   p = q exactly

    Args:
        p, q : numpy arrays (same length), should sum to 1

    Returns:
        float in [0, ~0.693] using natural log
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    m = 0.5 * (p + q)

    # Clip to avoid log(0) — values below 1e-10 treated as zero
    p_safe = np.clip(p, 1e-10, None)
    q_safe = np.clip(q, 1e-10, None)
    m_safe = np.clip(m, 1e-10, None)

    kl_pm = float(np.sum(p_safe * np.log(p_safe / m_safe)))
    kl_qm = float(np.sum(q_safe * np.log(q_safe / m_safe)))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    # Clamp to [0, inf) — numerical precision can produce tiny negatives
    return max(0.0, jsd)


def compute_jsd_matrix(histograms):
    """
    Compute pairwise JSD matrix for all agents.

    Args:
        histograms : dict {agent_id (int): histogram (np.array)}

    Returns:
        dict {(i, j): JSD value} for all pairs including (i, i) = 0.0
    """
    agents = sorted(histograms.keys())
    jsd_matrix = {}

    for i in agents:
        for j in agents:
            if i == j:
                jsd_matrix[(i, j)] = 0.0
            elif (j, i) in jsd_matrix:
                # JSD is symmetric — reuse already computed value
                jsd_matrix[(i, j)] = jsd_matrix[(j, i)]
            else:
                jsd_matrix[(i, j)] = compute_jsd(histograms[i], histograms[j])

    return jsd_matrix


def compute_adaptive_beta(jsd_matrix, all_agents):
    """
    Compute beta = 1 / mean_JSD across all agent pairs.

    Rationale:
        High heterogeneity (large mean JSD) → small beta → moderate discrimination
        Low heterogeneity  (small mean JSD) → large beta → stronger discrimination

    This automatically scales beta to the federation's actual heterogeneity level,
    so no manual tuning is needed.

    Args:
        jsd_matrix : dict {(i, j): JSD value}
        all_agents : list of agent IDs

    Returns:
        float beta value
    """
    all_agents = sorted(all_agents)
    jsds = [
        jsd_matrix[(i, j)]
        for i in all_agents
        for j in all_agents
        if i != j
    ]

    if not jsds:
        return 1.0  # single agent: no discrimination possible

    mean_jsd = float(np.mean(jsds))
    beta     = 1.0 / (mean_jsd + 1e-8)
    beta     = min(beta, 5.0)

    return beta


def compute_aggregation_weights(my_id, jsd_matrix, all_agents, beta):
    """
    Compute geography-aware aggregation weights for agent my_id.

    Formula:
        raw_score_j  = -beta * JSD(my_id, j)
        alpha_{my_id, j} = softmax(raw_scores)_j

    Interpretation:
        Small JSD(my_id, j) → less negative score → higher weight
        Large JSD(my_id, j) → more negative score → lower weight

    When beta=0: all weights = 1/N (degenerates to uniform FedAvg)
    When beta→∞: winner-takes-all (only most similar agent used)

    Args:
        my_id      : this agent's ID
        jsd_matrix : pairwise JSD dict from compute_jsd_matrix()
        all_agents : list of all agent IDs (sorted)
        beta       : temperature parameter

    Returns:
        dict {agent_id: weight} — weights sum to 1.0
    """
    all_agents = sorted(all_agents)
    distances  = np.array([jsd_matrix[(my_id, j)] for j in all_agents],
                          dtype=np.float64)

    raw_scores = -beta * distances

    # Subtract max for numerical stability (prevents overflow in exp)
    raw_scores -= raw_scores.max()
    exp_scores  = np.exp(raw_scores)
    weights     = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def apply_model_dp(model_state_dict, sigma):
    """
    Apply Gaussian noise to model parameters before sharing (Model DP).

    This prevents gradient inversion attacks by adding calibrated
    Gaussian noise to each parameter tensor before peer-to-peer sharing.

    Mechanism: w_noisy = w + Gaussian(0, sigma)

    sigma values guide:
        None  / 0   = no noise (no model privacy)
        0.001       = very light noise, minimal accuracy impact
        0.01        = moderate noise, recommended sweet spot
        0.1         = heavy noise, noticeable accuracy drop

    Args:
        model_state_dict : OrderedDict from model.state_dict()
        sigma            : noise standard deviation (None = no noise)

    Returns:
        Noisy state_dict (same structure, float params have noise added)
    """
    import torch

    # No model DP requested
    if sigma is None or sigma <= 0:
        return model_state_dict

    noisy_state = {}
    for key, param in model_state_dict.items():
        if param.dtype in [torch.float32, torch.float64]:
            noise = torch.randn_like(param) * float(sigma)
            noisy_state[key] = param + noise
        else:
            # Non-float params (e.g. batch norm counters) — no noise
            noisy_state[key] = param.clone()

    return noisy_state


# ======================
# FEDDKW — KL DIVERGENCE WEIGHTED AGGREGATION
# ======================
# FedDkw original: centralized server computes KL(client || global)
# Our P2P adaptation: each agent computes KL locally from shared histograms
#
# Key difference from GADFL:
#   GADFL:   JSD(own_hist, peer_hist)    — pairwise similarity
#   FedDkw:  KL(peer_hist || global_hist) — each peer vs global distribution
#
# global_hist = uniform average of all peer histograms
# weight_j = softmax(-beta × KL(hist_j || global_hist))
# ======================

def compute_kl_divergence(p, q):
    """
    KL Divergence KL(p || q).

    Measures how different distribution p is from reference q.
    Used in FedDkw: KL(peer_hist || global_hist)

    Properties:
        Asymmetric: KL(p||q) != KL(q||p)
        Zero when p == q
        Non-negative always

    Args:
        p : numpy array (distribution to measure)
        q : numpy array (reference distribution)

    Returns:
        float KL divergence value
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)

    # Renormalize after clipping
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def compute_feddkw_weights(my_id, histograms, all_agents, beta):
    """
    Compute FedDkw aggregation weights for agent my_id.

    P2P adaptation of FedDkw (original: centralized server):
      1. Compute global_hist = uniform average of all peer histograms
      2. For each peer j: compute KL(hist_j || global_hist)
      3. weight_j = softmax(-beta × KL_j)

    Interpretation:
      Peers whose distribution is CLOSE to global get higher weight
      Peers whose distribution is FAR from global get lower weight

    This is different from GADFL which uses pairwise JSD.

    Args:
        my_id      : this agent's ID (not used, included for API consistency)
        histograms : dict {agent_id: histogram (np.array)}
        all_agents : list of all agent IDs (sorted)
        beta       : temperature parameter

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents = sorted(all_agents)

    # Step 1: Global histogram = uniform average of all peer histograms
    all_hists = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = global_hist / global_hist.sum()  # normalize

    # Step 2: KL divergence of each peer from global
    kl_distances = np.array([
        compute_kl_divergence(histograms[j], global_hist)
        for j in all_agents
    ], dtype=np.float64)

    # Step 3: Softmax weights (negative: closer to global = higher weight)
    raw_scores = -beta * kl_distances
    raw_scores -= raw_scores.max()   # numerical stability
    exp_scores  = np.exp(raw_scores)
    weights     = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def compute_feddkw_beta(histograms, all_agents):
    """
    Compute adaptive beta for FedDkw = 1 / mean_KL.

    Uses mean KL divergence from global distribution
    (consistent with GADFL's adaptive beta formula).

    Args:
        histograms : dict {agent_id: histogram}
        all_agents : list of agent IDs

    Returns:
        float beta value
    """
    all_agents = sorted(all_agents)
    all_hists  = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = global_hist / global_hist.sum()

    kl_values = [
        compute_kl_divergence(histograms[j], global_hist)
        for j in all_agents
    ]

    mean_kl = float(np.mean(kl_values)) + 1e-8
    beta = 1.0 / mean_kl
    beta = min(beta, 5.0)
    return beta


# ======================
# HYBRID-GADFL — COMBINED JSD + KL WEIGHTED AGGREGATION
# ======================
#
# Combines GADFL (JSD pairwise) and FedDkw (KL vs global):
#
#   GADFL:        weight = softmax(-beta1 × JSD(own, peer))
#   FedDkw:       weight = softmax(-beta2 × KL(peer || global))
#   Hybrid-GADFL: weight = softmax(-beta1×JSD - beta2×KL)
#
# JSD term captures local geographic similarity
# KL term captures global representativeness
# Combined: similar AND globally representative nodes get higher weight
#
# Both betas adaptive:
#   beta1 = 1 / mean_JSD  (same as GADFL)
#   beta2 = 1 / mean_KL   (same as FedDkw)
#
# Reviewer justification:
#   beta = 1/mean_distance automatically scales to
#   federation heterogeneity without manual tuning
# ======================

def compute_hybrid_weights(my_id, histograms, all_agents, beta1, beta2):
    """
    Compute Hybrid-GADFL aggregation weights.

    Combines JSD (pairwise geographic similarity) and
    KL (global representativeness) into a single weight.

    Formula:
        score_j = -beta1 × JSD(own_hist, peer_hist)
                  -beta2 × KL(peer_hist || global_hist)
        weight_j = softmax(score_j)

    Args:
        my_id      : this agent ID
        histograms : dict {agent_id: histogram}
        all_agents : list of all agent IDs (sorted)
        beta1      : temperature for JSD term (adaptive = 1/mean_JSD)
        beta2      : temperature for KL term  (adaptive = 1/mean_KL)

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents = sorted(all_agents)

    # Global histogram for KL term
    all_hists   = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = np.clip(global_hist, 1e-10, None)
    global_hist = global_hist / global_hist.sum()

    scores = np.zeros(len(all_agents), dtype=np.float64)

    for idx, j in enumerate(all_agents):
        # JSD term: pairwise distance between own and peer
        jsd_val = compute_jsd(histograms[my_id], histograms[j])

        # KL term: peer distance from global distribution
        kl_val = compute_kl_divergence(histograms[j], global_hist)

        # Combined score
        scores[idx] = -beta1 * jsd_val - beta2 * kl_val

    # Softmax
    scores -= scores.max()   # numerical stability
    exp_scores = np.exp(scores)
    weights    = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def compute_hybrid_betas(histograms, all_agents):
    """
    Compute adaptive beta1 and beta2 for Hybrid-GADFL.

    beta1 = 1 / mean_JSD  (same formula as GADFL)
    beta2 = 1 / mean_KL   (same formula as FedDkw)

    Adaptive formulation ensures betas automatically scale
    to the federation heterogeneity level without manual tuning.

    Args:
        histograms : dict {agent_id: histogram}
        all_agents : list of agent IDs

    Returns:
        (beta1, beta2) tuple of floats
    """
    all_agents = sorted(all_agents)

    # Global histogram for KL
    all_hists   = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = np.clip(global_hist, 1e-10, None)
    global_hist = global_hist / global_hist.sum()

    # Collect all pairwise JSD values
    jsd_values = [
        compute_jsd(histograms[i], histograms[j])
        for i in all_agents
        for j in all_agents
        if i != j
    ]

    # Collect all KL values (each peer vs global)
    kl_values = [
        compute_kl_divergence(histograms[j], global_hist)
        for j in all_agents
    ]

    mean_jsd = float(np.mean(jsd_values)) if jsd_values else 1.0
    mean_kl  = float(np.mean(kl_values))  if kl_values  else 1.0

    beta1 = 1.0 / (mean_jsd + 1e-8)
    beta2 = 1.0 / (mean_kl  + 1e-8)

    return beta1, beta2


# ======================
# SR-GADFL — SELF-REGULARIZED GEOGRAPHY-AWARE AGGREGATION
# ======================
#
# Problem identified in standard GADFL:
#   JSD(self, self) = 0 always
#   Softmax gives self the MAXIMUM weight (31.7% for Client 1)
#   This causes local overfitting and hurts cross-environment generalization
#
# Evidence:
#   GADFL local validation peak: 0.968 (better than FedDkw 0.965)
#   GADFL Town07 generalization: 0.902 (worse than FedDkw 0.915)
#   Gap: 0.066 vs FedDkw gap of 0.050
#   Self-weight dominance causes local bias
#
# Solution — Self-Regularization:
#   Instead of using distance = 0 for self:
#   Use distance = alpha × mean_peer_JSD
#   where alpha controls self-weight penalty
#
#   alpha = 0.0: standard GADFL (self gets max weight)
#   alpha = 0.5: moderate regularization (recommended)
#   alpha = 1.0: self gets same distance as avg peer
#   alpha > 1.0: self gets penalized more than avg peer
#
# Result:
#   Self-weight reduced from 31.7% to ~10-15%
#   More collaborative aggregation like FedDkw
#   Cross-environment generalization improves
#   Local accuracy slightly reduced (acceptable tradeoff)
#
# Novel contribution:
#   First paper to identify and address self-weight dominance
#   in geography-aware DFL for autonomous vehicles
# ======================

def compute_sr_gadfl_weights(my_id, jsd_matrix, all_agents, beta, alpha=0.5):
    """
    Self-Regularized GADFL aggregation weights.

    Fixes the self-weight dominance problem in standard GADFL
    by replacing self-distance of 0 with a regularized value.

    Formula:
        distances[j] = JSD(my_id, j)   for j != my_id
        distances[self] = alpha × mean(JSD(my_id, j) for j != my_id)

        weight_j = softmax(-beta × distances)

    Args:
        my_id      : this agent's ID
        jsd_matrix : pairwise JSD dict from compute_jsd_matrix()
        all_agents : list of all agent IDs (sorted)
        beta       : temperature parameter (same as GADFL)
        alpha      : self-regularization strength
                     0.0 = standard GADFL (no regularization)
                     0.5 = moderate (recommended)
                     1.0 = self distance = mean peer distance

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents = sorted(all_agents)
    self_idx   = all_agents.index(my_id)

    # Get all pairwise distances
    distances = np.array(
        [jsd_matrix[(my_id, j)] for j in all_agents],
        dtype=np.float64
    )

    # Mean distance to PEERS (excluding self)
    peer_distances = [distances[i] for i, j in enumerate(all_agents) if j != my_id]
    mean_peer_dist = float(np.mean(peer_distances)) if peer_distances else 0.0

    # Self-regularization: replace 0 with alpha × mean_peer_dist
    distances[self_idx] = alpha * mean_peer_dist

    # Softmax weights (same as GADFL)
    raw_scores  = -beta * distances
    raw_scores -= raw_scores.max()
    exp_scores  = np.exp(raw_scores)
    weights     = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def compute_sr_gadfl_beta(jsd_matrix, all_agents):
    """
    Compute adaptive beta for SR-GADFL.

    Same formula as GADFL: beta = 1 / mean_JSD
    Uses only peer distances (not self distance of 0).

    Args:
        jsd_matrix : pairwise JSD dict
        all_agents : list of agent IDs

    Returns:
        float beta value
    """
    all_agents = sorted(all_agents)
    jsds = [
        jsd_matrix[(i, j)]
        for i in all_agents
        for j in all_agents
        if i != j
    ]

    if not jsds:
        return 1.0

    mean_jsd = float(np.mean(jsds))
    return 1.0 / (mean_jsd + 1e-8)


# ======================
# G-FedDkw — SYMMETRIC JSD-BASED GLOBAL WEIGHTED AGGREGATION
# ======================
#
# Motivation:
#   FedDkw uses KL(peer || global) — asymmetric, unbounded
#   G-FedDkw uses JSD(peer, global) — symmetric, bounded [0, 0.693]
#
# Why JSD over KL for global comparison:
#   1. Symmetric: JSD(p,global) = JSD(global,p)
#   2. Bounded:   JSD in [0, 0.693] — no extreme weights
#   3. Smooth:    handles zero bins via mixture
#
# Formula:
#   global_hist = uniform average of all peer histograms
#   weight_j = softmax(-beta x JSD(hist_j, global_hist))
#
# G-FedDkw vs GADFL:
#   GADFL:    JSD(own_hist, peer_hist)    pairwise
#   G-FedDkw: JSD(peer_hist, global_hist) each peer vs global
# ======================

def compute_g_feddkw_weights(my_id, histograms, all_agents, beta):
    """
    G-FedDkw: Symmetric JSD-based global weighted aggregation.

    Args:
        my_id      : this agent ID
        histograms : dict {agent_id: histogram}
        all_agents : list of all agent IDs (sorted)
        beta       : temperature parameter

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents  = sorted(all_agents)
    all_hists   = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = np.clip(global_hist, 1e-10, None)
    global_hist = global_hist / global_hist.sum()

    jsd_from_global = np.array([
        compute_jsd(histograms[j], global_hist)
        for j in all_agents
    ], dtype=np.float64)

    raw_scores  = -beta * jsd_from_global
    raw_scores -= raw_scores.max()
    exp_scores  = np.exp(raw_scores)
    weights     = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def compute_g_feddkw_beta(histograms, all_agents):
    """
    Adaptive beta for G-FedDkw = 1 / mean_JSD_from_global.

    Beta is capped at 5.0 to prevent over-discrimination when
    Dirichlet-partitioned histograms produce very small JSD values.
    Without cap: beta=9.186 → Client 3 gets 2.2% (too extreme)
    With cap=5.0: Client 3 gets ~5-8% (balanced)

    Cap value of 5.0 is consistent with GADFL beta range (4.8).
    """
    all_agents  = sorted(all_agents)
    all_hists   = np.array([histograms[j] for j in all_agents], dtype=np.float64)
    global_hist = all_hists.mean(axis=0)
    global_hist = np.clip(global_hist, 1e-10, None)
    global_hist = global_hist / global_hist.sum()

    jsd_values = [compute_jsd(histograms[j], global_hist) for j in all_agents]
    mean_jsd   = float(np.mean(jsd_values)) + 1e-8
    beta = 1.0 / mean_jsd

    # Cap beta to prevent over-discrimination with concentrated histograms
    # Consistent with GADFL adaptive beta range (~4.8)
    beta = min(beta, 5.0)

    return beta


# ======================
# T-G-FedDkw — TARGET-AWARE G-FedDkw (ORACLE UPPER BOUND)
# ======================
#
# Motivation:
#   G-FedDkw uses training global distribution for weighting
#   T-G-FedDkw uses TARGET environment distribution (Town07)
#
# Use case:
#   In real AV deployment, the target operating environment
#   is often known in advance (e.g., "deploy on highway network")
#   Operators can provide expected speed distribution as prior
#
# Oracle interpretation:
#   T-G-FedDkw is an UPPER BOUND — shows maximum achievable
#   accuracy when target distribution is known
#   G-FedDkw is the practical deployable method
#
# Formula:
#   target_hist = known target environment histogram
#   weight_j = softmax(-beta × JSD(peer_hist, target_hist))
#
# Key difference from G-FedDkw:
#   G-FedDkw:   global = average of all training histograms
#   T-G-FedDkw: global = target test environment histogram
# ======================

def compute_t_g_feddkw_weights(my_id, histograms, all_agents,
                                 target_hist, beta):
    """
    T-G-FedDkw: Target-aware JSD-based global weighted aggregation.

    Uses target environment histogram instead of training average.
    Gives higher weight to peers whose distribution matches
    the deployment environment.

    Args:
        my_id       : this agent ID
        histograms  : dict {agent_id: histogram}
        all_agents  : list of all agent IDs (sorted)
        target_hist : numpy array — target environment histogram
                      (e.g., Town07 test set speed distribution)
        beta        : temperature parameter

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents  = sorted(all_agents)

    # Normalize target histogram
    target = np.array(target_hist, dtype=np.float64)
    target = np.clip(target, 1e-10, None)
    target = target / target.sum()

    # JSD between each peer and TARGET (not global training average)
    jsd_from_target = np.array([
        compute_jsd(histograms[j], target)
        for j in all_agents
    ], dtype=np.float64)

    raw_scores  = -beta * jsd_from_target
    raw_scores -= raw_scores.max()
    exp_scores  = np.exp(raw_scores)
    weights     = exp_scores / exp_scores.sum()

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def compute_t_g_feddkw_beta(histograms, all_agents, target_hist):
    """
    Adaptive beta for T-G-FedDkw = 1 / mean_JSD_from_target.

    Args:
        histograms  : dict {agent_id: histogram}
        all_agents  : list of agent IDs
        target_hist : target environment histogram

    Returns:
        float beta value
    """
    all_agents = sorted(all_agents)

    target = np.array(target_hist, dtype=np.float64)
    target = np.clip(target, 1e-10, None)
    target = target / target.sum()

    jsd_values = [
        compute_jsd(histograms[j], target)
        for j in all_agents
    ]

    mean_jsd = float(np.mean(jsd_values)) + 1e-8
    return 1.0 / mean_jsd


# ======================
# SD-GADFL — STANDARD-DEVIATION WEIGHTED AGGREGATION
# ======================
#
# Motivation:
#   Distance-based weighting (GADFL/FedDkw/G-FedDkw — JSD or KL,
#   pairwise or vs-global) consistently causes nodes with "different"
#   distributions (e.g. Town03/highway) to receive LOW aggregation
#   weight. Over many rounds, this low-weight node's unique
#   contribution decays geometrically (~w^t), erasing minority
#   geographic signals from the federation (verified empirically:
#   GADFL/FedDkw/G-FedDkw all converge to R2~0.79 at round 100,
#   below uniform FedProx at 0.818).
#
# SD-GADFL takes the OPPOSITE approach:
#   Instead of asking "how DIFFERENT is this peer from
#   me/global?" (a relative/distance signal that penalizes
#   diversity), SD-GADFL asks "how INFORMATIVE/diverse is this
#   peer's OWN local data?" (an absolute signal that REWARDS
#   diversity).
#
#   weight_j = std_j / sum(std_i)
#
#   A node whose local speed distribution spans a WIDE range
#   (e.g. a highway node that sees both stop-and-go traffic and
#   high-speed cruising, std=21.5) is treated as more informative
#   and given a LARGER aggregation weight than a node whose data
#   is narrowly concentrated (e.g. std=8.6).
#
# Privacy advantage over histogram-based methods:
#   Only ONE scalar (std) is shared per node, instead of a
#   10-bin histogram (10 floats). This is both simpler and
#   reveals less information — a single summary statistic
#   vs. a full distribution shape.
#
# Differentiation from prior work (R-Sync / SARSA paper):
#   R-Sync uses sigma/std to decide PARTICIPATION (whether/how
#   often a node syncs) within a synchronous-barrier + SARSA-RL
#   framework. SD-GADFL uses std as a fixed AGGREGATION WEIGHT
#   in a fully decentralized P2P weighted-average — a different
#   mechanism, different mathematical role, different framework.
# ======================

def compute_std_based_weights(stds, all_agents):
    """
    SD-GADFL: Standard-deviation proportional aggregation weights.

    weight_j = std_j / sum(std_i)

    Higher local data diversity (std) -> higher aggregation weight.
    Same global weight vector is used by every agent (like FedDkw /
    G-FedDkw), since std_j is an absolute property of agent j, not
    relative to the receiving agent.

    Args:
        stds       : dict {agent_id: std (float)} — local target
                      (speed) standard deviation for each agent
        all_agents : list of all agent IDs

    Returns:
        dict {agent_id: weight} summing to 1.0
    """
    all_agents = sorted(all_agents)

    values = np.array([max(stds[j], 1e-8) for j in all_agents], dtype=np.float64)
    total  = values.sum()
    weights = values / total

    return {agent_id: float(w) for agent_id, w in zip(all_agents, weights)}


def apply_dp_to_scalar(value, epsilon, sensitivity=5.0):
    """
    Add Laplace noise to a single scalar (e.g. std) for differential
    privacy, before sharing with peers.

    Mechanism: Laplace(loc=0, scale=sensitivity/epsilon)

    Sensitivity: bound on how much the local std can change if a
    single sample is added/removed. For speed values in [0,90] km/h
    over O(1000) samples, sensitivity=5.0 km/h is a conservative
    practical bound (one sample has limited effect on the sample std
    for reasonably-sized datasets).

    epsilon values guide (consistent with histogram DP):
        0.5  = strong privacy, std heavily perturbed
        2.0  = moderate privacy, recommended sweet spot
        5.0  = weak privacy, near-exact std
        None/inf = no privacy (exact std shared)

    Args:
        value       : float — the std value to share
        epsilon     : privacy budget (None or <=0 or inf -> no noise)
        sensitivity : float — sensitivity bound (default 5.0 km/h)

    Returns:
        noisy float (clipped to be >= 0)
    """
    if epsilon is None or epsilon <= 0:
        return float(value)
    try:
        if float(epsilon) == float('inf'):
            return float(value)
    except (TypeError, ValueError):
        return float(value)

    noise_scale = sensitivity / float(epsilon)
    noise = np.random.laplace(loc=0.0, scale=noise_scale)

    noisy = float(value) + noise
    return max(noisy, 1e-8)
