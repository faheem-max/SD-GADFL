#!/usr/bin/env python3
"""
Distributed Federated Learning over HTTP
Single client supporting:
- FedAvg   baseline
- FedProx  baseline
- SARSA    SARSA-based selective synchronization (R-Sync paper)
- GADFL    Geography-Aware DFL with JSD-weighted aggregation  ← NEW

Algorithm selection is done via config file (algorithm: fedavg/fedprox/sarsa/gadfl).
All outputs are saved in algorithm-specific folders.

=========================================================
GADFL-specific behavior:
  - Everyone participates every round (no selective participation)
  - Full synchronization barrier (same as FedAvg/FedProx)
  - Local training: FedProx with proximal regularization (mu from gadfl config)
  - Aggregation: WEIGHTED average using JSD-based geography weights
  - Pre-training (ONE TIME): histogram sharing, JSD computation, weight computation

GADFL pre-training flow (before round 1):
  1. Compute local speed histogram from y_train (10 bins, 0-90 km/h)
  2. Optionally apply Laplace noise (differential privacy, controlled by dp_epsilon)
  3. Share histogram with all peers via POST /sync/share_histogram
  4. Poll own server until all peers have shared → GET /sync/get_all_histograms
  5. Compute pairwise JSD matrix between all agent histograms
  6. Compute adaptive beta = 1 / mean_JSD
  7. Compute aggregation weights: softmax(-beta * JSD) per peer
  8. Save histogram, JSD matrix, weights to CSV

GADFL per-round flow (rounds 1..T):
  1. Wait for all peers (full barrier)
  2. Fetch all peer models
  3. Weighted aggregation: Σ alpha_j * model_j  (instead of uniform mean)
  4. FedProx local training
  5. Upload model
  6. Broadcast barrier

CSV files saved (same format as other algorithms):
  - client{id}_metrics_gadfl.csv   : per-round train/test metrics
  - client{id}_waiting_gadfl.csv   : per-round timing (SARSA columns = nan)
  - client{id}_times.csv           : training times

GADFL-specific CSV files (saved ONCE before training):
  - client{id}_histogram_gadfl.csv          : local speed histogram (10 bins)
  - client{id}_jsd_matrix_gadfl.csv         : pairwise JSD matrix
  - client{id}_aggregation_weights_gadfl.csv : final weights used each round
=========================================================

SARSA-specific changes (unchanged from R-Sync paper):
  CHANGE 1 — wait_for_round_completion: conditional barrier.
  CHANGE 2 — Skipping clients broadcast barrier IMMEDIATELY.
  CHANGE 3 — effective_waiting_time logged to CSV.
  CHANGE 4 — Epsilon decay at round 20.
"""

import asyncio
import logging
import os
import random
import sys
import time
import traceback

import aiohttp
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from utils.model_utils import (
    RegressionModel,
    train_local_fedprox,
    train_local_batched,
    evaluate,
    fedavg,
    weighted_fedavg,
    load_or_init_weights,
)
from utils.sync_utils import (
    save_participation,
    SARSA_BetaOptimizer,
    list_saved_initial_clients,
    # GADFL functions
    compute_speed_histogram,
    apply_differential_privacy,
    compute_jsd_matrix,
    compute_adaptive_beta,
    compute_aggregation_weights,
    apply_model_dp,
    # FedDkw functions
    compute_kl_divergence,
    compute_feddkw_weights,
    compute_feddkw_beta,
    # Hybrid-GADFL functions
    compute_hybrid_weights,
    compute_hybrid_betas,
    # SR-GADFL functions
    compute_sr_gadfl_weights,
    compute_sr_gadfl_beta,
    # G-FedDkw functions
    compute_g_feddkw_weights,
    compute_g_feddkw_beta,
    compute_jsd,
    # T-G-FedDkw functions
    compute_t_g_feddkw_weights,
    compute_t_g_feddkw_beta,
    # SD-GADFL functions
    compute_std_based_weights,
    apply_dp_to_scalar,
)

from network.http_server import DFLServer
from network.http_client import DFLClient


def setup_logging(client_id: int, algorithm: str, logs_dir: str) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f"client{client_id}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
    )

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(sh)


logger = logging.getLogger(__name__)


class DistributedClient:
    def __init__(
        self,
        client_id             : int,
        config_path           : str = "config/peer_config.yaml",
        dp_epsilon_override         = "USE_CONFIG",
        model_sigma_override        = "USE_CONFIG",
    ):
        """
        Args:
            client_id             : peer ID (1-7)
            config_path           : path to peer_config.yaml
            dp_epsilon_override   : histogram DP epsilon
                                    "USE_CONFIG" -> read from peer_config.yaml
                                    None         -> no histogram DP
                                    float        -> use this epsilon value
            model_sigma_override  : model DP sigma (Gaussian noise on model)
                                    "USE_CONFIG" -> read from peer_config.yaml
                                    None         -> no model DP
                                    float        -> use this sigma value
        """
        self.CLIENT_ID              = int(client_id)
        self._dp_epsilon_override   = dp_epsilon_override
        self._model_sigma_override  = model_sigma_override

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.ALGORITHM = str(self.config.get("algorithm", "fedavg")).lower()

        if self.ALGORITHM not in ["fedavg", "fedprox", "sarsa", "gadfl", "feddkw", "hybrid_gadfl", "sr_gadfl", "g_feddkw", "t_g_feddkw", "sd_gadfl"]:
            raise ValueError(
                f"Invalid algorithm '{self.ALGORITHM}'. "
                f"Must be one of: fedavg, fedprox, sarsa, gadfl, feddkw, hybrid_gadfl, sr_gadfl, g_feddkw, t_g_feddkw, sd_gadfl"
            )

        raw_peers = self.config["peers"]
        self.peers_by_int_id = {int(k): v for k, v in raw_peers.items()}

        if self.CLIENT_ID not in self.peers_by_int_id:
            raise KeyError(
                f"Client {self.CLIENT_ID} not found in peer config. "
                f"Available IDs: {sorted(self.peers_by_int_id.keys())}"
            )

        self.peer_config  = self.peers_by_int_id[self.CLIENT_ID]
        self.CLIENT_IDS   = sorted(self.peers_by_int_id.keys())

        self.SEED         = int(self.config["training"]["seed"])
        self.VAL_SPLIT    = float(self.config["training"]["val_split"])
        self.TOTAL_ROUNDS = int(self.config["training"]["total_rounds"])
        self.LOCAL_EPOCHS = int(self.config["training"]["local_epochs"])
        self.BATCH_SIZE   = int(self.config["training"]["batch_size"])
        self.LR           = float(self.config["training"]["learning_rate"])
        self.TARGET       = self.config["training"]["target_column"]

        # ── Algorithm-specific config ──────────────────────────────────────
        if self.ALGORITHM == "fedprox":
            self.MU = float(self.config["fedprox"].get("mu", 0.03))
        elif self.ALGORITHM == "gadfl":
            gadfl_cfg = self.config.get("gadfl", {})
            self.MU   = float(gadfl_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "feddkw":
            feddkw_cfg = self.config.get("feddkw", {})
            self.MU    = float(feddkw_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "hybrid_gadfl":
            hybrid_cfg = self.config.get("hybrid_gadfl", {})
            self.MU    = float(hybrid_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "sr_gadfl":
            sr_cfg  = self.config.get("sr_gadfl", {})
            self.MU = float(sr_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "g_feddkw":
            g_cfg   = self.config.get("g_feddkw", {})
            self.MU = float(g_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "t_g_feddkw":
            tg_cfg  = self.config.get("t_g_feddkw", {})
            self.MU = float(tg_cfg.get("mu", 0.03))
        elif self.ALGORITHM == "sd_gadfl":
            sd_cfg  = self.config.get("sd_gadfl", {})
            self.MU = float(sd_cfg.get("mu", 0.03))
        else:
            self.MU = None

        if self.ALGORITHM == "sarsa":
            sarsa_cfg    = self.config.get("sarsa", {})
            self.LAMBDA1 = float(sarsa_cfg.get("lambda1", 1.8))
            self.LAMBDA2 = float(sarsa_cfg.get("lambda2", 1.5))
            self.LAMBDA3 = float(sarsa_cfg.get("lambda3", 2.5))
            self.SARSA_LR = float(sarsa_cfg.get("learning_rate", 0.12))
            self.GAMMA   = float(sarsa_cfg.get("gamma", 0.92))
            self.EPSILON = float(sarsa_cfg.get("epsilon", 0.15))
            self.DIST_WEIGHT_MODE = str(
                sarsa_cfg.get("distribution_weight_mode", "variance")
            ).lower()
            raw_oracle = sarsa_cfg.get("oracle_weights", {})
            self.ORACLE_WEIGHTS = {int(k): float(v) for k, v in raw_oracle.items()}
        else:
            self.LAMBDA1  = None
            self.LAMBDA2  = None
            self.LAMBDA3  = None
            self.SARSA_LR = None
            self.GAMMA    = None
            self.EPSILON  = None
            self.DIST_WEIGHT_MODE = None
            self.ORACLE_WEIGHTS   = {}

        if self.ALGORITHM == "gadfl":
            gadfl_cfg         = self.config.get("gadfl", {})
            self.GADFL_NUM_BINS = int(gadfl_cfg.get("num_bins", 10))
            beta_raw = gadfl_cfg.get("beta", "adaptive")
            if str(beta_raw).lower() == "adaptive":
                self.GADFL_BETA_MODE = "adaptive"
                self.GADFL_BETA_FIXED = None
            else:
                self.GADFL_BETA_MODE  = "fixed"
                self.GADFL_BETA_FIXED = float(beta_raw)
            # dp_epsilon: histogram DP
            if self._dp_epsilon_override != "USE_CONFIG":
                self.GADFL_DP_EPSILON = self._dp_epsilon_override
            else:
                dp_raw = gadfl_cfg.get("dp_epsilon", None)
                self.GADFL_DP_EPSILON = float(dp_raw) if dp_raw is not None else None
            # model_sigma: model DP
            if self._model_sigma_override != "USE_CONFIG":
                self.GADFL_MODEL_SIGMA = self._model_sigma_override
            else:
                sig_raw = gadfl_cfg.get("model_sigma", None)
                self.GADFL_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None
        else:
            self.GADFL_NUM_BINS    = None
            self.GADFL_BETA_MODE   = None
            self.GADFL_BETA_FIXED  = None
            self.GADFL_DP_EPSILON  = None
            self.GADFL_MODEL_SIGMA = None

        # ── FedDkw config ──────────────────────────────────────────────────
        if self.ALGORITHM == "feddkw":
            feddkw_cfg          = self.config.get("feddkw", {})
            self.FEDDKW_NUM_BINS = int(feddkw_cfg.get("num_bins", 10))
            beta_raw = feddkw_cfg.get("beta", "adaptive")
            if str(beta_raw).lower() == "adaptive":
                self.FEDDKW_BETA_MODE  = "adaptive"
                self.FEDDKW_BETA_FIXED = None
            else:
                self.FEDDKW_BETA_MODE  = "fixed"
                self.FEDDKW_BETA_FIXED = float(beta_raw)
        else:
            self.FEDDKW_NUM_BINS   = None
            self.FEDDKW_BETA_MODE  = None
            self.FEDDKW_BETA_FIXED = None

        # ── Hybrid-GADFL config ────────────────────────────────────────────
        if self.ALGORITHM == "hybrid_gadfl":
            hybrid_cfg = self.config.get("hybrid_gadfl", {})
            self.HYBRID_NUM_BINS = int(hybrid_cfg.get("num_bins", 10))

            # dp_epsilon: histogram DP (same as GADFL)
            if self._dp_epsilon_override != "USE_CONFIG":
                self.HYBRID_DP_EPSILON = self._dp_epsilon_override
            else:
                dp_raw = hybrid_cfg.get("dp_epsilon", None)
                self.HYBRID_DP_EPSILON = float(dp_raw) if dp_raw is not None else None

            # model_sigma: model DP (same as GADFL)
            if self._model_sigma_override != "USE_CONFIG":
                self.HYBRID_MODEL_SIGMA = self._model_sigma_override
            else:
                sig_raw = hybrid_cfg.get("model_sigma", None)
                self.HYBRID_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None
        else:
            self.HYBRID_NUM_BINS    = None
            self.HYBRID_DP_EPSILON  = None
            self.HYBRID_MODEL_SIGMA = None

        # ── SR-GADFL config ────────────────────────────────────────────────
        if self.ALGORITHM == "sr_gadfl":
            sr_cfg = self.config.get("sr_gadfl", {})
            self.SR_NUM_BINS = int(sr_cfg.get("num_bins", 10))
            self.SR_ALPHA    = float(sr_cfg.get("alpha", 0.5))

            # dp_epsilon: histogram DP (same as GADFL)
            if self._dp_epsilon_override != "USE_CONFIG":
                self.SR_DP_EPSILON = self._dp_epsilon_override
            else:
                dp_raw = sr_cfg.get("dp_epsilon", None)
                self.SR_DP_EPSILON = float(dp_raw) if dp_raw is not None else None

            # model_sigma: model DP (same as GADFL)
            if self._model_sigma_override != "USE_CONFIG":
                self.SR_MODEL_SIGMA = self._model_sigma_override
            else:
                sig_raw = sr_cfg.get("model_sigma", None)
                self.SR_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None
        else:
            self.SR_NUM_BINS    = None
            self.SR_ALPHA       = None
            self.SR_DP_EPSILON  = None
            self.SR_MODEL_SIGMA = None

        # ── G-FedDkw config ────────────────────────────────────────────────
        if self.ALGORITHM == "g_feddkw":
            g_cfg = self.config.get("g_feddkw", {})
            self.GFEDDKW_NUM_BINS = int(g_cfg.get("num_bins", 10))
            dp_raw  = g_cfg.get("dp_epsilon", None)
            self.GFEDDKW_DP_EPSILON  = float(dp_raw)  if dp_raw  is not None else None
            sig_raw = g_cfg.get("model_sigma", None)
            self.GFEDDKW_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None

            # Raw histogram: use full town data for aggregation weights
            # (geographic reality) while training on Dirichlet-partitioned
            # data (realistic non-IID FL conditions).
            #
            # Path is read from THIS node's own peer entry
            # (peers[CLIENT_ID].raw_histogram_path) so a single shared
            # peer_config.yaml works for all nodes — each node picks
            # its own raw town file automatically.
            #
            # Fallback: g_feddkw.raw_histogram_path (single-node override)
            # null/missing = use Dirichlet-partitioned data for histogram (old behavior)
            peers_cfg   = self.config.get("peers", {})
            my_peer_cfg = peers_cfg.get(self.CLIENT_ID,
                              peers_cfg.get(str(self.CLIENT_ID), {}))
            raw_path = my_peer_cfg.get("raw_histogram_path",
                           g_cfg.get("raw_histogram_path", None))
            self.GFEDDKW_RAW_HIST_PATH  = str(raw_path) if raw_path else None
            self.GFEDDKW_RAW_SPEED_COL  = str(g_cfg.get("raw_speed_col", "speed_kmh"))
        else:
            self.GFEDDKW_NUM_BINS      = None
            self.GFEDDKW_DP_EPSILON    = None
            self.GFEDDKW_MODEL_SIGMA   = None
            self.GFEDDKW_RAW_HIST_PATH = None
            self.GFEDDKW_RAW_SPEED_COL = None

        # ── T-G-FedDkw config ──────────────────────────────────────────────
        if self.ALGORITHM == "t_g_feddkw":
            tg_cfg = self.config.get("t_g_feddkw", {})
            self.TG_NUM_BINS          = int(tg_cfg.get("num_bins", 10))
            self.TG_TARGET_DATA_PATH  = str(tg_cfg.get("target_data_path",
                                            "/home/ubuntu/distributed_avg/test_set_cleaned.csv"))
            self.TG_TARGET_SPEED_COL  = str(tg_cfg.get("target_speed_col", "speed_kmh"))
            dp_raw  = tg_cfg.get("dp_epsilon", None)
            self.TG_DP_EPSILON  = float(dp_raw)  if dp_raw  is not None else None
            sig_raw = tg_cfg.get("model_sigma", None)
            self.TG_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None
        else:
            self.TG_NUM_BINS         = None
            self.TG_TARGET_DATA_PATH = None
            self.TG_TARGET_SPEED_COL = None
            self.TG_DP_EPSILON       = None
            self.TG_MODEL_SIGMA      = None

        # ── SD-GADFL config ────────────────────────────────────────────────
        if self.ALGORITHM == "sd_gadfl":
            sd_cfg = self.config.get("sd_gadfl", {})

            # dp_epsilon: CLI override takes priority over config file
            if self._dp_epsilon_override != "USE_CONFIG":
                self.SD_DP_EPSILON = self._dp_epsilon_override
            else:
                dp_raw = sd_cfg.get("dp_epsilon", None)
                self.SD_DP_EPSILON = float(dp_raw) if dp_raw is not None else None

            # model_sigma: CLI override takes priority over config file
            if self._model_sigma_override != "USE_CONFIG":
                self.SD_MODEL_SIGMA = self._model_sigma_override
            else:
                sig_raw = sd_cfg.get("model_sigma", None)
                self.SD_MODEL_SIGMA = float(sig_raw) if sig_raw is not None else None

            self.SD_SENSITIVITY = float(sd_cfg.get("dp_sensitivity", 5.0))
        else:
            self.SD_DP_EPSILON  = None
            self.SD_MODEL_SIGMA = None
            self.SD_SENSITIVITY = None

        network_cfg = self.config.get("network", {})
        self.HTTP_TIMEOUT        = int(network_cfg.get("timeout_seconds", 10))
        self.BARRIER_TIMEOUT     = int(network_cfg.get("barrier_timeout_seconds", 30))
        self.HTTP_RETRIES        = int(network_cfg.get("retry_attempts", 2))
        self.FETCH_POLL_INTERVAL = float(network_cfg.get("poll_interval_seconds", 0.5))
        self.SERVER_STARTUP_SECONDS = 1.5

        scaling_cfg = self.config.get("scaling", {})
        self.initial_weights_dir = scaling_cfg.get(
            "initial_weights_dir", "initial_weights"
        )

        self._setup_directories()

        client_seed = self.SEED + self.CLIENT_ID
        random.seed(client_seed)
        np.random.seed(client_seed)
        torch.manual_seed(client_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        logger.info(f"[Client {self.CLIENT_ID}] startup delay...")
        time.sleep(random.uniform(1.5, 5.5))
        logger.info(
            f"[Client {self.CLIENT_ID}] Using algorithm: {self.ALGORITHM.upper()}"
        )

        self.device = torch.device("cpu")

        # SARSA agent: only for SARSA algorithm
        self.agent = (
            SARSA_BetaOptimizer(self.CLIENT_ID)
            if self.ALGORITHM == "sarsa" else None
        )

        self.server = DFLServer(
            peer_id=self.CLIENT_ID,
            config=self.config,
            port=self.peer_config["port"],
        )
        self.client = DFLClient(
            peer_id=self.CLIENT_ID,
            peer_config=self.peers_by_int_id,
            timeout=self.HTTP_TIMEOUT,
            retry_attempts=self.HTTP_RETRIES,
        )

        self.model     = None
        self.X_train   = None
        self.y_train   = None
        self.X_val     = None
        self.y_val     = None
        self.input_dim = None

        # SARSA state variables
        self.sarsa_prev_state  = None
        self.sarsa_prev_action = None
        self.prev_accuracy        = 0.0
        self.prev_training_time   = 0.5
        self.prev_straggler_score = 0.0
        self.distribution_weight  = 0.5  # for SARSA; nan for GADFL

        # GADFL state variables (set by _compute_gadfl_weights)
        self.gadfl_weights    = {}   # {client_id: float} aggregation weights
        self.gadfl_beta       = np.nan  # beta value used (for logging)
        self.gadfl_histogram  = None   # local histogram (after DP if enabled)
        self.gadfl_jsd_matrix = {}   # {(i,j): JSD value}

        # FedDkw state variables (set by _compute_feddkw_weights)
        self.feddkw_weights   = {}   # {client_id: float} aggregation weights
        self.feddkw_beta      = np.nan  # beta value used (for logging)

        # Hybrid-GADFL state variables (set by _compute_hybrid_weights)
        self.hybrid_weights   = {}    # {client_id: float} aggregation weights
        self.hybrid_beta1     = np.nan  # JSD beta (for logging)
        self.hybrid_beta2     = np.nan  # KL beta (for logging)
        self.hybrid_histogram = None    # local histogram (after DP)

        # SR-GADFL state variables (set by _compute_sr_gadfl_weights)
        self.sr_weights    = {}
        self.sr_beta       = np.nan
        self.sr_histogram  = None
        self.sr_jsd_matrix = {}

        # G-FedDkw state variables
        self.gfeddkw_weights   = {}
        self.gfeddkw_beta      = np.nan
        self.gfeddkw_histogram = None

        # T-G-FedDkw state variables
        self.tg_weights       = {}
        self.tg_beta          = np.nan
        self.tg_histogram     = None
        self.tg_target_hist   = None

        # SD-GADFL state variables
        self.sd_weights = {}
        self.sd_my_std  = np.nan

        logger.info(
            f"[Client {self.CLIENT_ID}] initialized ({self.ALGORITHM.upper()})"
        )

    def _setup_directories(self):
        """
        Create algorithm-specific directory structure AND the shared
        initial-weights directory used for incremental client scaling.

        Structure:
        ├── initial_weights/        ← shared across all algorithms & scaling steps
        ├── fedavg/
        │   ├── models/
        │   ├── results/
        │   ├── logs/
        │   ├── training_times/
        │   └── sync_decisions/
        ├── fedprox/  (same sub-dirs)
        ├── sarsa/
        │   ├── models/
        │   ├── results/
        │   ├── logs/
        │   ├── training_times/
        │   ├── sync_decisions/
        │   └── sarsa_qtables/
        └── gadfl/               ← NEW
            ├── models/
            ├── results/
            ├── logs/
            ├── training_times/
            └── sync_decisions/
        """
        # For GADFL: each epsilon value gets its own directory so runs
        # do not overwrite each other.
        #   gadfl_ep_null/   (no DP, dp_epsilon=null)
        #   gadfl_ep_0.5/    (dp_epsilon=0.5)
        #   gadfl_ep_1.0/    (dp_epsilon=1.0)
        #   gadfl_ep_2.0/    (dp_epsilon=2.0)
        #   gadfl_ep_5.0/    (dp_epsilon=5.0)
        # All other algorithms: directory = algorithm name (unchanged)
        if self.ALGORITHM == "gadfl":
            # Histogram DP tag
            if self.GADFL_DP_EPSILON is None:
                ep_tag = "null"
            else:
                ep_tag = f"{self.GADFL_DP_EPSILON:.1f}"

            # Model DP tag
            if self.GADFL_MODEL_SIGMA is None:
                sig_tag = "null"
            else:
                sig_tag = f"{self.GADFL_MODEL_SIGMA}"

            # Folder: gadfl_ep_0.5_sig_0.01
            algo_dir = f"gadfl_ep_{ep_tag}_sig_{sig_tag}"

        elif self.ALGORITHM == "feddkw":
            algo_dir = "feddkw"

        elif self.ALGORITHM == "hybrid_gadfl":
            if self.HYBRID_DP_EPSILON is None:
                ep_tag = "null"
            else:
                ep_tag = f"{self.HYBRID_DP_EPSILON:.1f}"
            if self.HYBRID_MODEL_SIGMA is None:
                sig_tag = "null"
            else:
                sig_tag = f"{self.HYBRID_MODEL_SIGMA}"
            algo_dir = f"hybrid_gadfl_ep_{ep_tag}_sig_{sig_tag}"

        elif self.ALGORITHM == "sr_gadfl":
            # SR-GADFL: alpha + DP options
            # Folder: sr_gadfl_a0.5_ep_null_sig_null
            alpha_tag = f"{self.SR_ALPHA}"
            if self.SR_DP_EPSILON is None:
                ep_tag = "null"
            else:
                ep_tag = f"{self.SR_DP_EPSILON:.1f}"
            if self.SR_MODEL_SIGMA is None:
                sig_tag = "null"
            else:
                sig_tag = f"{self.SR_MODEL_SIGMA}"
            algo_dir = f"sr_gadfl_a{alpha_tag}_ep_{ep_tag}_sig_{sig_tag}"

        elif self.ALGORITHM == "g_feddkw":
            algo_dir = "g_feddkw"
        elif self.ALGORITHM == "t_g_feddkw":
            algo_dir = "t_g_feddkw"
        elif self.ALGORITHM == "sd_gadfl":
            if self.SD_DP_EPSILON is None:
                ep_tag = "null"
            else:
                ep_tag = f"{self.SD_DP_EPSILON:.1f}"
            if self.SD_MODEL_SIGMA is None:
                sig_tag = "null"
            else:
                sig_tag = f"{self.SD_MODEL_SIGMA}"
            algo_dir = f"sd_gadfl_ep_{ep_tag}_sig_{sig_tag}"
        else:
            algo_dir = self.ALGORITHM

        self.models_dir         = os.path.join(algo_dir, "models")
        self.results_dir        = os.path.join(algo_dir, "results")
        self.logs_dir           = os.path.join(algo_dir, "logs")
        self.training_times_dir = os.path.join(algo_dir, "training_times")
        self.sync_decisions_dir = os.path.join(algo_dir, "sync_decisions")

        for dir_path in [
            self.models_dir,
            self.results_dir,
            self.logs_dir,
            self.training_times_dir,
            self.sync_decisions_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        if self.ALGORITHM == "sarsa":
            self.sarsa_qtables_dir = os.path.join(algo_dir, "sarsa_qtables")
            os.makedirs(self.sarsa_qtables_dir, exist_ok=True)

        os.makedirs(self.initial_weights_dir, exist_ok=True)

    def load_data(self):
        logger.info(f"[Client {self.CLIENT_ID}] Loading data...")

        data_path = self.peer_config["dataset_path"]
        df = pd.read_csv(data_path)

        if self.TARGET not in df.columns:
            raise ValueError(
                f"Target column '{self.TARGET}' missing in {data_path}"
            )

        X_np = df.drop(columns=[self.TARGET]).values
        y_np = df[self.TARGET].values.reshape(-1, 1)

        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.VAL_SPLIT,
            random_state=self.SEED,
            shuffle=True,
        )

        self.X_train   = X_train
        self.y_train   = y_train
        self.X_val     = X_val
        self.y_val     = y_val
        self.input_dim = X_train.shape[1]

        already_saved = list_saved_initial_clients(self.initial_weights_dir)
        logger.info(
            f"[Client {self.CLIENT_ID}] Initial weights directory: "
            f"'{self.initial_weights_dir}' — "
            f"clients with saved weights: {already_saved}"
        )

        self.model = load_or_init_weights(
            client_id=self.CLIENT_ID,
            input_dim=self.input_dim,
            init_dir=self.initial_weights_dir,
            device=self.device,
        )

        torch.save(
            self.model.state_dict(),
            os.path.join(self.models_dir, f"round_0_client{self.CLIENT_ID}.pt")
        )

        logger.info(
            f"[Client {self.CLIENT_ID}] Data loaded: "
            f"{len(X_train)} train, {len(X_val)} val samples"
        )

    async def _compute_gadfl_weights(self):
        """
        GADFL: Compute geography-aware aggregation weights before training begins.

        Steps (one-time, before round 1):
          1. Compute speed histogram from local y_train
          2. Optionally apply differential privacy noise
          3. Share histogram with all peers
          4. Wait until all peers have shared their histograms
          5. Compute pairwise JSD matrix
          6. Compute adaptive beta (or use fixed beta from config)
          7. Compute aggregation weights for this client
          8. Save histogram, JSD matrix, weights to CSV

        After this method, self.gadfl_weights is ready for use in aggregation.
        """
        if self.ALGORITHM != "gadfl":
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Computing geography-aware weights..."
        )

        # ── STEP 1: Compute local speed histogram ──────────────────────────
        local_speeds = self.y_train.numpy().flatten()
        histogram    = compute_speed_histogram(local_speeds, num_bins=self.GADFL_NUM_BINS)

        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Local histogram computed: "
            f"{[f'{v:.4f}' for v in histogram]}"
        )

        # ── STEP 2: Apply differential privacy (optional) ──────────────────
        if self.GADFL_DP_EPSILON is not None:
            histogram_to_share = apply_differential_privacy(
                histogram, self.GADFL_DP_EPSILON
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [GADFL] DP applied "
                f"(epsilon={self.GADFL_DP_EPSILON:.3f})"
            )
        else:
            histogram_to_share = histogram.copy()
            logger.info(
                f"[Client {self.CLIENT_ID}] [GADFL] No DP (dp_epsilon=null)"
            )

        # Save local histogram (before sharing) for reproducibility
        self.gadfl_histogram = histogram_to_share

        # ── STEP 3: Share histogram with all peers ──────────────────────────
        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Sharing histogram with peers..."
        )
        ok = await self.client.share_histogram(histogram_to_share)
        if not ok:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [GADFL] Some peers did not acknowledge "
                f"histogram — continuing with available data"
            )

        # ── STEP 4: Poll until all histograms received ──────────────────────
        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Waiting for all peer histograms..."
        )
        all_histograms = await self.client.get_all_histograms(timeout=60.0)

        if not all_histograms:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [GADFL] Could not collect all histograms. "
                f"Falling back to uniform FedAvg weights."
            )
            n = len(self.CLIENT_IDS)
            self.gadfl_weights = {cid: 1.0 / n for cid in self.CLIENT_IDS}
            self.gadfl_beta    = 0.0
            return

        # Convert numpy arrays to plain numpy (from JSON lists)
        all_histograms_np = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in all_histograms.items()
        }

        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Received histograms from "
            f"{len(all_histograms_np)} peers"
        )

        # ── STEP 5: Compute pairwise JSD matrix ────────────────────────────
        self.gadfl_jsd_matrix = compute_jsd_matrix(all_histograms_np)

        # Log JSD distances from this client to all others
        for other_id in sorted(self.CLIENT_IDS):
            if other_id != self.CLIENT_ID:
                jsd_val = self.gadfl_jsd_matrix.get((self.CLIENT_ID, other_id), 0.0)
                logger.info(
                    f"[Client {self.CLIENT_ID}] [GADFL] "
                    f"JSD({self.CLIENT_ID}, {other_id}) = {jsd_val:.4f}"
                )

        # ── STEP 6: Compute beta ────────────────────────────────────────────
        if self.GADFL_BETA_MODE == "adaptive":
            self.gadfl_beta = compute_adaptive_beta(
                self.gadfl_jsd_matrix, self.CLIENT_IDS
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [GADFL] Adaptive beta = {self.gadfl_beta:.4f}"
            )
        else:
            self.gadfl_beta = float(self.GADFL_BETA_FIXED)
            logger.info(
                f"[Client {self.CLIENT_ID}] [GADFL] Fixed beta = {self.gadfl_beta:.4f}"
            )

        # ── STEP 7: Compute aggregation weights ────────────────────────────
        self.gadfl_weights = compute_aggregation_weights(
            my_id      = self.CLIENT_ID,
            jsd_matrix = self.gadfl_jsd_matrix,
            all_agents = self.CLIENT_IDS,
            beta       = self.gadfl_beta,
        )

        # Log weights
        for cid, w in sorted(self.gadfl_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [GADFL] "
                f"Weight for Client {cid}: {w:.4f}"
            )

        # ── STEP 8: Save GADFL-specific CSVs ───────────────────────────────
        self._save_gadfl_csvs(all_histograms_np)

    def _save_gadfl_csvs(self, all_histograms_np: dict):
        """
        Save GADFL-specific CSV files (one-time, called from _compute_gadfl_weights).

        Files saved:
          1. client{id}_histogram_gadfl.csv          — local histogram per bin
          2. client{id}_jsd_matrix_gadfl.csv         — pairwise JSD matrix
          3. client{id}_aggregation_weights_gadfl.csv — final aggregation weights
        """
        # 1. Histogram CSV
        hist_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_histogram_gadfl.csv"
        )
        with open(hist_file, "w", encoding="utf-8") as f:
            num_bins = len(self.gadfl_histogram)
            bin_size = 90.0 / num_bins
            f.write("bin_index,bin_start_kmh,bin_end_kmh,probability\n")
            for i, prob in enumerate(self.gadfl_histogram):
                f.write(
                    f"{i},{i * bin_size:.1f},{(i + 1) * bin_size:.1f},{prob:.8f}\n"
                )
        logger.info(f"[Client {self.CLIENT_ID}] [GADFL] Histogram saved → {hist_file}")

        # 2. JSD Matrix CSV
        jsd_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_jsd_matrix_gadfl.csv"
        )
        all_ids = sorted(self.CLIENT_IDS)
        with open(jsd_file, "w", encoding="utf-8") as f:
            header = "client_i,client_j,jsd_value\n"
            f.write(header)
            for i in all_ids:
                for j in all_ids:
                    jsd_val = self.gadfl_jsd_matrix.get((i, j), 0.0)
                    f.write(f"{i},{j},{jsd_val:.8f}\n")
        logger.info(f"[Client {self.CLIENT_ID}] [GADFL] JSD matrix saved → {jsd_file}")

        # 3. Aggregation Weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_gadfl.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# GADFL aggregation weights for Client {self.CLIENT_ID}\n"
                f"# beta={self.gadfl_beta:.6f} (mode={self.GADFL_BETA_MODE})\n"
                f"# dp_epsilon={self.GADFL_DP_EPSILON}\n"
                f"# These weights are fixed for all {self.TOTAL_ROUNDS} training rounds\n"
            )
            f.write("peer_id,jsd_distance,aggregation_weight\n")
            for cid in all_ids:
                jsd_val = self.gadfl_jsd_matrix.get((self.CLIENT_ID, cid), 0.0)
                weight  = self.gadfl_weights.get(cid, 0.0)
                f.write(f"{cid},{jsd_val:.8f},{weight:.8f}\n")
        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] Aggregation weights saved → {weights_file}"
        )

        # 4. Also save all peer histograms for reference
        all_hist_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_all_histograms_gadfl.csv"
        )
        num_bins = self.GADFL_NUM_BINS
        bin_size = 90.0 / num_bins
        with open(all_hist_file, "w", encoding="utf-8") as f:
            bin_headers = ",".join(
                [f"bin_{i}_{i*bin_size:.0f}_{(i+1)*bin_size:.0f}kmh"
                 for i in range(num_bins)]
            )
            f.write(f"peer_id,{bin_headers}\n")
            for peer_id in sorted(all_histograms_np.keys()):
                hist = all_histograms_np[peer_id]
                bins_str = ",".join([f"{v:.8f}" for v in hist])
                f.write(f"{peer_id},{bins_str}\n")
        logger.info(
            f"[Client {self.CLIENT_ID}] [GADFL] All histograms saved → {all_hist_file}"
        )






    async def _compute_t_g_feddkw_weights(self):
        """
        T-G-FedDkw: Target-aware geography weighted aggregation.

        Oracle upper bound — uses target environment histogram
        (test_set_cleaned.csv) instead of training average.

        Peers whose distribution matches the deployment environment
        (Town07) get higher aggregation weight.
        """
        if self.ALGORITHM != "t_g_feddkw":
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
            f"Loading target histogram from {self.TG_TARGET_DATA_PATH}..."
        )

        # Step 1: Load target environment histogram (Town07 test set)
        try:
            import pandas as pd
            target_df    = pd.read_csv(self.TG_TARGET_DATA_PATH)
            target_speeds = target_df[self.TG_TARGET_SPEED_COL].values
            self.tg_target_hist = compute_speed_histogram(
                target_speeds, num_bins=self.TG_NUM_BINS
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
                f"Target histogram loaded ({len(target_speeds)} samples)"
            )
        except Exception as e:
            logger.error(
                f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
                f"Failed to load target histogram: {e}. Falling back to uniform."
            )
            n = len(self.CLIENT_IDS)
            self.tg_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            self.tg_beta    = 0.0
            return

        # Step 2: Compute local histogram + optional DP
        local_speeds = self.y_train.numpy().flatten()
        histogram    = compute_speed_histogram(local_speeds, num_bins=self.TG_NUM_BINS)

        if self.TG_DP_EPSILON is not None:
            histogram_to_share = apply_differential_privacy(
                histogram, self.TG_DP_EPSILON
            )
        else:
            histogram_to_share = histogram.copy()

        self.tg_histogram = histogram_to_share

        # Step 3: Share histogram with peers
        await self.client.share_histogram(histogram_to_share)

        # Step 4: Receive all histograms
        all_histograms = await self.client.get_all_histograms(timeout=60.0)
        if not all_histograms:
            n = len(self.CLIENT_IDS)
            self.tg_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            self.tg_beta    = 0.0
            return

        all_histograms_np = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in all_histograms.items()
        }

        # Step 5: Compute beta using target histogram
        self.tg_beta = compute_t_g_feddkw_beta(
            all_histograms_np, self.CLIENT_IDS, self.tg_target_hist
        )
        logger.info(
            f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
            f"beta={self.tg_beta:.4f} (1/mean_JSD_from_target)"
        )

        # Step 6: Compute weights using target histogram
        self.tg_weights = compute_t_g_feddkw_weights(
            my_id       = self.CLIENT_ID,
            histograms  = all_histograms_np,
            all_agents  = self.CLIENT_IDS,
            target_hist = self.tg_target_hist,
            beta        = self.tg_beta,
        )

        for cid, w in sorted(self.tg_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
                f"Client {cid}: {w:.4f}"
            )

        # Step 7: Save weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_t_g_feddkw.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# T-G-FedDkw weights for Client {self.CLIENT_ID}\n"
                f"# beta={self.tg_beta:.6f} (1/mean_JSD_from_target)\n"
                f"# target={self.TG_TARGET_DATA_PATH}\n"
                f"# Oracle: uses test environment histogram for weighting\n"
            )
            f.write("peer_id,jsd_from_target,aggregation_weight\n")
            for cid in sorted(self.CLIENT_IDS):
                jsd_val = compute_jsd(all_histograms_np[cid], self.tg_target_hist)
                weight  = self.tg_weights.get(cid, 0.0)
                f.write(f"{cid},{jsd_val:.8f},{weight:.8f}\n")

        logger.info(
            f"[Client {self.CLIENT_ID}] [T-G-FedDkw] Weights saved → {weights_file}"
        )


    async def _compute_sd_gadfl_weights(self):
        """
        SD-GADFL: Standard-deviation proportional aggregation weights.

        Each agent shares ONE scalar (its local target/speed std)
        instead of a 10-bin histogram — simpler and more private
        (a single summary statistic vs. a full distribution shape).

        weight_j = std_j / sum(std_i)

        Higher local data diversity -> higher aggregation weight.
        Same global weight vector used by every agent.

        Steps (one-time, before round 1):
          1. Compute local std of y_train (Dirichlet-partitioned data)
          2. Optional: Laplace DP noise on the scalar
          3. Share std (as a 1-element "histogram") with all peers
          4. Receive all peers' std values
          5. Compute weight_j = std_j / sum(std_i)
          6. Save to CSV
        """
        if self.ALGORITHM != "sd_gadfl":
            return

        logger.info(f"[Client {self.CLIENT_ID}] [SD-GADFL] Computing std-based weights...")

        # Step 1: Local std (from Dirichlet-partitioned training data)
        local_speeds = self.y_train.numpy().flatten()
        local_std    = float(np.std(local_speeds))
        self.sd_my_std = local_std
        logger.info(f"[Client {self.CLIENT_ID}] [SD-GADFL] Local std = {local_std:.4f}")

        # Step 2: Optional DP noise on the scalar
        if self.SD_DP_EPSILON is not None:
            std_to_share = apply_dp_to_scalar(
                local_std, self.SD_DP_EPSILON, sensitivity=self.SD_SENSITIVITY
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [SD-GADFL] "
                f"DP applied to std: {local_std:.4f} -> {std_to_share:.4f} "
                f"(epsilon={self.SD_DP_EPSILON})"
            )
        else:
            std_to_share = local_std

        # Step 3: Share std as a 1-element list (reuses histogram channel)
        ok = await self.client.share_histogram([std_to_share])
        if not ok:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [SD-GADFL] "
                f"Some peers did not acknowledge std value"
            )

        # Step 4: Receive all peers' std values
        all_stds_raw = await self.client.get_all_histograms(timeout=60.0)
        if not all_stds_raw:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [SD-GADFL] "
                f"Could not collect std values. Falling back to uniform."
            )
            n = len(self.CLIENT_IDS)
            self.sd_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            return

        all_stds = {
            int(k): float(np.array(v, dtype=np.float64).flatten()[0])
            for k, v in all_stds_raw.items()
        }

        # Step 5: Compute weights
        self.sd_weights = compute_std_based_weights(all_stds, self.CLIENT_IDS)

        for cid, w in sorted(self.sd_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [SD-GADFL] "
                f"Client {cid}: std={all_stds[cid]:.4f} -> weight={w:.4f}"
            )

        # Step 6: Save weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_sd_gadfl.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# SD-GADFL weights for Client {self.CLIENT_ID}\n"
                f"# weight_j = std_j / sum(std_i)\n"
                f"# dp_epsilon={self.SD_DP_EPSILON}\n"
                f"# model_sigma={self.SD_MODEL_SIGMA}\n"
                f"# These weights are fixed for all training rounds\n"
            )
            f.write("peer_id,std,aggregation_weight\n")
            for cid in sorted(self.CLIENT_IDS):
                f.write(f"{cid},{all_stds[cid]:.8f},{self.sd_weights[cid]:.8f}\n")

        logger.info(f"[Client {self.CLIENT_ID}] [SD-GADFL] Weights saved -> {weights_file}")

    async def _compute_g_feddkw_weights(self):
        """
        G-FedDkw: Symmetric JSD-based global weighted aggregation.
        Replaces KL in FedDkw with symmetric JSD for cleaner math.
        """
        if self.ALGORITHM != "g_feddkw":
            return

        logger.info(f"[Client {self.CLIENT_ID}] [G-FedDkw] Computing JSD-global weights...")

        # Histogram source: RAW town data (geographic reality) vs
        # Dirichlet-partitioned training data (distorted distributions)
        #
        # Rationale: Dirichlet partitioning concentrates each client's
        # data into 1-2 speed bins, distorting the apparent geographic
        # profile (e.g. Town03/highway appears as a slow-speed node).
        # Using the RAW (full town) histogram for AGGREGATION WEIGHTS
        # while still TRAINING on the Dirichlet-partitioned data
        # preserves realistic non-IID FL conditions while letting
        # the geography-aware weighting reflect true environmental
        # characteristics.
        if self.GFEDDKW_RAW_HIST_PATH:
            try:
                import pandas as pd
                raw_df     = pd.read_csv(self.GFEDDKW_RAW_HIST_PATH)
                raw_speeds = raw_df[self.GFEDDKW_RAW_SPEED_COL].values
                histogram  = compute_speed_histogram(raw_speeds, num_bins=self.GFEDDKW_NUM_BINS)
                logger.info(
                    f"[Client {self.CLIENT_ID}] [G-FedDkw] "
                    f"Histogram from RAW town data: {self.GFEDDKW_RAW_HIST_PATH} "
                    f"({len(raw_speeds)} samples)"
                )
            except Exception as e:
                logger.warning(
                    f"[Client {self.CLIENT_ID}] [G-FedDkw] "
                    f"Failed to load raw histogram ({e}). "
                    f"Falling back to Dirichlet-partitioned local data."
                )
                local_speeds = self.y_train.numpy().flatten()
                histogram    = compute_speed_histogram(local_speeds, num_bins=self.GFEDDKW_NUM_BINS)
        else:
            local_speeds = self.y_train.numpy().flatten()
            histogram    = compute_speed_histogram(local_speeds, num_bins=self.GFEDDKW_NUM_BINS)
            logger.info(
                f"[Client {self.CLIENT_ID}] [G-FedDkw] "
                f"Histogram from Dirichlet-partitioned training data"
            )

        if self.GFEDDKW_DP_EPSILON is not None:
            histogram_to_share = apply_differential_privacy(histogram, self.GFEDDKW_DP_EPSILON)
        else:
            histogram_to_share = histogram.copy()

        self.gfeddkw_histogram = histogram_to_share
        await self.client.share_histogram(histogram_to_share)

        all_histograms = await self.client.get_all_histograms(timeout=60.0)
        if not all_histograms:
            n = len(self.CLIENT_IDS)
            self.gfeddkw_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            self.gfeddkw_beta    = 0.0
            return

        all_histograms_np = {int(k): np.array(v, dtype=np.float64) for k, v in all_histograms.items()}

        self.gfeddkw_beta = compute_g_feddkw_beta(all_histograms_np, self.CLIENT_IDS)
        self.gfeddkw_weights = compute_g_feddkw_weights(
            self.CLIENT_ID, all_histograms_np, self.CLIENT_IDS, self.gfeddkw_beta
        )

        for cid, w in sorted(self.gfeddkw_weights.items()):
            logger.info(f"[Client {self.CLIENT_ID}] [G-FedDkw] Client {cid}: {w:.4f}")

        # Save weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_g_feddkw.csv"
        )
        all_hists   = np.array([all_histograms_np[j] for j in sorted(self.CLIENT_IDS)])
        global_hist = all_hists.mean(axis=0)
        global_hist = np.clip(global_hist, 1e-10, None)
        global_hist = global_hist / global_hist.sum()

        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# G-FedDkw weights for Client {self.CLIENT_ID}\n"
                f"# beta={self.gfeddkw_beta:.6f} (1/mean_JSD_from_global)\n"
                f"# Uses symmetric JSD instead of asymmetric KL\n"
            )
            f.write("peer_id,jsd_from_global,aggregation_weight\n")
            for cid in sorted(self.CLIENT_IDS):
                jsd_val = compute_jsd(all_histograms_np[cid], global_hist)
                weight  = self.gfeddkw_weights.get(cid, 0.0)
                f.write(f"{cid},{jsd_val:.8f},{weight:.8f}\n")
        logger.info(f"[Client {self.CLIENT_ID}] [G-FedDkw] Weights saved → {weights_file}")

    async def _compute_sr_gadfl_weights(self):
        """
        SR-GADFL: Self-Regularized geography-aware aggregation weights.

        Fixes GADFL's self-weight dominance problem:
          Standard GADFL: JSD(self,self)=0 → self gets max weight (31.7%)
          SR-GADFL:       self-distance = alpha × mean_peer_JSD
                          → self gets realistic weight (~12-20%)

        Result: more collaborative aggregation → better cross-environment
        generalization while preserving geographic similarity structure.

        Steps (one-time, before round 1):
          1. Compute local speed histogram
          2. Optional DP noise
          3. Share histogram with peers
          4. Compute JSD matrix
          5. Compute SR-GADFL weights with self-regularization
          6. Save to CSV
        """
        if self.ALGORITHM != "sr_gadfl":
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] [SR-GADFL] "
            f"Computing self-regularized weights (alpha={self.SR_ALPHA})..."
        )

        # Step 1: Histogram
        local_speeds = self.y_train.numpy().flatten()
        histogram    = compute_speed_histogram(local_speeds, num_bins=self.SR_NUM_BINS)

        # Step 2: Histogram DP (optional)
        if self.SR_DP_EPSILON is not None:
            histogram_to_share = apply_differential_privacy(
                histogram, self.SR_DP_EPSILON
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                f"Histogram DP applied (epsilon={self.SR_DP_EPSILON})"
            )
        else:
            histogram_to_share = histogram.copy()

        self.sr_histogram = histogram_to_share

        # Step 3: Share histogram
        ok = await self.client.share_histogram(histogram_to_share)
        if not ok:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                f"Some peers did not acknowledge histogram"
            )

        # Step 4: Receive all histograms
        logger.info(
            f"[Client {self.CLIENT_ID}] [SR-GADFL] "
            f"Waiting for all peer histograms..."
        )
        all_histograms = await self.client.get_all_histograms(timeout=60.0)

        if not all_histograms:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                f"Could not collect histograms. Falling back to uniform."
            )
            n = len(self.CLIENT_IDS)
            self.sr_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            self.sr_beta    = 0.0
            return

        all_histograms_np = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in all_histograms.items()
        }

        # Step 5a: Compute JSD matrix
        self.sr_jsd_matrix = compute_jsd_matrix(all_histograms_np)

        # Step 5b: Compute adaptive beta (same as GADFL)
        self.sr_beta = compute_sr_gadfl_beta(self.sr_jsd_matrix, self.CLIENT_IDS)
        logger.info(
            f"[Client {self.CLIENT_ID}] [SR-GADFL] "
            f"beta={self.sr_beta:.4f}, alpha={self.SR_ALPHA}"
        )

        # Step 5c: Compute SR weights
        self.sr_weights = compute_sr_gadfl_weights(
            my_id      = self.CLIENT_ID,
            jsd_matrix = self.sr_jsd_matrix,
            all_agents = self.CLIENT_IDS,
            beta       = self.sr_beta,
            alpha      = self.SR_ALPHA,
        )

        for cid, w in sorted(self.sr_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                f"Weight for Client {cid}: {w:.4f} "
                f"(GADFL self was 31.7%, now {self.sr_weights[self.CLIENT_ID]*100:.1f}%)"
                if cid == self.CLIENT_ID else
                f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                f"Weight for Client {cid}: {w:.4f}"
            )

        # Step 6: Save CSV
        self._save_sr_gadfl_csvs()

    def _save_sr_gadfl_csvs(self):
        """Save SR-GADFL specific CSV files (one-time)."""
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_sr_gadfl.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# SR-GADFL weights for Client {self.CLIENT_ID}\n"
                f"# beta={self.sr_beta:.6f} (adaptive=1/mean_JSD)\n"
                f"# alpha={self.SR_ALPHA} (self-regularization strength)\n"
                f"# dp_epsilon={self.SR_DP_EPSILON}\n"
                f"# model_sigma={self.SR_MODEL_SIGMA}\n"
                f"# Self-distance = alpha x mean_peer_JSD\n"
            )
            f.write("peer_id,jsd_distance,sr_distance,aggregation_weight\n")

            peer_jsds = [
                self.sr_jsd_matrix.get((self.CLIENT_ID, j), 0.0)
                for j in sorted(self.CLIENT_IDS)
                if j != self.CLIENT_ID
            ]
            mean_peer_jsd = float(np.mean(peer_jsds)) if peer_jsds else 0.0

            for cid in sorted(self.CLIENT_IDS):
                jsd_val = self.sr_jsd_matrix.get((self.CLIENT_ID, cid), 0.0)
                sr_dist = self.SR_ALPHA * mean_peer_jsd if cid == self.CLIENT_ID else jsd_val
                weight  = self.sr_weights.get(cid, 0.0)
                f.write(f"{cid},{jsd_val:.8f},{sr_dist:.8f},{weight:.8f}\n")

        logger.info(
            f"[Client {self.CLIENT_ID}] [SR-GADFL] "
            f"Weights saved → {weights_file}"
        )

    async def _compute_hybrid_weights(self):
        """
        Hybrid-GADFL: Compute combined JSD + KL aggregation weights.

        Combines GADFL (pairwise JSD) and FedDkw (KL vs global):
          score_j = -beta1 × JSD(own_hist, peer_hist)
                    -beta2 × KL(peer_hist || global_hist)
          weight_j = softmax(score_j)

        Both betas adaptive:
          beta1 = 1/mean_JSD (same as GADFL)
          beta2 = 1/mean_KL  (same as FedDkw)

        DP same as GADFL:
          Histogram DP: optional Laplace noise (HYBRID_DP_EPSILON)
          Model DP:     optional Gaussian noise (HYBRID_MODEL_SIGMA)
        """
        if self.ALGORITHM != "hybrid_gadfl":
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
            f"Computing JSD+KL combined weights..."
        )

        # Step 1: Compute local histogram
        local_speeds = self.y_train.numpy().flatten()
        histogram    = compute_speed_histogram(
            local_speeds, num_bins=self.HYBRID_NUM_BINS
        )

        # Step 2: Apply histogram DP (optional)
        if self.HYBRID_DP_EPSILON is not None:
            histogram_to_share = apply_differential_privacy(
                histogram, self.HYBRID_DP_EPSILON
            )
            logger.info(
                f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
                f"Histogram DP applied (epsilon={self.HYBRID_DP_EPSILON})"
            )
        else:
            histogram_to_share = histogram.copy()

        self.hybrid_histogram = histogram_to_share

        # Step 3: Share histogram with peers
        ok = await self.client.share_histogram(histogram_to_share)
        if not ok:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
                f"Some peers did not acknowledge histogram"
            )

        # Step 4: Receive all histograms
        logger.info(
            f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
            f"Waiting for all peer histograms..."
        )
        all_histograms = await self.client.get_all_histograms(timeout=60.0)

        if not all_histograms:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
                f"Could not collect histograms. Falling back to uniform."
            )
            n = len(self.CLIENT_IDS)
            self.hybrid_weights = {cid: 1.0/n for cid in self.CLIENT_IDS}
            self.hybrid_beta1   = 0.0
            self.hybrid_beta2   = 0.0
            return

        all_histograms_np = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in all_histograms.items()
        }

        # Step 5: Compute adaptive betas
        self.hybrid_beta1, self.hybrid_beta2 = compute_hybrid_betas(
            all_histograms_np, self.CLIENT_IDS
        )
        logger.info(
            f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
            f"beta1(JSD)={self.hybrid_beta1:.4f} "
            f"beta2(KL)={self.hybrid_beta2:.4f}"
        )

        # Step 6: Compute weights
        self.hybrid_weights = compute_hybrid_weights(
            my_id      = self.CLIENT_ID,
            histograms = all_histograms_np,
            all_agents = self.CLIENT_IDS,
            beta1      = self.hybrid_beta1,
            beta2      = self.hybrid_beta2,
        )

        for cid, w in sorted(self.hybrid_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
                f"Weight for Client {cid}: {w:.4f}"
            )

        # Step 7: Save CSVs
        self._save_hybrid_csvs(all_histograms_np)

    def _save_hybrid_csvs(self, all_histograms_np: dict):
        """Save Hybrid-GADFL specific CSV files (one-time)."""

        # Global histogram for KL reference
        all_hists   = np.array(
            [all_histograms_np[j] for j in sorted(self.CLIENT_IDS)],
            dtype=np.float64
        )
        global_hist = all_hists.mean(axis=0)
        global_hist = np.clip(global_hist, 1e-10, None)
        global_hist = global_hist / global_hist.sum()

        # Weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_hybrid.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# Hybrid-GADFL weights for Client {self.CLIENT_ID}\n"
                f"# beta1(JSD)={self.hybrid_beta1:.6f}\n"
                f"# beta2(KL)={self.hybrid_beta2:.6f}\n"
                f"# dp_epsilon={self.HYBRID_DP_EPSILON}\n"
                f"# model_sigma={self.HYBRID_MODEL_SIGMA}\n"
            )
            f.write("peer_id,jsd_distance,kl_distance,combined_score,weight\n")
            for cid in sorted(self.CLIENT_IDS):
                jsd_val = compute_jsd(
                    all_histograms_np[self.CLIENT_ID],
                    all_histograms_np[cid]
                )
                kl_val = compute_kl_divergence(
                    all_histograms_np[cid], global_hist
                )
                score  = (-self.hybrid_beta1 * jsd_val
                         - self.hybrid_beta2 * kl_val)
                weight = self.hybrid_weights.get(cid, 0.0)
                f.write(
                    f"{cid},{jsd_val:.8f},{kl_val:.8f},"
                    f"{score:.8f},{weight:.8f}\n"
                )
        logger.info(
            f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
            f"Weights saved → {weights_file}"
        )

    async def _compute_feddkw_weights(self):
        """
        FedDkw: Compute KL-divergence based aggregation weights.

        P2P adaptation of FedDkw (Federated Learning with
        Data Distribution Knowledge):
          Original: central server computes KL(client || global)
          Ours:     each agent computes locally from shared histograms

        Steps (one-time, before round 1):
          1. Compute local speed histogram from y_train
          2. Share histogram with all peers (same as GADFL)
          3. Receive all peer histograms
          4. Compute global_hist = mean of all histograms
          5. Compute KL(peer_hist || global_hist) for each peer
          6. Compute adaptive beta = 1 / mean_KL
          7. Weights = softmax(-beta × KL)
          8. Save to CSV

        Difference from GADFL:
          GADFL:  JSD(own, peer)          — pairwise distance
          FedDkw: KL(peer || global_mean) — each peer vs global
        """
        if self.ALGORITHM != "feddkw":
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] [FedDkw] Computing KL-based weights..."
        )

        # Step 1: Compute local histogram
        local_speeds = self.y_train.numpy().flatten()
        histogram    = compute_speed_histogram(
            local_speeds, num_bins=self.FEDDKW_NUM_BINS
        )
        logger.info(
            f"[Client {self.CLIENT_ID}] [FedDkw] Histogram computed"
        )

        # Step 2: Share histogram with peers (reuse GADFL infrastructure)
        ok = await self.client.share_histogram(histogram)
        if not ok:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [FedDkw] "
                f"Some peers did not acknowledge histogram"
            )

        # Step 3: Receive all peer histograms
        logger.info(
            f"[Client {self.CLIENT_ID}] [FedDkw] Waiting for all histograms..."
        )
        all_histograms = await self.client.get_all_histograms(timeout=60.0)

        if not all_histograms:
            logger.warning(
                f"[Client {self.CLIENT_ID}] [FedDkw] "
                f"Could not collect all histograms. Falling back to uniform."
            )
            n = len(self.CLIENT_IDS)
            self.feddkw_weights = {cid: 1.0 / n for cid in self.CLIENT_IDS}
            self.feddkw_beta    = 0.0
            return

        all_histograms_np = {
            int(k): np.array(v, dtype=np.float64)
            for k, v in all_histograms.items()
        }

        # Step 4-5: Compute beta and weights using KL divergence
        if self.FEDDKW_BETA_MODE == "adaptive":
            self.feddkw_beta = compute_feddkw_beta(
                all_histograms_np, self.CLIENT_IDS
            )
        else:
            self.feddkw_beta = float(self.FEDDKW_BETA_FIXED)

        logger.info(
            f"[Client {self.CLIENT_ID}] [FedDkw] beta = {self.feddkw_beta:.4f}"
        )

        # Step 6: Compute weights
        self.feddkw_weights = compute_feddkw_weights(
            my_id      = self.CLIENT_ID,
            histograms = all_histograms_np,
            all_agents = self.CLIENT_IDS,
            beta       = self.feddkw_beta,
        )

        for cid, w in sorted(self.feddkw_weights.items()):
            logger.info(
                f"[Client {self.CLIENT_ID}] [FedDkw] "
                f"Weight for Client {cid}: {w:.4f}"
            )

        # Step 7: Save to CSV
        self._save_feddkw_csvs(all_histograms_np)

    def _save_feddkw_csvs(self, all_histograms_np: dict):
        """Save FedDkw-specific CSV files (one-time)."""

        # Global histogram
        all_hists   = np.array(
            [all_histograms_np[j] for j in sorted(self.CLIENT_IDS)],
            dtype=np.float64
        )
        global_hist = all_hists.mean(axis=0)

        # Weights CSV
        weights_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_aggregation_weights_feddkw.csv"
        )
        with open(weights_file, "w", encoding="utf-8") as f:
            f.write(
                f"# FedDkw aggregation weights for Client {self.CLIENT_ID}\n"
                f"# beta={self.feddkw_beta:.6f} (mode={self.FEDDKW_BETA_MODE})\n"
                f"# KL(peer_hist || global_hist) based weights\n"
            )
            f.write("peer_id,kl_distance,aggregation_weight\n")
            for cid in sorted(self.CLIENT_IDS):
                kl_val = compute_kl_divergence(
                    all_histograms_np[cid], global_hist
                )
                weight = self.feddkw_weights.get(cid, 0.0)
                f.write(f"{cid},{kl_val:.8f},{weight:.8f}\n")
        logger.info(
            f"[Client {self.CLIENT_ID}] [FedDkw] Weights saved → {weights_file}"
        )

    async def _compute_distribution_weight(self):
        """
        SARSA: Compute distribution_weight based on config mode.

        ORACLE mode: reads precomputed weight from peer_config.yaml.
        VARIANCE mode: computes from local std, shares with peers.

        This method is only called for SARSA algorithm.
        For GADFL, use _compute_gadfl_weights() instead.
        """
        if self.ALGORITHM != "sarsa":
            self.distribution_weight = 1.0
            return

        if self.DIST_WEIGHT_MODE == "oracle":
            w = self.ORACLE_WEIGHTS.get(self.CLIENT_ID, 0.5)
            self.distribution_weight = float(w)
            logger.info(
                f"[Client {self.CLIENT_ID}] Distribution weight (oracle): "
                f"{self.distribution_weight:.4f}"
            )

        else:  # variance mode (default)
            local_std = float(self.y_train.std())

            logger.info(
                f"[Client {self.CLIENT_ID}] Sharing data std={local_std:.4f} "
                f"with all peers..."
            )
            await self.client.share_data_stat(local_std)

            _timeout   = 60.0
            all_stats  = await self.client.get_all_data_stats(timeout=_timeout)

            if not all_stats:
                logger.warning(
                    f"[Client {self.CLIENT_ID}] Could not collect all data stats. "
                    f"Using default weight=0.5"
                )
                self.distribution_weight = 0.5
                return

            max_std = max(all_stats.values()) + 1e-8
            self.distribution_weight = local_std / max_std

            logger.info(
                f"[Client {self.CLIENT_ID}] Distribution weight (variance): "
                f"{self.distribution_weight:.4f} "
                f"(local_std={local_std:.4f}, max_std={max_std:.4f})"
            )

        # Save weight to results dir for reproducibility
        weight_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_distribution_weight_{self.ALGORITHM}.txt"
        )
        with open(weight_file, "w") as f:
            f.write(
                f"mode={self.DIST_WEIGHT_MODE}\n"
                f"weight={self.distribution_weight:.6f}\n"
            )

    def setup_log_files(self):
        """Setup CSV log files in algorithm-specific results directory."""
        metrics_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_metrics_{self.ALGORITHM}.csv"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("round,train_MSE,train_R2,test_MSE,test_RMSE,test_MAE,test_R2\n")

        waiting_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_waiting_{self.ALGORITHM}.csv"
        )
        with open(waiting_file, "w", encoding="utf-8") as f:
            # Column notes:
            # straggler_score : SARSA only (nan for other algorithms)
            # dist_weight     : SARSA only (nan for other algorithms)
            # beta            : SARSA: selected beta | GADFL: adaptive beta | others: nan
            # sync_prob       : SARSA: computed probability | others: 1.0
            # participated    : SARSA: 0 or 1 | others: always 1
            f.write(
                "round,training_time,waiting_time,effective_waiting_time,"
                "straggler_score,dist_weight,beta,sync_prob,participated,test_R2\n"
            )

        training_file = os.path.join(
            self.training_times_dir,
            f"client{self.CLIENT_ID}_times.csv"
        )
        with open(training_file, "w", encoding="utf-8") as f:
            f.write("round,training_time\n")

    def save_training_time(self, rnd: int, training_time: float):
        training_file = os.path.join(
            self.training_times_dir,
            f"client{self.CLIENT_ID}_times.csv"
        )
        with open(training_file, "a", encoding="utf-8") as f:
            f.write(f"{rnd},{training_time:.4f}\n")

    async def wait_for_round_completion(self, round_num: int):
        """
        Wait until the required peers report completed round >= round_num.

        FedAvg / FedProx / GADFL — full barrier:
            Every client must be ready. All clients always participate.

        SARSA — conditional barrier (CHANGE 1):
            Only the PARTICIPATING clients of round_num are required.
        """
        if round_num < 1:
            return

        logger.info(
            f"[Client {self.CLIENT_ID}] Waiting for round {round_num} "
            f"completion..."
        )

        if self.ALGORITHM != "sarsa":
            # ── FedAvg / FedProx / GADFL: full barrier ───────────────────────
            ready_peers = await self.client.wait_for_peers_ready(
                expected_round=round_num,
                timeout=self.BARRIER_TIMEOUT,
            )
            expected_ready = len(self.CLIENT_IDS)
            if len(ready_peers) != expected_ready:
                raise RuntimeError(
                    f"Round {round_num} sync failed: "
                    f"got {len(ready_peers)}/{expected_ready} ready peers"
                )

        else:
            # ── SARSA: conditional barrier ───────────────────────────────────
            status = await self.client.wait_for_round_status_complete(
                round_num,
                timeout=self.BARRIER_TIMEOUT,
            )
            if status is None:
                raise RuntimeError(
                    f"[Client {self.CLIENT_ID}] Timed out waiting for "
                    f"round {round_num} status"
                )

            participant_ids = await self.client.get_round_participants(round_num)
            participant_ids = sorted(set(int(x) for x in participant_ids))

            if not participant_ids:
                raise RuntimeError(
                    f"No participants recorded for round {round_num} "
                    f"in SARSA mode"
                )

            logger.info(
                f"[Client {self.CLIENT_ID}] SARSA barrier round {round_num}: "
                f"checking participants {participant_ids} only"
            )

            ready_peers = await self.client.wait_for_peers_ready(
                expected_round=round_num,
                timeout=self.BARRIER_TIMEOUT,
            )

            ready_ids = [int(p) for p in ready_peers]
            missing   = [cid for cid in participant_ids if cid not in ready_ids]
            if missing:
                raise RuntimeError(
                    f"Round {round_num} SARSA barrier failed: "
                    f"declared participants {missing} did not signal ready"
                )

            logger.info(
                f"[Client {self.CLIENT_ID}] SARSA barrier satisfied — "
                f"round {round_num}, participants: {participant_ids}"
            )

    async def get_expected_participants(self, round_num: int):
        """
        FedAvg / FedProx / GADFL: all clients participated.
        SARSA: ask transport layer for the actual participants of that round.
        """
        if self.ALGORITHM != "sarsa":
            return list(self.CLIENT_IDS)

        status = await self.client.wait_for_round_status_complete(
            round_num,
            timeout=self.BARRIER_TIMEOUT,
        )
        if status is None:
            raise RuntimeError(
                f"Timed out waiting for round-status completion for "
                f"round {round_num}"
            )

        participant_ids = await self.client.get_round_participants(round_num)
        participant_ids = sorted(set(int(x) for x in participant_ids))

        if not participant_ids:
            raise RuntimeError(
                f"No participants recorded for round {round_num} in SARSA mode"
            )

        return participant_ids

    async def fetch_models_strict(self, round_num: int, participant_ids):
        """
        Fetch exactly the expected models for round_num.
        Remote peer models via HTTP; self model from local file.
        """
        participant_ids = sorted(set(int(x) for x in participant_ids))
        expected_remote = [cid for cid in participant_ids if cid != self.CLIENT_ID]

        logger.info(
            f"[Client {self.CLIENT_ID}] Waiting for models from "
            f"participants for round {round_num}: {participant_ids}"
        )

        start_time = time.time()
        peer_state_dicts = {}

        excluded = [
            cid for cid in self.CLIENT_IDS
            if cid not in expected_remote and cid != self.CLIENT_ID
        ]

        while True:
            elapsed = time.time() - start_time

            peer_state_dicts = await self.client.fetch_all_peer_models(
                round_num,
                exclude_peers=excluded,
            )

            fetched_remote = sorted(int(k) for k in peer_state_dicts.keys())

            if all(cid in fetched_remote for cid in expected_remote):
                break

            if elapsed > self.BARRIER_TIMEOUT:
                missing = [cid for cid in expected_remote
                           if cid not in fetched_remote]
                raise RuntimeError(
                    f"Timeout fetching models for round {round_num}. "
                    f"Missing peers: {missing}"
                )

            await asyncio.sleep(self.FETCH_POLL_INTERVAL)

        models = []

        for cid in participant_ids:
            if cid == self.CLIENT_ID:
                local_path = os.path.join(
                    self.models_dir,
                    f"round_{round_num}_client{cid}.pt"
                )
                if not os.path.exists(local_path):
                    raise FileNotFoundError(
                        f"Missing local round model: {local_path}"
                    )
                m = RegressionModel(self.input_dim)
                m.load_state_dict(torch.load(local_path, map_location="cpu"))
                models.append(m)
            else:
                key = cid if cid in peer_state_dicts else str(cid)
                if key not in peer_state_dicts:
                    raise RuntimeError(
                        f"Peer {cid} model missing after strict fetch"
                    )
                m = RegressionModel(self.input_dim)
                m.load_state_dict(peer_state_dicts[key])
                models.append(m)

        return models

    async def wait_for_all_peers_healthy(self, timeout=120):
        """
        Wait until all peer HTTP servers are responding.
        Prevents race condition where early starters broadcast before late
        starters are ready.
        """
        start_time = time.time()
        k = len(self.peers_by_int_id)

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                healthy_count = 0

                for peer_id, config in self.peers_by_int_id.items():
                    try:
                        url = f"http://{config['ip']}:{config['port']}/health"
                        async with session.get(url, timeout=2) as response:
                            if response.status == 200:
                                healthy_count += 1
                    except Exception:
                        pass

                if healthy_count == k:
                    logger.info(f"✓ All {k} peers healthy!")
                    return True

                logger.info(f"Waiting... {healthy_count}/{k} peers healthy")
                await asyncio.sleep(2)

        raise RuntimeError(
            f"Timeout: Only {healthy_count}/{k} peers became healthy"
        )

    async def training_loop(self):
        logger.info("=" * 60)
        logger.info(
            f"[Client {self.CLIENT_ID}] STARTING {self.ALGORITHM.upper()} "
            f"TRAINING"
        )
        logger.info("=" * 60)

        self.server.run_threaded()
        await asyncio.sleep(self.SERVER_STARTUP_SECONDS)

        await self.wait_for_all_peers_healthy(timeout=120)

        self.load_data()

        # ── Algorithm-specific pre-training setup ──────────────────────────
        if self.ALGORITHM == "gadfl":
            await self._compute_gadfl_weights()
        elif self.ALGORITHM == "feddkw":
            await self._compute_feddkw_weights()
        elif self.ALGORITHM == "hybrid_gadfl":
            await self._compute_hybrid_weights()
        elif self.ALGORITHM == "sr_gadfl":
            await self._compute_sr_gadfl_weights()
        elif self.ALGORITHM == "g_feddkw":
            await self._compute_g_feddkw_weights()
        elif self.ALGORITHM == "t_g_feddkw":
            await self._compute_t_g_feddkw_weights()
        elif self.ALGORITHM == "sd_gadfl":
            await self._compute_sd_gadfl_weights()
        else:
            await self._compute_distribution_weight()  # SARSA: shares std; others: sets 1.0

        self.setup_log_files()

        metrics_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_metrics_{self.ALGORITHM}.csv"
        )
        waiting_file = os.path.join(
            self.results_dir,
            f"client{self.CLIENT_ID}_waiting_{self.ALGORITHM}.csv"
        )

        for rnd in range(1, self.TOTAL_ROUNDS + 1):
            logger.info(
                f"\n=== Round {rnd} - Client {self.CLIENT_ID} "
                f"({self.ALGORITHM.upper()}) ==="
            )

            try:
                # ── BARRIER WAIT ─────────────────────────────────────────
                # SARSA: conditional barrier (CHANGE 1)
                # All others (incl. GADFL): full barrier
                wait_start = time.time()
                if rnd > 1:
                    await self.wait_for_round_completion(rnd - 1)
                wait_time = time.time() - wait_start

                # ── AGGREGATE ────────────────────────────────────────────
                if rnd == 1:
                    models_for_aggregation = [self.model]
                    init_eval = evaluate(self.model, self.X_val, self.y_val)
                    self.prev_accuracy = max(float(init_eval["R2"]), -1.0)
                    logger.info(
                        f"[Client {self.CLIENT_ID}] "
                        f"Initial pre-training R2: {self.prev_accuracy:.4f}"
                    )
                else:
                    participant_ids = await self.get_expected_participants(rnd - 1)
                    models_for_aggregation = await self.fetch_models_strict(
                        rnd - 1, participant_ids
                    )

                # ── AGGREGATION STEP ──────────────────────────────────────
                # GADFL:   weighted aggregation using JSD weights
                # FedDkw:  weighted aggregation using KL weights
                # Others:  uniform FedAvg
                if self.ALGORITHM == "gadfl" and rnd > 1:
                    participant_ids_sorted = sorted(
                        set(int(x) for x in participant_ids)
                    )
                    weights_ordered = [
                        self.gadfl_weights.get(pid, 1.0 / len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(
                        models_for_aggregation, weights_ordered
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [GADFL] "
                        f"JSD weighted aggregation from {len(models_for_aggregation)} peers"
                    )

                elif self.ALGORITHM == "feddkw" and rnd > 1:
                    participant_ids_sorted = sorted(
                        set(int(x) for x in participant_ids)
                    )
                    weights_ordered = [
                        self.feddkw_weights.get(pid, 1.0 / len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(
                        models_for_aggregation, weights_ordered
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [FedDkw] "
                        f"KL weighted aggregation from {len(models_for_aggregation)} peers"
                    )

                elif self.ALGORITHM == "hybrid_gadfl" and rnd > 1:
                    participant_ids_sorted = sorted(
                        set(int(x) for x in participant_ids)
                    )
                    weights_ordered = [
                        self.hybrid_weights.get(pid, 1.0 / len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(
                        models_for_aggregation, weights_ordered
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [Hybrid-GADFL] "
                        f"JSD+KL weighted aggregation from "
                        f"{len(models_for_aggregation)} peers"
                    )

                elif self.ALGORITHM == "sr_gadfl" and rnd > 1:
                    participant_ids_sorted = sorted(
                        set(int(x) for x in participant_ids)
                    )
                    weights_ordered = [
                        self.sr_weights.get(pid, 1.0 / len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(
                        models_for_aggregation, weights_ordered
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [SR-GADFL] "
                        f"Self-regularized weighted aggregation "
                        f"(alpha={self.SR_ALPHA}, self-weight={self.sr_weights.get(self.CLIENT_ID,0):.3f})"
                    )

                elif self.ALGORITHM == "g_feddkw" and rnd > 1:
                    participant_ids_sorted = sorted(set(int(x) for x in participant_ids))
                    weights_ordered = [
                        self.gfeddkw_weights.get(pid, 1.0/len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(models_for_aggregation, weights_ordered)
                    logger.info(f"[Client {self.CLIENT_ID}] [G-FedDkw] JSD-global aggregation")

                elif self.ALGORITHM == "t_g_feddkw" and rnd > 1:
                    participant_ids_sorted = sorted(set(int(x) for x in participant_ids))
                    weights_ordered = [
                        self.tg_weights.get(pid, 1.0/len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(models_for_aggregation, weights_ordered)
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [T-G-FedDkw] "
                        f"Target-aware aggregation (oracle)"
                    )

                elif self.ALGORITHM == "sd_gadfl" and rnd > 1:
                    participant_ids_sorted = sorted(set(int(x) for x in participant_ids))
                    weights_ordered = [
                        self.sd_weights.get(pid, 1.0/len(participant_ids_sorted))
                        for pid in participant_ids_sorted
                    ]
                    aggregated_state = weighted_fedavg(models_for_aggregation, weights_ordered)
                    logger.info(
                        f"[Client {self.CLIENT_ID}] [SD-GADFL] "
                        f"Std-proportional weighted aggregation"
                    )

                else:
                    aggregated_state = fedavg(models_for_aggregation)

                self.model.load_state_dict(aggregated_state)
                torch.save(
                    aggregated_state,
                    os.path.join(
                        self.models_dir,
                        f"aggregated_model_before_round_{rnd}_"
                        f"client{self.CLIENT_ID}.pt"
                    )
                )

                # ── STEP 1: PARTICIPATION DECISION ────────────────────────
                # SARSA: use SARSA agent to decide
                # All others (incl. GADFL): always participate
                beta_to_log   = np.nan
                sync_prob     = 1.0
                participated  = True

                if self.ALGORITHM == "sarsa":
                    current_state  = self.agent.discretize(
                        self.prev_training_time, self.prev_accuracy
                    )
                    current_action = self.agent.choose_action(current_state)
                    current_beta   = self.agent.beta_values[current_action]
                    beta_to_log    = current_beta

                    est_straggler = self.prev_training_time / (
                        wait_time + self.prev_training_time + 1e-8
                    )
                    sync_prob = 1.0 / (
                        1.0 + np.exp(
                            -(self.LAMBDA1 * self.prev_accuracy
                              + self.LAMBDA2 * self.distribution_weight
                              - self.LAMBDA3 * est_straggler * current_beta)
                        )
                    )
                    participated = bool(np.random.rand() < sync_prob)

                    threshold         = 0.73
                    penalty           = 3.0 * max(0, threshold - self.prev_accuracy)
                    reward_prev_round = -2.0 * self.prev_straggler_score - penalty
                    reward_prev_round = np.clip(reward_prev_round, -5.0, 1.0)

                    if (self.sarsa_prev_state is not None
                            and self.sarsa_prev_action is not None):
                        self.agent.update_with_state_action(
                            self.sarsa_prev_state, self.sarsa_prev_action,
                            reward_prev_round, current_state, current_action,
                        )
                    self.sarsa_prev_state  = current_state
                    self.sarsa_prev_action = current_action
                    self.agent.save(rnd, dir=self.sarsa_qtables_dir)

                    if rnd == 20:
                        new_eps = self.agent.decay_epsilon(min_epsilon=0.05)
                        logger.info(
                            f"[Client {self.CLIENT_ID}] Epsilon decayed to "
                            f"{new_eps:.3f} at round 20"
                        )

                elif self.ALGORITHM == "gadfl":
                    # GADFL: always participate, log adaptive beta
                    beta_to_log  = self.gadfl_beta
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "feddkw":
                    beta_to_log  = self.feddkw_beta
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "hybrid_gadfl":
                    beta_to_log  = self.hybrid_beta1
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "sr_gadfl":
                    beta_to_log  = self.sr_beta
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "g_feddkw":
                    beta_to_log  = self.gfeddkw_beta
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "t_g_feddkw":
                    beta_to_log  = self.tg_beta
                    sync_prob    = 1.0
                    participated = True

                elif self.ALGORITHM == "sd_gadfl":
                    beta_to_log  = np.nan  # no beta in SD-GADFL
                    sync_prob    = 1.0
                    participated = True

                # ── STEP 2: DECLARE (SARSA only) ─────────────────────────
                # GADFL/FedAvg/FedProx skip declaration — everyone participates
                if self.ALGORITHM == "sarsa":
                    logger.info(
                        f"[Client {self.CLIENT_ID}] Round {rnd}: "
                        f"Declaring participation={participated}, "
                        f"wait_time={wait_time:.3f}s"
                    )
                    await self.client.declare_participation(
                        rnd, participated, wait_time
                    )

                    # ── STEP 3: POLL ENFORCEMENT (SARSA only) ─────────────
                    _poll_interval = 0.05
                    _max_wait      = 15.0
                    _elapsed       = 0.0
                    enforcement_status = {}
                    while _elapsed < _max_wait:
                        enforcement_status = await self.client.get_enforcement_status(rnd)
                        if enforcement_status.get("enforcement_complete", False):
                            break
                        await asyncio.sleep(_poll_interval)
                        _elapsed += _poll_interval
                    else:
                        logger.warning(
                            f"[Client {self.CLIENT_ID}] Round {rnd}: "
                            f"Enforcement poll timed out."
                        )
                    original_participated = participated
                    participated = enforcement_status.get("participated", participated)
                    was_forced   = enforcement_status.get("forced", False)
                    if was_forced:
                        participated = True
                        logger.warning(
                            f"[Client {self.CLIENT_ID}] Round {rnd}: "
                            f"FORCED to participate (original: {original_participated})."
                        )
                else:
                    was_forced = False  # Not applicable for non-SARSA

                # Save final participation decision
                save_participation(rnd, self.CLIENT_ID, participated,
                                   dir=self.sync_decisions_dir)

                # ── STEP 4: TRAINING ──────────────────────────────────────
                # FedAvg:    plain SGD every round
                # FedProx:   FedProx with mu every round
                # GADFL:     FedProx with mu every round
                # SARSA:     plain SGD if participated, skip if not
                training_time   = 0.0
                straggler_score = 0.0
                train_res       = None
                test_res        = None
                accuracy        = self.prev_accuracy

                should_train = (
                    participated
                    or was_forced
                    or self.ALGORITHM != "sarsa"
                )

                if should_train:
                    train_start = time.time()

                    if self.ALGORITHM in ["fedprox", "gadfl", "feddkw", "hybrid_gadfl", "sr_gadfl", "g_feddkw", "t_g_feddkw", "sd_gadfl"]:
                        # FedProx proximal term — prevents local drift
                        global_state = {
                            k: v.clone()
                            for k, v in self.model.state_dict().items()
                        }
                        self.model = train_local_fedprox(
                            self.model, self.X_train, self.y_train,
                            global_state=global_state, mu=self.MU,
                            epochs=self.LOCAL_EPOCHS, lr=self.LR,
                            batch_size=self.BATCH_SIZE,
                        )
                    else:
                        # FedAvg / SARSA: plain SGD
                        self.model = train_local_batched(
                            self.model, self.X_train, self.y_train,
                            epochs=self.LOCAL_EPOCHS, lr=self.LR,
                            batch_size=self.BATCH_SIZE,
                        )

                    training_time = time.time() - train_start
                    self.save_training_time(rnd, training_time)

                    train_res = evaluate(self.model, self.X_train, self.y_train)
                    test_res  = evaluate(self.model, self.X_val,   self.y_val)
                    accuracy  = max(float(test_res["R2"]), -1.0)
                    straggler_score = training_time / (
                        wait_time + training_time + 1e-8
                    )
                    self.prev_accuracy        = accuracy
                    self.prev_training_time   = training_time
                    self.prev_straggler_score = straggler_score
                else:
                    # SARSA skip — no training, no upload
                    self.save_training_time(rnd, 0.0)
                    self.prev_straggler_score = 0.0

                # ── SAVE LOCAL CHECKPOINT ────────────────────────────────
                local_round_path = os.path.join(
                    self.models_dir,
                    f"round_{rnd}_client{self.CLIENT_ID}.pt"
                )
                torch.save(self.model.state_dict(), local_round_path)

                # ── BARRIER BROADCAST ─────────────────────────────────────
                # Safe fallbacks for metrics when client skipped
                _train_mse  = float(train_res["MSE"])  if train_res else 0.0
                _train_r2   = float(train_res["R2"])   if train_res else self.prev_accuracy
                _test_mse   = float(test_res["MSE"])   if test_res  else 0.0
                _test_rmse  = float(test_res["RMSE"])  if test_res  else 0.0
                _test_mae   = float(test_res["MAE"])   if test_res  else 0.0
                _test_r2    = float(test_res["R2"])    if test_res  else self.prev_accuracy

                if not participated:
                    # SARSA skip: broadcast barrier immediately, then set metrics
                    notified = await self.client.broadcast_barrier(
                        rnd, participated=False
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] Barrier broadcast "
                        f"(skipping — immediate, {notified} notifications)"
                    )
                    self.server.set_metrics({
                        "train_mse":             _train_mse,
                        "train_r2":              _train_r2,
                        "test_mse":              _test_mse,
                        "test_rmse":             _test_rmse,
                        "test_mae":              _test_mae,
                        "test_r2":               _test_r2,
                        "training_time_seconds": float(training_time),
                        "waiting_time_seconds":  float(wait_time),
                        "normalized_waiting":    float(straggler_score),
                    })

                else:
                    # Participating: upload model, set metrics, then broadcast barrier
                    # Apply Model DP before sharing (GADFL only)
                    # Gaussian noise added to model params before upload
                    # Protects against gradient inversion attacks
                    # Local model stays clean (noise only on shared copy)
                    if self.ALGORITHM == "gadfl" and self.GADFL_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.GADFL_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.GADFL_MODEL_SIGMA}")
                    elif self.ALGORITHM == "hybrid_gadfl" and self.HYBRID_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.HYBRID_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.HYBRID_MODEL_SIGMA}")
                    elif self.ALGORITHM == "sr_gadfl" and self.SR_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.SR_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.SR_MODEL_SIGMA}")
                    elif self.ALGORITHM == "g_feddkw" and self.GFEDDKW_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.GFEDDKW_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.GFEDDKW_MODEL_SIGMA}")
                    elif self.ALGORITHM == "t_g_feddkw" and self.TG_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.TG_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.TG_MODEL_SIGMA}")
                    elif self.ALGORITHM == "sd_gadfl" and self.SD_MODEL_SIGMA:
                        model_to_share = apply_model_dp(
                            self.model.state_dict(), self.SD_MODEL_SIGMA)
                        logger.info(f"[Client {self.CLIENT_ID}] [ModelDP] sigma={self.SD_MODEL_SIGMA}")
                    else:
                        model_to_share = self.model.state_dict()

                    ok = await self.client.upload_model(
                        model_to_share, round_num=rnd
                    )
                    if not ok:
                        raise RuntimeError(f"Upload failed for round {rnd}")
                    # Server stores noisy model (for peers)
                    # Local model stays clean (for own training)
                    self.server.set_model(model_to_share, rnd)

                    self.server.set_metrics({
                        "train_mse":             _train_mse,
                        "train_r2":              _train_r2,
                        "test_mse":              _test_mse,
                        "test_rmse":             _test_rmse,
                        "test_mae":              _test_mae,
                        "test_r2":               _test_r2,
                        "training_time_seconds": float(training_time),
                        "waiting_time_seconds":  float(wait_time),
                        "normalized_waiting":    float(straggler_score),
                    })

                    notified = await self.client.broadcast_barrier(
                        rnd, participated=True
                    )
                    logger.info(
                        f"[Client {self.CLIENT_ID}] Barrier broadcast "
                        f"(participating, {notified} notifications)"
                    )

                # ── WRITE METRICS CSV ────────────────────────────────────
                with open(metrics_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{rnd},{_train_mse:.6f},{_train_r2:.6f},"
                        f"{_test_mse:.6f},{_test_rmse:.6f},"
                        f"{_test_mae:.6f},{_test_r2:.6f}\n"
                    )

                # ── WRITE WAITING CSV ────────────────────────────────────
                # effective_waiting_time:
                #   = waiting_time if participated
                #   = 0.0          if skipped (SARSA)
                effective_waiting_time = wait_time if participated else 0.0

                # SARSA-specific columns → nan for GADFL/FedAvg/FedProx
                straggler_log = (
                    straggler_score if self.ALGORITHM == "sarsa" else np.nan
                )
                dist_weight_log = (
                    self.distribution_weight
                    if self.ALGORITHM == "sarsa" else np.nan
                )

                with open(waiting_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{rnd},{training_time:.4f},{wait_time:.4f},"
                        f"{effective_waiting_time:.4f},{straggler_log:.4f},"
                        f"{dist_weight_log:.4f},"
                        f"{beta_to_log},{sync_prob:.4f},"
                        f"{int(participated)},{accuracy:.6f}\n"
                    )

                logger.info(
                    f"[C{self.CLIENT_ID} R{rnd}] "
                    f"R2={accuracy:.4f} | "
                    f"Train={training_time:.2f}s | "
                    f"Wait={wait_time:.2f}s | "
                    f"EffWait={effective_waiting_time:.2f}s | "
                    f"Sync={participated}"
                )

            except KeyboardInterrupt:
                logger.info(f"[Client {self.CLIENT_ID}] Interrupted by user")
                break
            except Exception as e:
                logger.error(
                    f"[Client {self.CLIENT_ID}] Error in round {rnd}: {e}"
                )
                traceback.print_exc()
                raise

        logger.info("\n" + "=" * 60)
        logger.info(
            f"[Client {self.CLIENT_ID}] Creating FINAL GLOBAL MODEL "
            f"({self.ALGORITHM.upper()})"
        )
        logger.info("=" * 60)

        await self.wait_for_round_completion(self.TOTAL_ROUNDS)

        final_participants = await self.get_expected_participants(self.TOTAL_ROUNDS)
        final_models       = await self.fetch_models_strict(
            self.TOTAL_ROUNDS, final_participants
        )

        # Final aggregation: weighted for weighted-aggregation algorithms,
        # uniform FedAvg for the rest
        final_participants_sorted = sorted(set(int(x) for x in final_participants))
        n_final = len(final_participants_sorted)

        if self.ALGORITHM == "gadfl":
            final_weights = [
                self.gadfl_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "feddkw":
            final_weights = [
                self.feddkw_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "hybrid_gadfl":
            final_weights = [
                self.hybrid_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "sr_gadfl":
            final_weights = [
                self.sr_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "g_feddkw":
            final_weights = [
                self.gfeddkw_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "t_g_feddkw":
            final_weights = [
                self.tg_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        elif self.ALGORITHM == "sd_gadfl":
            final_weights = [
                self.sd_weights.get(pid, 1.0 / n_final)
                for pid in final_participants_sorted
            ]
            final_global_state = weighted_fedavg(final_models, final_weights)

        else:
            # fedavg, fedprox, sarsa: uniform FedAvg
            final_global_state = fedavg(final_models)

        logger.info(
            f"[Client {self.CLIENT_ID}] Final aggregation method: "
            f"{'weighted' if self.ALGORITHM != 'fedavg' and self.ALGORITHM != 'fedprox' and self.ALGORITHM != 'sarsa' else 'uniform'} "
            f"({self.ALGORITHM})"
        )

        final_model = RegressionModel(self.input_dim)
        final_model.load_state_dict(final_global_state)

        torch.save(
            final_global_state,
            os.path.join(
                self.models_dir,
                f"final_global_model_{self.ALGORITHM}.pt"
            )
        )
        torch.save(
            final_global_state,
            os.path.join(
                self.models_dir,
                f"aggregated_model_final_round_"
                f"{self.TOTAL_ROUNDS}_{self.ALGORITHM}.pt"
            )
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.models_dir,
                f"final_local_client{self.CLIENT_ID}_{self.ALGORITHM}.pt"
            )
        )

        final_test_res = evaluate(final_model, self.X_val, self.y_val)

        logger.info(
            f"[Client {self.CLIENT_ID}] FINAL GLOBAL MODEL "
            f"({self.ALGORITHM.upper()}) Performance:"
        )
        logger.info(f"   MSE:  {final_test_res['MSE']:.4f}")
        logger.info(f"   RMSE: {final_test_res['RMSE']:.4f}")
        logger.info(f"   MAE:  {final_test_res['MAE']:.4f}")
        logger.info(f"   R2:   {final_test_res['R2']:.4f}")

        logger.info(f"[Client {self.CLIENT_ID}] Training complete!")
        logger.info(
            f"[Client {self.CLIENT_ID}] Keeping server alive briefly "
            f"for peer finalization..."
        )
        await asyncio.sleep(10.0)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Distributed Federated Learning Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 client.py 1                          # use config file settings
  python3 client.py 1 --dp_epsilon null        # GADFL, no differential privacy
  python3 client.py 1 --dp_epsilon 2.0         # GADFL, epsilon=2.0
  python3 client.py 1 --dp_epsilon 0.5         # GADFL, epsilon=0.5 (strong privacy)
  python3 client.py 3 --config custom.yaml     # custom config file

Note:
  --dp_epsilon only has effect when algorithm=gadfl in config.
  Each epsilon value saves results in a separate folder:
    gadfl_ep_null/   (no DP)
    gadfl_ep_0.5/
    gadfl_ep_1.0/
    gadfl_ep_2.0/
    gadfl_ep_5.0/
        """
    )

    parser.add_argument(
        "client_id",
        type=int,
        help="Client ID (1-7 matching peer_config.yaml)"
    )
    parser.add_argument(
        "--dp_epsilon",
        type=str,
        default=None,
        help=(
            "Differential privacy epsilon for GADFL histogram sharing. "
            "'null' = no DP (exact histogram). "
            "Float = Laplace noise scale (smaller = more private). "
            "If not provided, uses value from peer_config.yaml."
        )
    )
    parser.add_argument(
        "--model_sigma",
        type=str,
        default=None,
        help=(
            "Gaussian noise sigma for Model DP before sharing. "
            "'null' = no model DP. "
            "Float = sigma value (e.g. 0.01). "
            "If not provided, uses value from peer_config.yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/peer_config.yaml",
        help="Path to peer config YAML file (default: config/peer_config.yaml)"
    )

    args = parser.parse_args()

    # Parse dp_epsilon argument
    dp_epsilon_override = None
    dp_epsilon_provided = args.dp_epsilon is not None

    if dp_epsilon_provided:
        if args.dp_epsilon.lower() in ["null", "none", "inf", "infinity"]:
            dp_epsilon_override = None
        else:
            try:
                dp_epsilon_override = float(args.dp_epsilon)
                if dp_epsilon_override <= 0:
                    print(f"Error: --dp_epsilon must be positive, got {dp_epsilon_override}")
                    sys.exit(1)
            except ValueError:
                print(f"Error: --dp_epsilon must be a number or 'null', got '{args.dp_epsilon}'")
                sys.exit(1)

    # Parse model_sigma argument
    model_sigma_override = None
    model_sigma_provided = args.model_sigma is not None

    if model_sigma_provided:
        if args.model_sigma.lower() in ["null", "none", "0"]:
            model_sigma_override = None
        else:
            try:
                model_sigma_override = float(args.model_sigma)
                if model_sigma_override <= 0:
                    print(f"Error: --model_sigma must be positive, got {model_sigma_override}")
                    sys.exit(1)
            except ValueError:
                print(f"Error: --model_sigma must be a number or 'null', got '{args.model_sigma}'")
                sys.exit(1)

    setup_logging(args.client_id, "algorithm", "logs")

    try:
        client = DistributedClient(
            client_id            = args.client_id,
            config_path          = args.config,
            dp_epsilon_override  = dp_epsilon_override  if dp_epsilon_provided  else "USE_CONFIG",
            model_sigma_override = model_sigma_override if model_sigma_provided else "USE_CONFIG",
        )
        setup_logging(args.client_id, client.ALGORITHM, client.logs_dir)
        await client.training_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal top-level error: {e}")
        traceback.print_exc()
        sys.exit(1)
