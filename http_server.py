import io
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any

import torch
from flask import Flask, request, jsonify


class DFLServer:
    """
    HTTP server for DFL peer communication.

    Supports:
    - Model upload/download by round
    - Health checks
    - Metrics export
    - Barrier synchronization
    - Per-round participation tracking
    - Data stat sharing (std for SARSA variance mode)
    - Histogram sharing (for GADFL geography-aware aggregation)  ← NEW
    """

    def __init__(self, peer_id: int, config: Dict[str, Any], port: int = 9000):
        self.peer_id = int(peer_id)
        self.config = config
        self.port = int(port)
        self.app = Flask(f"dfl_peer_{self.peer_id}")

        # Store models by round
        self.models_by_round: Dict[int, Dict] = {}
        self.current_model_round = 0
        self.last_update_time = None

        # Completed round is separate from model round
        self.completed_round = 0

        # { round_num: { peer_id: {"participated": bool, "timestamp": str} } }
        self.barriers_by_round: Dict[int, Dict[int, Dict[str, Any]]] = {}

        # { round_num: { peer_id: {"participated": bool, "wait_time": float, "forced": bool} } }
        self.participation_declarations: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.enforcement_complete: Dict[int, bool] = {}

        # ── DISTRIBUTION WEIGHT SUPPORT (SARSA variance mode) ──────────────────
        # Stores local data std shared by each client at startup.
        # { peer_id: float }
        self.data_stats: Dict[int, float] = {}

        # ── GADFL HISTOGRAM SUPPORT ─────────────────────────────────────────────
        # Stores speed histograms shared by each client before training begins.
        # Each histogram is a list of `num_bins` floats summing to 1.0.
        # Shared ONCE before round 1; used to compute JSD and aggregation weights.
        # { peer_id: [float, float, ..., float] }
        self.histograms: Dict[int, list] = {}

        self.last_metrics = {
            "train_mse": 0.0,
            "train_r2": 0.0,
            "test_mse": 0.0,
            "test_rmse": 0.0,
            "test_mae": 0.0,
            "test_r2": 0.0,
            "training_time_seconds": 0.0,
            "waiting_time_seconds": 0.0,
            "normalized_waiting": 0.0,
        }

        self.server_start_time = time.time()
        self.logger = logging.getLogger(f"HTTPServer-{self.peer_id}")
        self._lock = threading.Lock()

        self._register_routes()
        self.logger.info(f"DFL Server initialized for Peer {self.peer_id} on port {self.port}")

    def _peer_name(self) -> str:
        peers = self.config["peers"]
        if self.peer_id in peers:
            return peers[self.peer_id].get("name", f"peer_{self.peer_id}")
        if str(self.peer_id) in peers:
            return peers[str(self.peer_id)].get("name", f"peer_{self.peer_id}")
        return f"peer_{self.peer_id}"

    def _ensure_round_entry(self, round_num: int):
        if round_num not in self.barriers_by_round:
            self.barriers_by_round[round_num] = {}

    def mark_round_complete(self, round_num: int, participated: bool):
        round_num = int(round_num)
        with self._lock:
            self._ensure_round_entry(round_num)
            self.barriers_by_round[round_num][self.peer_id] = {
                "participated": bool(participated),
                "timestamp": datetime.utcnow().isoformat(),
            }
            if round_num > self.completed_round:
                self.completed_round = round_num

    def get_round_participants(self, round_num: int):
        with self._lock:
            round_info = self.barriers_by_round.get(int(round_num), {})
            return sorted(
                peer_id
                for peer_id, meta in round_info.items()
                if bool(meta.get("participated", False))
            )

    def _enforce_minimum_participants(self, round_num: int):
        """
        Ensure at least 2 clients participate in this round.
        If total < 2, force the clients with lowest wait_time to participate.
        """
        declarations = self.participation_declarations[round_num]

        participants = [
            client_id for client_id, info in declarations.items()
            if info.get("participated", False)
        ]

        if len(participants) < 2:
            min_required = 2 - len(participants)

            non_participants = [
                (client_id, info["wait_time"])
                for client_id, info in declarations.items()
                if not info.get("participated", False)
            ]

            non_participants.sort(key=lambda x: x[1])

            forced_clients = [client_id for client_id, _ in non_participants[:min_required]]

            for client_id in forced_clients:
                declarations[client_id]["participated"] = True
                declarations[client_id]["forced"] = True
                self.logger.info(
                    f"[Enforcement] Round {round_num}: "
                    f"Client {client_id} forced to participate "
                    f"(wait_time={declarations[client_id]['wait_time']:.3f}s)"
                )

            wait_times = [declarations[cid]["wait_time"] for cid in forced_clients]
            self.logger.warning(
                f"Round {round_num}: Total participants < 2. "
                f"Forced clients {forced_clients} to participate "
                f"(wait times: {[f'{t:.3f}s' for t in wait_times]})"
            )

        self.enforcement_complete[round_num] = True
        self.logger.info(f"Round {round_num}: Enforcement complete.")

    def _register_routes(self):
        """Register all HTTP routes."""

        @self.app.route("/sync/get_enforcement_status", methods=["GET"])
        def get_enforcement_status():
            try:
                round_num = request.args.get("round", type=int)
                peer_id = request.args.get("peer_id", type=int)
                if round_num is None:
                    return jsonify({"error": "Missing round parameter"}), 400
                if peer_id is None:
                    return jsonify({"error": "Missing peer_id parameter"}), 400
                with self._lock:
                    declarations = self.participation_declarations.get(round_num, {})
                    their_declaration = declarations.get(peer_id, {})

                    if not their_declaration:
                        return jsonify({
                            "peer_id": self.peer_id,
                            "round": round_num,
                            "participated": False,
                            "forced": False,
                            "enforcement_complete": self.enforcement_complete.get(round_num, False),
                            "timestamp": datetime.utcnow().isoformat(),
                        }), 200

                    return jsonify({
                        "peer_id": peer_id,
                        "round": round_num,
                        "participated": their_declaration.get("participated", False),
                        "forced": their_declaration.get("forced", False),
                        "enforcement_complete": self.enforcement_complete.get(round_num, False),
                        "timestamp": datetime.utcnow().isoformat(),
                    }), 200

            except Exception as e:
                self.logger.error(f"Get enforcement status failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/health", methods=["GET"])
        def health():
            try:
                with self._lock:
                    response = {
                        "status": "healthy",
                        "peer_id": self.peer_id,
                        "peer_name": self._peer_name(),
                        "round": self.completed_round,
                        "model_round": self.current_model_round,
                        "timestamp": datetime.utcnow().isoformat(),
                        "uptime_seconds": round(time.time() - self.server_start_time, 2),
                        "has_model": self.current_model_round in self.models_by_round,
                        "stored_model_rounds": sorted(self.models_by_round.keys()),
                    }
                return jsonify(response), 200
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500

        @self.app.route("/model/download", methods=["GET"])
        def download_model():
            try:
                round_num = request.args.get("round", type=int)

                with self._lock:
                    if round_num is not None:
                        if round_num not in self.models_by_round:
                            return jsonify({
                                "error": f"Model for round {round_num} not found",
                                "available_rounds": sorted(self.models_by_round.keys()),
                            }), 404
                        model_to_send = self.models_by_round[round_num]
                        model_round = round_num
                    else:
                        if self.current_model_round not in self.models_by_round:
                            return jsonify({"error": "No model available"}), 404
                        model_to_send = self.models_by_round[self.current_model_round]
                        model_round = self.current_model_round

                    buffer = io.BytesIO()
                    torch.save(model_to_send, buffer)
                    model_bytes = buffer.getvalue()

                self.logger.info(
                    f"Model download request: round {model_round}, size {len(model_bytes)} bytes"
                )

                return model_bytes, 200, {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(model_bytes)),
                }

            except Exception as e:
                self.logger.error(f"Download failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/model/upload", methods=["POST"])
        def upload_model():
            try:
                round_num = request.args.get("round", type=int)
                if round_num is None:
                    return jsonify({"error": "Missing round query parameter"}), 400

                model_bytes = request.get_data()
                if len(model_bytes) > 10_000_000:
                    self.logger.warning(f"Model too large: {len(model_bytes)} bytes (>10MB)")
                    return jsonify({"error": "Model too large (>10MB)"}), 413

                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location="cpu")

                if not isinstance(state_dict, dict):
                    return jsonify({"error": "Invalid model format (not a state_dict)"}), 400

                with self._lock:
                    self.models_by_round[int(round_num)] = state_dict
                    self.current_model_round = int(round_num)
                    self.last_update_time = datetime.utcnow().isoformat()

                self.logger.info(
                    f"Model uploaded: round {round_num}, size {len(model_bytes)} bytes"
                )

                return jsonify({
                    "status": "saved",
                    "round": int(round_num),
                    "model_size_bytes": len(model_bytes),
                    "timestamp": self.last_update_time,
                    "peer_id": self.peer_id,
                    "available_rounds": sorted(self.models_by_round.keys()),
                }), 200

            except Exception as e:
                self.logger.error(f"Upload failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/metrics", methods=["GET"])
        def get_metrics():
            try:
                with self._lock:
                    response = {
                        "peer_id": self.peer_id,
                        "peer_name": self._peer_name(),
                        "completed_round": self.completed_round,
                        "model_round": self.current_model_round,
                        "timestamp": datetime.utcnow().isoformat(),
                        "training_metrics": self.last_metrics,
                        "model_updated_at": self.last_update_time,
                    }
                return jsonify(response), 200
            except Exception as e:
                self.logger.error(f"Metrics request failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/sync/declare_participation", methods=["POST"])
        def declare_participation():
            """
            Client declares its participation intent and wait time.
            Server collects all declarations and enforces minimum 2 participants.
            """
            try:
                data = request.get_json(force=True)
                round_num = int(data.get("round"))
                peer_id = int(data.get("peer_id"))
                participated = bool(data.get("participated"))
                wait_time = float(data.get("wait_time", 0.0))

                with self._lock:
                    if round_num not in self.participation_declarations:
                        self.participation_declarations[round_num] = {}

                    self.participation_declarations[round_num][peer_id] = {
                        "participated": participated,
                        "wait_time": wait_time,
                        "forced": False,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    total_clients = len(self.config["peers"])
                    declared_count = len(self.participation_declarations[round_num])

                    self.logger.info(
                        f"Round {round_num}: Client {peer_id} declared "
                        f"participated={participated} (wait_time={wait_time:.3f}s) "
                        f"({declared_count}/{total_clients} declared)"
                    )

                    if declared_count == total_clients:
                        self.logger.info(
                            f"Round {round_num}: All {total_clients} clients declared. "
                            f"Running enforcement..."
                        )
                        self._enforce_minimum_participants(round_num)

                return jsonify({"status": "recorded"}), 200

            except Exception as e:
                self.logger.error(f"Declare participation failed: {e}")
                return jsonify({"error": str(e)}), 400

        @self.app.route("/sync/get_final_participation", methods=["GET"])
        def get_final_participation():
            """Get final participation status after enforcement logic applied."""
            try:
                round_num = request.args.get("round", type=int)
                if round_num is None:
                    return jsonify({"error": "Missing round parameter"}), 400

                with self._lock:
                    declarations = self.participation_declarations.get(round_num, {})
                    participants = [
                        client_id for client_id, info in declarations.items()
                        if info.get("participated", False)
                    ]

                return jsonify({
                    "peer_id": self.peer_id,
                    "round": round_num,
                    "participants": sorted(participants),
                    "count": len(participants),
                    "timestamp": datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"Get final participation failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/sync/barrier", methods=["POST"])
        def barrier_signal():
            try:
                data = request.get_json(force=True)
                round_num = int(data.get("round"))
                source_peer = int(data.get("peer_id"))
                participated = bool(data.get("participated", True))
                ts = data.get("timestamp") or datetime.utcnow().isoformat()

                with self._lock:
                    self._ensure_round_entry(round_num)
                    self.barriers_by_round[round_num][source_peer] = {
                        "participated": participated,
                        "timestamp": ts,
                    }

                    if source_peer == self.peer_id and round_num > self.completed_round:
                        self.completed_round = round_num

                    received_from = sorted(self.barriers_by_round[round_num].keys())
                    participants = sorted(
                        pid for pid, meta in self.barriers_by_round[round_num].items()
                        if bool(meta.get("participated", False))
                    )
                    total_peers = len(self.config["peers"])

                return jsonify({
                    "status": "barrier received",
                    "round": round_num,
                    "source_peer": source_peer,
                    "participated": participated,
                    "received_from": received_from,
                    "participants": participants,
                    "all_received": len(received_from) == total_peers,
                    "timestamp": datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"Barrier processing failed: {e}")
                return jsonify({"error": str(e)}), 400

        @self.app.route("/sync/round_status", methods=["GET"])
        def round_status():
            try:
                round_num = request.args.get("round", type=int)
                if round_num is None:
                    return jsonify({"error": "Missing round query parameter"}), 400

                with self._lock:
                    info = self.barriers_by_round.get(round_num, {})
                    received_from = sorted(info.keys())
                    participants = sorted(
                        peer_id
                        for peer_id, meta in info.items()
                        if bool(meta.get("participated", False))
                    )
                    total_peers = len(self.config["peers"])

                return jsonify({
                    "peer_id": self.peer_id,
                    "round": round_num,
                    "received_from": received_from,
                    "participants": participants,
                    "all_received": len(received_from) == total_peers,
                    "num_received": len(received_from),
                    "total_peers": total_peers,
                    "timestamp": datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"Round status request failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/sync/participants", methods=["GET"])
        def round_participants():
            try:
                round_num = request.args.get("round", type=int)
                if round_num is None:
                    return jsonify({"error": "Missing round query parameter"}), 400

                participants = self.get_round_participants(round_num)
                return jsonify({
                    "peer_id": self.peer_id,
                    "round": round_num,
                    "participants": participants,
                    "timestamp": datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"Participants request failed: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/info", methods=["GET"])
        def info():
            try:
                peers = self.config["peers"]
                peer_config = peers[self.peer_id] if self.peer_id in peers else peers[str(self.peer_id)]
                with self._lock:
                    response = {
                        "peer_id": self.peer_id,
                        "name": peer_config.get("name"),
                        "device_type": peer_config.get("device_type"),
                        "ip": peer_config.get("ip"),
                        "port": self.port,
                        "dataset_path": peer_config.get("dataset_path"),
                        "completed_round": self.completed_round,
                        "model_round": self.current_model_round,
                        "stored_model_rounds": sorted(self.models_by_round.keys()),
                        "server_uptime_seconds": round(time.time() - self.server_start_time, 2),
                    }
                return jsonify(response), 200
            except Exception as e:
                self.logger.error(f"Info request failed: {e}")
                return jsonify({"error": str(e)}), 500

        # ── DATA STAT ENDPOINTS (for SARSA variance-based distribution weights) ──

        @self.app.route("/sync/share_data_stat", methods=["POST"])
        def share_data_stat():
            """
            Client shares its local data std at startup.
            Used for SARSA variance-based distribution weights.

            Body: { "peer_id": int, "stat_name": str, "value": float }
            """
            try:
                data     = request.get_json(force=True)
                peer_id  = int(data["peer_id"])
                value    = float(data["value"])

                with self._lock:
                    self.data_stats[peer_id] = value

                total_clients = len(self.config["peers"])
                self.logger.info(
                    f"[DataStat] Client {peer_id} shared std={value:.4f} "
                    f"({len(self.data_stats)}/{total_clients} received)"
                )
                return jsonify({"status": "recorded", "peer_id": peer_id}), 200

            except Exception as e:
                self.logger.error(f"share_data_stat failed: {e}")
                return jsonify({"error": str(e)}), 400

        @self.app.route("/sync/get_all_data_stats", methods=["GET"])
        def get_all_data_stats():
            """Returns all collected data stats (std per client)."""
            try:
                total_clients = len(self.config["peers"])
                with self._lock:
                    stats_copy = dict(self.data_stats)

                return jsonify({
                    "stats":         stats_copy,
                    "count":         len(stats_copy),
                    "total_clients": total_clients,
                    "all_received":  len(stats_copy) == total_clients,
                    "timestamp":     datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"get_all_data_stats failed: {e}")
                return jsonify({"error": str(e)}), 500

        # ── GADFL HISTOGRAM ENDPOINTS ─────────────────────────────────────────
        # These endpoints handle histogram sharing for GADFL.
        # Called ONCE before training begins (not every round).
        # The histogram is a list of num_bins floats summing to 1.0.
        # This is the basis for computing JSD distances and aggregation weights.

        @self.app.route("/sync/share_histogram", methods=["POST"])
        def share_histogram():
            """
            GADFL: Client shares its local speed histogram before training.

            The histogram is computed from local speed targets and represents
            the geographic speed distribution of this agent's environment.
            Optionally includes differential privacy noise before sharing.

            Called ONCE per client before round 1.

            Body: {
                "peer_id":   int,
                "histogram": [float, float, ..., float]  (num_bins floats, sums to 1)
            }
            """
            try:
                data      = request.get_json(force=True)
                peer_id   = int(data["peer_id"])
                histogram = data["histogram"]

                # Validate
                if not isinstance(histogram, list) or len(histogram) == 0:
                    return jsonify({"error": "histogram must be a non-empty list"}), 400

                with self._lock:
                    self.histograms[peer_id] = histogram

                total_clients = len(self.config["peers"])
                self.logger.info(
                    f"[GADFL] Client {peer_id} shared histogram "
                    f"({len(histogram)} bins, sum={sum(histogram):.4f}) "
                    f"({len(self.histograms)}/{total_clients} received)"
                )

                return jsonify({
                    "status":    "recorded",
                    "peer_id":   peer_id,
                    "num_bins":  len(histogram),
                    "received":  len(self.histograms),
                    "total":     total_clients,
                }), 200

            except Exception as e:
                self.logger.error(f"share_histogram failed: {e}")
                return jsonify({"error": str(e)}), 400

        @self.app.route("/sync/get_all_histograms", methods=["GET"])
        def get_all_histograms():
            """
            GADFL: Returns all collected histograms when every client has shared.

            Client polls this endpoint until all_received=True, then computes
            the JSD matrix and aggregation weights locally.

            Returns:
            {
                "histograms":   { "1": [...], "2": [...], ... },
                "count":        int,
                "total_clients": int,
                "all_received": bool,
                "timestamp":    str
            }
            """
            try:
                total_clients = len(self.config["peers"])
                with self._lock:
                    histograms_copy = {
                        str(k): v for k, v in self.histograms.items()
                    }

                return jsonify({
                    "histograms":    histograms_copy,
                    "count":         len(histograms_copy),
                    "total_clients": total_clients,
                    "all_received":  len(histograms_copy) == total_clients,
                    "timestamp":     datetime.utcnow().isoformat(),
                }), 200

            except Exception as e:
                self.logger.error(f"get_all_histograms failed: {e}")
                return jsonify({"error": str(e)}), 500

    def set_model(self, model_state_dict: Dict, round_num: int):
        with self._lock:
            self.models_by_round[int(round_num)] = model_state_dict
            self.current_model_round = int(round_num)
            self.last_update_time = datetime.utcnow().isoformat()
        self.logger.debug(f"Model updated: round {round_num}")

    def set_metrics(self, metrics: Dict[str, float]):
        with self._lock:
            self.last_metrics.update(metrics)
        self.logger.debug(f"Metrics updated: R²={metrics.get('test_r2', 0):.4f}")

    def run(self, host: str = "0.0.0.0", debug: bool = False):
        self.logger.info(f"Starting HTTP server on {host}:{self.port}")
        self.app.run(
            host=host,
            port=self.port,
            debug=debug,
            use_reloader=False,
            threaded=True,
        )

    def run_threaded(self, host: str = "0.0.0.0", daemon: bool = True) -> threading.Thread:
        server_thread = threading.Thread(
            target=self.run,
            args=(host,),
            daemon=daemon,
        )
        server_thread.start()
        self.logger.info(f"HTTP server started in background thread on {host}:{self.port}")
        return server_thread


def create_and_start_server(peer_id: int, config: Dict, port: int = 9000) -> DFLServer:
    server = DFLServer(peer_id, config, port)
    server.run_threaded()
    return server
