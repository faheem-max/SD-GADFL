import aiohttp
import asyncio
import io
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Set, List, Any

import torch


class DFLClient:
    """
    Async HTTP client for peer communication in DFL system.

    Handles:
    - Model fetching from peers
    - Model uploading to own server
    - Health checks
    - Barrier synchronization
    - Participation lookup
    - Data stat sharing (SARSA)
    - Histogram sharing (GADFL)   ← NEW
    """

    def __init__(
        self,
        peer_id: int,
        peer_config: Dict[str, Any],
        timeout: int = 30,
        retry_attempts: int = 2,
    ):
        self.peer_id = int(peer_id)
        self.peer_config = {int(k): v for k, v in peer_config.items()}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.logger = logging.getLogger(f"HTTPClient-{self.peer_id}")

        self.logger.info(
            f"DFL Client initialized for Peer {self.peer_id} "
            f"(timeout={timeout}s, retries={retry_attempts})"
        )

    def _get_peer_url(self, peer_id: int, endpoint: str) -> str:
        peer_id = int(peer_id)
        config = self.peer_config[peer_id]
        ip = config["ip"]
        port = config["port"]
        return f"http://{ip}:{port}{endpoint}"

    async def get_health(self, target_peer_id: int) -> Optional[int]:
        """Returns the peer's completed round number, or None if unavailable."""
        url = self._get_peer_url(target_peer_id, "/health")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        round_num = data.get("round", -1)
                        self.logger.debug(
                            f"Peer {target_peer_id} healthy: completed round {round_num}"
                        )
                        return int(round_num)
                    return None
        except asyncio.TimeoutError:
            self.logger.debug(f"Health check timeout for peer {target_peer_id}")
            return None
        except Exception as e:
            self.logger.debug(
                f"Health check failed for peer {target_peer_id}: {type(e).__name__}: {e}"
            )
            return None

    async def fetch_model(
        self,
        target_peer_id: int,
        round_num: int,
    ) -> Optional[Dict]:
        url = self._get_peer_url(target_peer_id, f"/model/download?round={round_num}")

        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            model_bytes = await resp.read()
                            buffer = io.BytesIO(model_bytes)
                            state_dict = torch.load(buffer, map_location="cpu")

                            self.logger.info(
                                f"Fetched model from peer {target_peer_id} "
                                f"round {round_num} ({len(model_bytes)} bytes)"
                            )
                            return state_dict

                        if resp.status == 404:
                            self.logger.debug(
                                f"Peer {target_peer_id}: round {round_num} model not ready"
                            )
                            return None

                        self.logger.warning(
                            f"Peer {target_peer_id}: HTTP {resp.status} on model fetch"
                        )
                        return None

            except asyncio.TimeoutError:
                if attempt < self.retry_attempts - 1:
                    self.logger.debug(
                        f"Fetch timeout from peer {target_peer_id} "
                        f"(attempt {attempt + 1}/{self.retry_attempts}), retrying..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    self.logger.warning(
                        f"Model fetch timeout from peer {target_peer_id} "
                        f"(round {round_num}) after {self.retry_attempts} attempts"
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"Model fetch error from peer {target_peer_id}: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                return None

        return None

    async def upload_model(
        self,
        model_state_dict: Dict,
        round_num: int,
    ) -> bool:
        url = self._get_peer_url(self.peer_id, f"/model/upload?round={round_num}")

        try:
            buffer = io.BytesIO()
            torch.save(model_state_dict, buffer)
            model_bytes = buffer.getvalue()

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    url,
                    data=model_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                ) as resp:
                    if resp.status == 200:
                        self.logger.info(
                            f"Model uploaded: round {round_num}, size {len(model_bytes)} bytes"
                        )
                        return True

                    self.logger.error(f"Upload failed: HTTP {resp.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Model upload error: {type(e).__name__}: {e}")
            return False

    async def broadcast_barrier(
        self,
        round_num: int,
        participated: bool = True,
    ) -> int:
        """
        Notify ALL peers, including self, that this peer completed the round.
        Including self is important so the local server can advance completed_round
        even when no model was uploaded (e.g. SARSA non-participation).
        """
        barrier_msg = {
            "round": int(round_num),
            "peer_id": self.peer_id,
            "participated": bool(participated),
            "timestamp": datetime.utcnow().isoformat(),
        }

        async def notify_peer(target_peer_id: int) -> bool:
            url = self._get_peer_url(target_peer_id, "/sync/barrier")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(url, json=barrier_msg) as resp:
                        return resp.status == 200
            except Exception as e:
                self.logger.debug(
                    f"Barrier notify to peer {target_peer_id} failed: {type(e).__name__}: {e}"
                )
                return False

        tasks = [notify_peer(peer_id) for peer_id in self.peer_config.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        total_targets = len(self.peer_config)

        self.logger.info(
            f"Barrier broadcast: {success_count}/{total_targets} targets notified "
            f"(round {round_num}, participated={participated})"
        )

        return success_count

    async def wait_for_peers_ready(
        self,
        expected_round: int,
        timeout: int = 30,
    ) -> Set[int]:
        """Wait until ALL peers (including self) report completed round >= expected_round."""
        start_time = time.time()
        peers_ready: Set[int] = set()
        poll_interval = 0.5

        while len(peers_ready) < len(self.peer_config):
            elapsed = time.time() - start_time

            if elapsed > timeout:
                self.logger.warning(
                    f"wait_for_peers_ready timeout: "
                    f"{len(peers_ready)}/{len(self.peer_config)} ready "
                    f"after {timeout}s"
                )
                break

            for peer_id in list(self.peer_config.keys()):
                if peer_id in peers_ready:
                    continue
                round_num = await self.get_health(peer_id)
                if round_num is not None and round_num >= expected_round:
                    peers_ready.add(peer_id)

            if len(peers_ready) < len(self.peer_config):
                await asyncio.sleep(poll_interval)

        return peers_ready

    async def wait_for_round_status_complete(
        self,
        round_num: int,
        timeout: int = 30,
    ) -> Optional[Dict]:
        """Wait for the server to mark round_num as fully declared."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            url = self._get_peer_url(self.peer_id, f"/sync/round_status?round={round_num}")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("all_received", False):
                                return data
            except Exception as e:
                self.logger.warning(f"round_status poll failed: {e}")
            await asyncio.sleep(0.5)

        self.logger.error(f"wait_for_round_status_complete timed out (round {round_num})")
        return None

    async def get_round_participants(self, round_num: int) -> List[int]:
        """Get list of participant peer IDs for a given round."""
        url = self._get_peer_url(self.peer_id, f"/sync/participants?round={round_num}")
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [int(p) for p in data.get("participants", [])]
        except Exception as e:
            self.logger.error(f"get_round_participants failed: {type(e).__name__}: {e}")
        return []

    async def fetch_all_peer_models(
        self,
        round_num: int,
        exclude_peers: Optional[List[int]] = None,
    ) -> Dict[int, Dict]:
        """Fetch models from all non-excluded peers for the given round."""
        exclude_peers = exclude_peers or []

        target_peers = [
            pid for pid in self.peer_config.keys()
            if pid != self.peer_id and pid not in exclude_peers
        ]

        async def fetch_one(target_id: int):
            return target_id, await self.fetch_model(target_id, round_num)

        results = await asyncio.gather(
            *[fetch_one(pid) for pid in target_peers],
            return_exceptions=True,
        )

        peer_state_dicts = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            peer_id, state_dict = result
            if state_dict is not None:
                peer_state_dicts[peer_id] = state_dict

        return peer_state_dicts

    async def fetch_all_metrics(self) -> Dict[int, Dict]:
        """Fetch metrics from all peers."""
        metrics = {}

        async def fetch_metrics(target_peer_id: int) -> Optional[Dict]:
            url = self._get_peer_url(target_peer_id, "/metrics")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.json()
            except Exception:
                pass
            return None

        tasks = [
            (peer_id, fetch_metrics(peer_id))
            for peer_id in self.peer_config.keys()
        ]

        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        for (peer_id, _), result in zip(tasks, results):
            if result is not None and not isinstance(result, Exception):
                metrics[int(peer_id)] = result

        return metrics

    async def declare_participation(
        self,
        round_num: int,
        participated: bool,
        wait_time: float,
    ) -> bool:
        """
        Declare this client's participation intent and wait time to all peers.
        Server will collect all declarations and enforce minimum 2 participants.
        """
        data = {
            "round": round_num,
            "peer_id": self.peer_id,
            "participated": participated,
            "wait_time": wait_time,
        }

        async def notify_peer(target_peer_id: int) -> bool:
            url = self._get_peer_url(target_peer_id, "/sync/declare_participation")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(url, json=data) as resp:
                        return resp.status == 200
            except Exception as e:
                self.logger.debug(
                    f"Declare to peer {target_peer_id} failed: {type(e).__name__}: {e}"
                )
                return False

        tasks = [notify_peer(peer_id) for peer_id in self.peer_config.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        total_targets = len(self.peer_config)

        if success_count >= total_targets:
            self.logger.info(
                f"Declared participation for round {round_num}: "
                f"participated={participated}, wait_time={wait_time:.3f}s"
            )
            return True

        self.logger.warning(
            f"Declaration failed: only {success_count}/{total_targets} peers acknowledged"
        )
        return False

    async def get_final_participation(
        self,
        round_num: int,
        source_peer_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get final participation status after enforcement logic applied."""
        target_peer = self.peer_id if source_peer_id is None else int(source_peer_id)
        url = self._get_peer_url(target_peer, f"/sync/get_final_participation?round={round_num}")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
        except Exception as e:
            self.logger.error(
                f"Get final participation from peer {target_peer} failed: {type(e).__name__}: {e}"
            )

        return {"participants": [], "count": 0}

    async def get_enforcement_status(
        self,
        round_num: int,
        source_peer_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query the server to see if THIS CLIENT was forced to participate."""
        target_peer = self.peer_id if source_peer_id is None else int(source_peer_id)
        url = self._get_peer_url(
            target_peer,
            f"/sync/get_enforcement_status?round={round_num}&peer_id={self.peer_id}"
        )

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
                    else:
                        self.logger.warning(
                            f"Get enforcement status failed for round {round_num}: "
                            f"HTTP {resp.status}"
                        )
                        return {
                            "peer_id": self.peer_id,
                            "round": round_num,
                            "participated": False,
                            "forced": False,
                        }
        except Exception as e:
            self.logger.error(
                f"Get enforcement status failed for round {round_num}: "
                f"{type(e).__name__}: {e}"
            )
            return {
                "peer_id": self.peer_id,
                "round": round_num,
                "participated": False,
                "forced": False,
            }

    async def share_data_stat(self, value: float) -> bool:
        """
        Share this client's local data std with all peers.
        Called once at startup before training begins.
        Used for SARSA variance-based distribution weight calculation.
        """
        payload = {
            "peer_id": self.peer_id,
            "stat_name": "std",
            "value": float(value),
        }
        success_count = 0
        for peer_id in self.peer_config.keys():
            url = self._get_peer_url(int(peer_id), "/sync/share_data_stat")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            success_count += 1
            except Exception as e:
                self.logger.warning(
                    f"share_data_stat to peer {peer_id} failed: {e}"
                )
        self.logger.info(
            f"[DataStat] Shared std={value:.4f} to "
            f"{success_count}/{len(self.peer_config)} peers"
        )
        return success_count == len(self.peer_config)

    async def get_all_data_stats(self, timeout: float = 30.0) -> dict:
        """
        Poll own server until all clients have shared their data stat.
        Returns { peer_id: std_value } for all clients.
        """
        url = self._get_peer_url(self.peer_id, "/sync/get_all_data_stats")
        start = __import__("time").time()

        while __import__("time").time() - start < timeout:
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("all_received", False):
                                return {
                                    int(k): float(v)
                                    for k, v in data["stats"].items()
                                }
            except Exception as e:
                self.logger.warning(f"get_all_data_stats poll failed: {e}")
            await __import__("asyncio").sleep(0.5)

        self.logger.error("get_all_data_stats: timed out waiting for all peers")
        return {}

    # ── GADFL HISTOGRAM SHARING METHODS ──────────────────────────────────────

    async def share_histogram(self, histogram) -> bool:
        """
        GADFL: Share this client's local speed histogram with ALL peers.

        Called ONCE before training begins (not every round).
        The histogram is a list of num_bins floats summing to ~1.0.
        Optionally includes differential privacy noise applied before calling this.

        Communication cost: 10 floats × 4 bytes = ~80 bytes per peer.
        For 7 peers: ~480 bytes total — negligible compared to model (44KB).

        Args:
            histogram : list or numpy array of floats (normalized speed histogram)

        Returns:
            True if all peers acknowledged, False if any failed
        """
        # Convert to plain Python list for JSON serialization
        if hasattr(histogram, 'tolist'):
            histogram_list = histogram.tolist()
        else:
            histogram_list = [float(v) for v in histogram]

        payload = {
            "peer_id":   self.peer_id,
            "histogram": histogram_list,
        }

        success_count = 0
        for peer_id in self.peer_config.keys():
            url = self._get_peer_url(int(peer_id), "/sync/share_histogram")
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            success_count += 1
                        else:
                            self.logger.warning(
                                f"share_histogram to peer {peer_id} failed: "
                                f"HTTP {resp.status}"
                            )
            except Exception as e:
                self.logger.warning(
                    f"share_histogram to peer {peer_id} failed: "
                    f"{type(e).__name__}: {e}"
                )

        self.logger.info(
            f"[GADFL] Shared histogram ({len(histogram_list)} bins) to "
            f"{success_count}/{len(self.peer_config)} peers"
        )
        return success_count == len(self.peer_config)

    async def get_all_histograms(self, timeout: float = 60.0) -> dict:
        """
        GADFL: Poll own server until all clients have shared their histogram.

        Returns { peer_id (int): histogram (list of floats) } for all clients.
        Polls every 0.5 seconds until all_received=True or timeout.

        Args:
            timeout : max seconds to wait (default 60s)

        Returns:
            dict { int: list } — peer_id → histogram,
            or empty dict if timeout reached
        """
        url   = self._get_peer_url(self.peer_id, "/sync/get_all_histograms")
        start = time.time()

        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("all_received", False):
                                # Convert string keys to int
                                return {
                                    int(k): v
                                    for k, v in data["histograms"].items()
                                }
            except Exception as e:
                self.logger.warning(f"get_all_histograms poll failed: {e}")

            await asyncio.sleep(0.5)

        self.logger.error(
            f"get_all_histograms: timed out after {timeout}s waiting for all peers"
        )
        return {}
