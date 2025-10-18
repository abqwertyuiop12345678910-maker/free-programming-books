#!/usr/bin/env python3
"""
big_code.py

A single-file, self-contained "big" Python codebase showcasing:
- structured modules (via classes/functions)
- configuration & logging
- an in-memory graph engine
- a simple plugin system (dynamic import)
- an async worker pool and scheduler
- a tiny HTTP JSON API (stdlib http.server)
- a lightweight persistence layer (sqlite)
- CLI interface and a demo-run mode
- built-in basic tests / demo at the bottom

No external dependencies required (stdlib only).
Run: python big_code.py --demo
"""

from __future__ import annotations
import argparse
import asyncio
import concurrent.futures
import contextlib
import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import queue
import random
import signal
import sqlite3
import socket
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

LOG = logging.getLogger("big_code")
LOG.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)7s %(name)s: %(message)s"))
LOG.addHandler(_handler)


@dataclasses.dataclass
class Config:
    db_path: str = "big_code.db"
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    worker_count: int = 4
    demo_nodes: int = 100
    demo_edges: int = 300
    plugin_paths: List[str] = dataclasses.field(default_factory=list)

# ---------------------------------------------------------------------------
# Persistence: simple SQLite wrapper
# ---------------------------------------------------------------------------


class DB:
    def __init__(self, path: str):
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None
        LOG.debug("DB init path=%s", path)

    def connect(self):
        if self._conn:
            return
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        LOG.debug("DB connected")
        self._init_schema()

    def _init_schema(self):
        assert self._conn
        cur = self._conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY,
            label TEXT
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY,
            src INTEGER,
            dst INTEGER,
            weight REAL
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            payload TEXT,
            status TEXT,
            created_at REAL
        );
        """
        )
        self._conn.commit()
        LOG.debug("DB schema ensured")

    def insert_nodes(self, nodes: Iterable[Tuple[int, str]]):
        assert self._conn
        cur = self._conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO nodes (id,label) VALUES (?,?)", nodes)
        self._conn.commit()
        LOG.debug("Inserted %d nodes", sum(1 for _ in nodes))

    def insert_edges(self, edges: Iterable[Tuple[int, int, float]]):
        assert self._conn
        cur = self._conn.cursor()
        cur.executemany("INSERT INTO edges (src,dst,weight) VALUES (?,?,?)", edges)
        self._conn.commit()
        LOG.debug("Inserted %d edges", sum(1 for _ in edges))

    def add_task(self, name: str, payload: dict):
        assert self._conn
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO tasks (name,payload,status,created_at) VALUES (?,?,?,?)",
            (name, json.dumps(payload), "pending", time.time()),
        )
        self._conn.commit()
        task_id = cur.lastrowid
        LOG.debug("Added task id=%s name=%s", task_id, name)
        return task_id

    def fetch_pending_tasks(self, limit: int = 10):
        assert self._conn
        cur = self._conn.cursor()
        cur.execute("SELECT id,name,payload FROM tasks WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [(r["id"], r["name"], json.loads(r["payload"])) for r in rows]

    def update_task_status(self, task_id: int, status: str):
        assert self._conn
        cur = self._conn.cursor()
        cur.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        self._conn.commit()
        LOG.debug("Task %s -> %s", task_id, status)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            LOG.debug("DB closed")


# ---------------------------------------------------------------------------
# Graph engine: simple in-memory graph with algorithms
# ---------------------------------------------------------------------------


class Graph:
    def __init__(self):
        # adjacency list: node -> list of (neighbor, weight)
        self._adj: Dict[int, List[Tuple[int, float]]] = {}
        self._lock = threading.RLock()
        LOG.debug("Graph initialized")

    def add_node(self, node_id: int):
        with self._lock:
            self._adj.setdefault(node_id, [])
            LOG.debug("Node added %s", node_id)

    def add_edge(self, src: int, dst: int, weight: float = 1.0):
        with self._lock:
            self._adj.setdefault(src, [])
            self._adj.setdefault(dst, [])
            self._adj[src].append((dst, weight))
            LOG.debug("Edge added %s -> %s (w=%s)", src, dst, weight)

    def neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        with self._lock:
            return list(self._adj.get(node_id, []))

    def nodes(self) -> List[int]:
        with self._lock:
            return list(self._adj.keys())

    def shortest_path_dijkstra(self, src: int) -> Dict[int, float]:
        # computes single-source shortest distances
        with self._lock:
            dist: Dict[int, float] = {n: math.inf for n in self._adj}
            dist[src] = 0.0
            visited: Set[int] = set()
            import heapq

            heap = [(0.0, src)]
            while heap:
                d, node = heapq.heappop(heap)
                if node in visited:
                    continue
                visited.add(node)
                for nbr, w in self._adj.get(node, []):
                    nd = d + w
                    if nd < dist[nbr]:
                        dist[nbr] = nd
                        heapq.heappush(heap, (nd, nbr))
            LOG.debug("Dijkstra from %s done", src)
            return dist

    def connected_components(self) -> List[Set[int]]:
        with self._lock:
            visited: Set[int] = set()
            comps: List[Set[int]] = []

            for node in self._adj:
                if node in visited:
                    continue
                comp = set()
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.add(n)
                    for nbr, _ in self._adj.get(n, []):
                        if nbr not in visited:
                            stack.append(nbr)
                    # also consider reverse edges: scan adjacency; this is O(N) per node worst case
                    for m, edges in self._adj.items():
                        for (nbr, _) in edges:
                            if nbr == n and m not in visited:
                                stack.append(m)
                comps.append(comp)
            LOG.debug("Connected components found: %d", len(comps))
            return comps

    def pagerank(self, iterations: int = 20, damping: float = 0.85) -> Dict[int, float]:
        with self._lock:
            nodes = list(self._adj.keys())
            n = len(nodes)
            if n == 0:
                return {}
            idx = {node: i for i, node in enumerate(nodes)}
            rank = [1.0 / n] * n
            out_counts = [len(self._adj[node]) for node in nodes]

            for it in range(iterations):
                new_rank = [(1.0 - damping) / n] * n
                for i, node in enumerate(nodes):
                    if out_counts[i] == 0:
                        # distribute to all
                        share = damping * rank[i] / n
                        for j in range(n):
                            new_rank[j] += share
                    else:
                        share = damping * rank[i] / out_counts[i]
                        for nbr, _ in self._adj[node]:
                            j = idx[nbr]
                            new_rank[j] += share
                rank = new_rank
            result = {nodes[i]: rank[i] for i in range(n)}
            LOG.debug("PageRank computed (%d iters)", iterations)
            return result


# ---------------------------------------------------------------------------
# Plugin system: dynamically import simple plugins that provide functions
# ---------------------------------------------------------------------------

PluginFunc = Callable[..., Any]


class PluginManager:
    """
    Loads plugins by module path strings.
    Each plugin module may export one or more functions named 'task', 'transform', or 'info'.
    """

    def __init__(self, plugin_paths: Optional[Iterable[str]] = None):
        self._paths: List[str] = list(plugin_paths or [])
        self._plugins: Dict[str, Any] = {}
        LOG.debug("PluginManager init with paths=%s", self._paths)

    def discover(self):
        for path in self._paths:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                # allow importing by dotted path or file path
                if os.path.exists(path):
                    # Add directory to sys.path temporally
                    dirpath = os.path.dirname(os.path.abspath(path)) or "."
                    if dirpath not in sys.path:
                        sys.path.insert(0, dirpath)
                    mod_name = name
                else:
                    mod_name = path
                LOG.debug("Importing plugin %s (mod=%s)", path, mod_name)
                module = importlib.import_module(mod_name)
                self._plugins[mod_name] = module
            except Exception:
                LOG.exception("Failed to import plugin %s", path)

    def get_task_funcs(self) -> List[PluginFunc]:
        funcs: List[PluginFunc] = []
        for mod in self._plugins.values():
            if hasattr(mod, "task") and callable(getattr(mod, "task")):
                funcs.append(getattr(mod, "task"))
        LOG.debug("Discovered task funcs: %d", len(funcs))
        return funcs

    def info(self) -> Dict[str, Dict[str, Any]]:
        ret = {}
        for name, mod in self._plugins.items():
            info = {}
            for attr in ("__version__", "__author__", "description"):
                if hasattr(mod, attr):
                    info[attr] = getattr(mod, attr)
            ret[name] = info
        return ret


# ---------------------------------------------------------------------------
# Async worker pool and scheduler
# ---------------------------------------------------------------------------


class TaskWorkerPool:
    """
    A simple worker pool that polls the DB for pending tasks and executes them.
    Uses concurrent.futures.ThreadPoolExecutor for isolation.
    """

    def __init__(self, db: DB, plugin_manager: PluginManager, max_workers: int = 4, poll_interval: float = 1.0):
        self.db = db
        self.plugin_manager = plugin_manager
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = threading.Event()
        self.poll_interval = poll_interval
        LOG.debug("TaskWorkerPool initialized max_workers=%d", max_workers)

    def start(self):
        LOG.info("Starting TaskWorkerPool with %d workers", self.max_workers)
        self._stop_event.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self, timeout: Optional[float] = None):
        LOG.info("Stopping TaskWorkerPool")
        self._stop_event.set()
        self.executor.shutdown(wait=True, timeout=timeout)
        LOG.info("WorkerPool stopped")

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                tasks = self.db.fetch_pending_tasks(limit=self.max_workers * 2)
                if not tasks:
                    time.sleep(self.poll_interval)
                    continue
                for task_id, name, payload in tasks:
                    # mark running immediately to avoid duplication
                    self.db.update_task_status(task_id, "running")
                    self.executor.submit(self._run_task, task_id, name, payload)
            except Exception:
                LOG.exception("Error in worker loop")
                time.sleep(self.poll_interval)

    def _run_task(self, task_id: int, name: str, payload: dict):
        try:
            # Look up plugin tasks by name; fallback to plugin.task interface
            executed = False
            for func in self.plugin_manager.get_task_funcs():
                try:
                    # callable signature variations handled via inspect
                    sig = inspect.signature(func)
                    kwargs = {}
                    if "payload" in sig.parameters:
                        kwargs["payload"] = payload
                    if "db" in sig.parameters:
                        kwargs["db"] = self.db
                    if "task_id" in sig.parameters:
                        kwargs["task_id"] = task_id
                    LOG.debug("Executing plugin func %s for task %s", func, task_id)
                    res = func(name=name, **kwargs) if "name" in sig.parameters else func(**kwargs)
                    LOG.info("Task %s executed plugin result: %s", task_id, res)
                    executed = True
                except Exception:
                    LOG.exception("Plugin function failed for task %s", task_id)
            if not executed:
                # default task execution
                LOG.info("Executing default handler for task %s name=%s", task_id, name)
                time.sleep(0.1 + random.random() * 0.5)  # simulate work
                LOG.debug("Default task %s done", task_id)
            self.db.update_task_status(task_id, "done")
        except Exception:
            LOG.exception("Task %s failed", task_id)
            self.db.update_task_status(task_id, "failed")


# ---------------------------------------------------------------------------
# Tiny HTTP API (stdlib-based)
# ---------------------------------------------------------------------------


def _json_response(obj: Any, status: int = 200) -> Tuple[bytes, str]:
    body = json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))).encode("utf-8")
    return body, "application/json; charset=utf-8"


class SimpleAPIHandler(BaseHTTPRequestHandler):
    server_version = "BigCodeHTTP/0.1"

    def _send(self, body: bytes, content_type: str = "application/json; charset=utf-8", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def graph(self) -> Graph:
        return self.server.graph  # type: ignore

    @property
    def db(self) -> DB:
        return self.server.db  # type: ignore

    @property
    def plugin_manager(self) -> PluginManager:
        return self.server.plugin_manager  # type: ignore

    def _parse_path(self):
        parts = self.path.split("?", 1)
        return parts[0], (parts[1] if len(parts) > 1 else "")

    def do_GET(self):
        try:
            path, _ = self._parse_path()
            if path == "/":
                body, ctype = _json_response({"status": "ok", "endpoints": ["/graph/nodes", "/graph/pagerank", "/tasks/add?name=..."]})
                self._send(body, ctype, status=200)
            elif path == "/graph/nodes":
                nodes = self.graph.nodes()
                body, ctype = _json_response({"count": len(nodes), "nodes": nodes})
                self._send(body, ctype)
            elif path == "/graph/pagerank":
                pr = self.graph.pagerank()
                body, ctype = _json_response({"pagerank": pr})
                self._send(body, ctype)
            elif path == "/plugins":
                body, ctype = _json_response({"plugins": self.plugin_manager.info()})
                self._send(body, ctype)
            else:
                self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
        except Exception:
            LOG.exception("Error handling GET")
            self._send(b"Internal Server Error", "text/plain; charset=utf-8", status=500)

    def do_POST(self):
        try:
            path, _ = self._parse_path()
            if path == "/tasks/add":
                length = int(self.headers.get("Content-Length", 0))
                payload_raw = self.rfile.read(length) if length else b""
                payload = json.loads(payload_raw.decode("utf-8")) if payload_raw else {}
                name = payload.get("name") or "unnamed"
                task_id = self.db.add_task(name, payload)
                body, ctype = _json_response({"task_id": task_id})
                self._send(body, ctype, status=201)
            else:
                self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
        except Exception:
            LOG.exception("Error handling POST")
            self._send(b"Internal Server Error", "text/plain; charset=utf-8", status=500)

    def log_message(self, format, *args):
        # route stdlib HTTP logs to LOG
        LOG.info("%s - - %s", self.client_address[0], format % args)


class BigHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, graph: Graph, db: DB, plugin_manager: PluginManager):
        super().__init__(server_address, RequestHandlerClass)
        self.graph = graph
        self.db = db
        self.plugin_manager = plugin_manager


def run_http_server(cfg: Config, graph: Graph, db: DB, plugin_manager: PluginManager, shutdown_event: threading.Event):
    server = BigHTTPServer((cfg.http_host, cfg.http_port), SimpleAPIHandler, graph, db, plugin_manager)
    LOG.info("HTTP server listening on http://%s:%d", cfg.http_host, cfg.http_port)

    def _serve():
        with contextlib.suppress(Exception):
            server.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    try:
        shutdown_event.wait()
    finally:
        LOG.info("Shutting down HTTP server")
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Utilities: demo data generator
# ---------------------------------------------------------------------------


def generate_demo_graph(graph: Graph, cfg: Config, db: Optional[DB] = None):
    LOG.info("Generating demo graph nodes=%d edges=%d", cfg.demo_nodes, cfg.demo_edges)
    # add nodes
    for i in range(cfg.demo_nodes):
        graph.add_node(i)
    edges = set()
    for _ in range(cfg.demo_edges):
        a = random.randrange(cfg.demo_nodes)
        b = random.randrange(cfg.demo_nodes)
        if a == b:
            continue
        w = round(random.random() * 10, 3)
        graph.add_edge(a, b, w)
        edges.add((a, b, w))
    # persist to DB if available
    if db:
        db.connect()
        nodes = [(i, f"node-{i}") for i in range(cfg.demo_nodes)]
        db.insert_nodes(nodes)
        db.insert_edges(edges)
    LOG.info("Demo graph generated")


# ---------------------------------------------------------------------------
# CLI and main runner
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="big_code demo CLI")
    p.add_argument("--db", help="path to sqlite db", default="big_code.db")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8000, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--demo", action="store_true", help="run demo")
    p.add_argument("--seed", default=None, type=int, help="random seed")
    p.add_argument("--plugins", nargs="*", default=[], help="plugin module paths to load")
    return p.parse_args(argv)


def graceful_shutdown_event() -> threading.Event:
    ev = threading.Event()

    def _sig(signum, frame):
        LOG.info("Received signal %s, shutting down...", signum)
        ev.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    return ev


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
    cfg = Config(db_path=args.db, http_host=args.host, http_port=args.port, worker_count=args.workers, plugin_paths=args.plugins)
    LOG.info("Starting big_code with config: %s", cfg)

    db = DB(cfg.db_path)
    db.connect()

    graph = Graph()
    plugin_manager = PluginManager(cfg.plugin_paths)
    plugin_manager.discover()

    if args.demo:
        generate_demo_graph(graph, cfg, db)
        # add a few demo tasks
        for i in range(10):
            db.add_task("demo_task", {"i": i, "info": f"demo-{i}"})

    shutdown = graceful_shutdown_event()

    # Start HTTP server
    run_http_server(cfg, graph, db, plugin_manager, shutdown)

    pool = TaskWorkerPool(db, plugin_manager, max_workers=cfg.worker_count)
    pool.start()

    try:
        # main thread waits for shutdown
        while not shutdown.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt received")
    finally:
        pool.stop(timeout=5)
        db.close()
        LOG.info("Exited")


# ---------------------------------------------------------------------------
# Basic unit-test style demos (self-run)
# ---------------------------------------------------------------------------

def _self_test_graph():
    g = Graph()
    for i in range(5):
        g.add_node(i)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 2.0)
    g.add_edge(2, 3, 3.0)
    g.add_edge(3, 4, 4.0)
    g.add_edge(4, 0, 5.0)
    dist = g.shortest_path_dijkstra(0)
    assert dist[0] == 0.0
    assert dist[1] == 1.0
    LOG.info("Graph Dijkstra test passed")
    pr = g.pagerank(iterations=5)
    assert isinstance(pr, dict) and len(pr) == 5
    LOG.info("Graph PageRank test passed")
    comps = g.connected_components()
    assert any(0 in c for c in comps)
    LOG.info("Graph connected components test passed")


def _self_test_db(tmp_path: str):
    path = os.path.join(tmp_path, "test.db")
    db = DB(path)
    db.connect()
    db.insert_nodes([(1, "a"), (2, "b")])
    db.insert_edges([(1, 2, 0.5)])
    tid = db.add_task("t1", {"x": 1})
    tasks = db.fetch_pending_tasks()
    assert any(tid == t[0] for t in tasks)
    db.update_task_status(tid, "done")
    db.close()
    LOG.info("DB tests passed")


def run_self_tests():
    LOG.info("Running self-tests")
    _self_test_graph()
    tmpdir = os.path.abspath(".")
    _self_test_db(tmpdir)
    LOG.info("All self-tests passed")


# ---------------------------------------------------------------------------
# If invoked directly, run demo or tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run as script, choose demo by default if --demo provided, else run tests
    if "--demo" in sys.argv:
        main(sys.argv[1:])
    else:
        # run quick self-tests
        logging.getLogger().setLevel(logging.INFO)
        run_self_tests()
