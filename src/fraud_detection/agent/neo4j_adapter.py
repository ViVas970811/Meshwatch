"""Neo4j adapter for the agent's graph-traversal tool (Phase 5).

The plan (page 10) specifies that ``explore_graph_neighborhood`` should
walk a Neo4j graph -- the docker-compose stack already runs a
``neo4j:5-community`` container. This module provides a thin
``.neighbors(node_id)``-shaped wrapper so the existing tool keeps
working unchanged when callers pass an instance to ``AgentDeps(graph=...)``.

Design rules (matching the rest of Phase 4/5):

* Optional dependency -- the ``neo4j`` driver is in the ``[agent]``
  extras; absence shouldn't break imports of the rest of the agent.
* Graceful fallback -- if the database isn't reachable we mark the
  adapter ``connected=False`` and ``.neighbors`` returns an empty list
  rather than crashing the investigation loop.
* No state mutation in ``.neighbors`` -- read-only Cypher, side-effect
  free, safe to call from concurrent agent runs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fraud_detection.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover -- typing only
    pass


log = get_logger(__name__)


# Default Cypher used by ``neighbors()``. Pulls peer cards that share a
# device or address with the seed card -- which is exactly the
# ``card-card`` projection Phase 2 builds in PyG.
DEFAULT_NEIGHBOR_QUERY = (
    "MATCH (c:Card {id: $node_id})-[:SHARED_DEVICE|SHARED_ADDRESS]-(peer:Card) "
    "RETURN DISTINCT peer.id AS peer_id "
    "LIMIT $limit"
)


class Neo4jGraphAdapter:
    """``.neighbors(node)``-compatible adapter backed by Neo4j.

    Pluggable into :class:`fraud_detection.agent.AgentDeps` via the
    ``graph`` kwarg, alongside any other ``.neighbors``-shaped object
    (networkx graphs, dicts, ...).

    Parameters
    ----------
    uri
        bolt:// URL of the Neo4j server. Defaults to ``NEO4J_URI`` env
        var, then ``bolt://localhost:7687``.
    user / password
        Auth credentials. Default to ``NEO4J_USER`` / ``NEO4J_PASSWORD``
        env vars.
    database
        Target database. Defaults to ``neo4j``.
    neighbor_query
        Cypher to use for :meth:`neighbors`. Must take ``$node_id`` and
        ``$limit`` parameters and return a single column of peer ids.
    default_limit
        Default ``LIMIT`` clause value.
    """

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        neighbor_query: str = DEFAULT_NEIGHBOR_QUERY,
        default_limit: int = 100,
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.database = database
        self.neighbor_query = neighbor_query
        self.default_limit = int(default_limit)

        self._driver: Any | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open a driver and verify connectivity.

        Returns ``True`` on success. On failure we log a warning and
        leave the adapter in disconnected mode -- :meth:`neighbors`
        will then yield an empty list rather than raise.
        """
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except Exception as exc:
            log.warning("neo4j_module_unavailable", error=str(exc))
            self._driver = None
            self._connected = False
            return False
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Hit the server cheaply to make sure the URL is reachable.
            self._driver.verify_connectivity()
            self._connected = True
            log.info("neo4j_connected", uri=self.uri, database=self.database)
        except Exception as exc:
            log.warning("neo4j_connect_failed_using_disconnected", error=str(exc))
            try:
                if self._driver is not None:
                    self._driver.close()
            except Exception:
                pass
            self._driver = None
            self._connected = False
        return self._connected

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def neighbors(self, node_id: int | str, *, limit: int | None = None) -> list[int | str]:
        """Return peer-card ids one hop away from ``node_id``.

        Mirrors the ``.neighbors`` interface of ``networkx.Graph`` so
        the agent's :func:`explore_graph_neighborhood` tool can use
        either backend interchangeably.
        """
        if not self._connected or self._driver is None:
            return []
        cap = int(limit if limit is not None else self.default_limit)
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    self.neighbor_query,
                    node_id=node_id,
                    limit=cap,
                )
                # ``record["peer_id"]`` matches the alias in the default
                # query; if the caller passed a custom query we just
                # take the first column so we don't have to teach the
                # adapter the alias.
                rows = result.data()
                if not rows:
                    return []
                first_key = next(iter(rows[0].keys()))
                return [r[first_key] for r in rows if r.get(first_key) is not None]
        except Exception as exc:
            log.warning("neo4j_neighbors_query_failed", error=str(exc), node_id=str(node_id))
            return []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as exc:
                log.warning("neo4j_close_failed", error=str(exc))
            finally:
                self._driver = None
                self._connected = False

    def __enter__(self) -> Neo4jGraphAdapter:
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["DEFAULT_NEIGHBOR_QUERY", "Neo4jGraphAdapter"]
