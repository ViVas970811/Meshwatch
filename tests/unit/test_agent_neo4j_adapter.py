"""Tests for :class:`Neo4jGraphAdapter` (Phase 5).

We don't bring up a real Neo4j -- the adapter must work against a
mocked driver so unit tests stay hermetic. The contract we care about:

* ``connect()`` returns False (not raise) when the driver isn't installed
  or the server isn't reachable.
* ``neighbors()`` returns an empty list when disconnected, never raises.
* ``neighbors()`` returns the peer-id column when connected.
* ``close()`` is idempotent.
* The adapter is interchangeable with the ``.neighbors()`` shape used
  by the rest of the agent (so ``explore_graph_neighborhood`` doesn't
  care which backend it gets).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``patch("neo4j.GraphDatabase.driver", ...)`` resolves the import path; without
# the real driver installed it would error out before any tests run.
pytest.importorskip("neo4j")

from fraud_detection.agent.neo4j_adapter import (
    DEFAULT_NEIGHBOR_QUERY,
    Neo4jGraphAdapter,
)
from fraud_detection.agent.tools import (
    explore_graph_neighborhood,
)

# ---------------------------------------------------------------------------
# connect / close
# ---------------------------------------------------------------------------


def test_connect_returns_false_when_driver_unavailable(monkeypatch) -> None:
    """If ``neo4j`` Python driver isn't importable we degrade to disconnected."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "neo4j":
            raise ImportError("neo4j not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    adapter = Neo4jGraphAdapter(uri="bolt://nowhere:0")
    assert adapter.connect() is False
    assert adapter.connected is False


def test_connect_returns_false_when_server_unreachable() -> None:
    """A bolt:// URL pointing at nothing should not crash the agent."""
    adapter = Neo4jGraphAdapter(uri="bolt://127.0.0.1:1")
    # connect() may succeed at constructing the driver but verify_connectivity()
    # should fail; either way the adapter ends up disconnected.
    ok = adapter.connect()
    assert ok is False
    assert adapter.connected is False


def test_close_is_idempotent_on_disconnected_adapter() -> None:
    adapter = Neo4jGraphAdapter(uri="bolt://nowhere:0")
    adapter.close()
    adapter.close()  # second close should not raise
    assert adapter.connected is False


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------


def test_neighbors_returns_empty_when_disconnected() -> None:
    adapter = Neo4jGraphAdapter(uri="bolt://nowhere:0")
    # Never connected -- list should be empty, not raise.
    assert adapter.neighbors(123) == []


def _mock_driver(rows: list[dict[str, Any]]) -> Any:
    """Build a MagicMock that quacks like neo4j.GraphDatabase.driver()."""
    driver = MagicMock(name="MockDriver")
    session_cm = MagicMock(name="SessionCM")
    session = MagicMock(name="Session")
    result = MagicMock(name="Result")
    result.data.return_value = rows
    session.run.return_value = result
    session_cm.__enter__.return_value = session
    session_cm.__exit__.return_value = False
    driver.session.return_value = session_cm
    driver.verify_connectivity.return_value = None
    driver.close.return_value = None
    return driver


def test_neighbors_returns_peer_ids_from_default_query() -> None:
    rows = [{"peer_id": 1}, {"peer_id": 2}, {"peer_id": 3}]
    with patch("neo4j.GraphDatabase.driver", return_value=_mock_driver(rows)):
        adapter = Neo4jGraphAdapter(uri="bolt://localhost:7687")
        assert adapter.connect() is True
        peers = adapter.neighbors(99)
        assert peers == [1, 2, 3]


def test_neighbors_passes_node_id_and_limit_params() -> None:
    rows = [{"peer_id": 7}]
    driver = _mock_driver(rows)
    with patch("neo4j.GraphDatabase.driver", return_value=driver):
        adapter = Neo4jGraphAdapter(uri="bolt://localhost:7687", default_limit=50)
        adapter.connect()
        adapter.neighbors(99, limit=10)
        # Verify the Cypher call args.
        session = driver.session.return_value.__enter__.return_value
        session.run.assert_called_once()
        args, kwargs = session.run.call_args
        assert args[0] == DEFAULT_NEIGHBOR_QUERY
        assert kwargs == {"node_id": 99, "limit": 10}


def test_neighbors_handles_query_failure_gracefully() -> None:
    driver = _mock_driver([])
    session = driver.session.return_value.__enter__.return_value
    session.run.side_effect = RuntimeError("cypher boom")
    with patch("neo4j.GraphDatabase.driver", return_value=driver):
        adapter = Neo4jGraphAdapter(uri="bolt://localhost:7687")
        adapter.connect()
        # Must not raise.
        assert adapter.neighbors(99) == []


def test_neighbors_works_with_custom_query_using_first_column() -> None:
    """Custom queries can rename the peer column; adapter takes whatever is first."""
    rows = [{"peer.id": "card-1"}, {"peer.id": "card-2"}]
    driver = _mock_driver(rows)
    with patch("neo4j.GraphDatabase.driver", return_value=driver):
        adapter = Neo4jGraphAdapter(
            uri="bolt://localhost:7687",
            neighbor_query="MATCH ... RETURN peer.id",
        )
        adapter.connect()
        peers = adapter.neighbors("card-99")
        assert peers == ["card-1", "card-2"]


# ---------------------------------------------------------------------------
# Integration with explore_graph_neighborhood (the consuming tool)
# ---------------------------------------------------------------------------


def test_adapter_is_drop_in_for_explore_graph_neighborhood() -> None:
    """The agent's tool must accept the adapter without any code changes."""
    rows = [{"peer_id": "p1"}, {"peer_id": "p2"}, {"peer_id": "p3"}]
    with patch("neo4j.GraphDatabase.driver", return_value=_mock_driver(rows)):
        adapter = Neo4jGraphAdapter(uri="bolt://localhost:7687")
        adapter.connect()
        out = explore_graph_neighborhood(
            transaction_id="t-1",
            card_id="card-99",
            graph=adapter,
            n_hops=1,
        )
        assert out["status"] == "ok"
        assert out["n_unique_neighbors"] == 3
        assert set(out["neighbors_by_hop"][0]) == {"p1", "p2", "p3"}


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


def test_context_manager_connects_and_closes() -> None:
    rows = [{"peer_id": 1}]
    with patch("neo4j.GraphDatabase.driver", return_value=_mock_driver(rows)):
        with Neo4jGraphAdapter(uri="bolt://localhost:7687") as adapter:
            assert adapter.connected
        # __exit__ should have called close().
        assert not adapter.connected


# ---------------------------------------------------------------------------
# env-var defaults
# ---------------------------------------------------------------------------


def test_env_var_defaults(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://envhost:7777")
    monkeypatch.setenv("NEO4J_USER", "alice")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    adapter = Neo4jGraphAdapter()
    assert adapter.uri == "bolt://envhost:7777"
    assert adapter.user == "alice"
    assert adapter.password == "secret"


def test_explicit_args_override_env(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://envhost:7777")
    adapter = Neo4jGraphAdapter(uri="bolt://override:9999")
    assert adapter.uri == "bolt://override:9999"


# ---------------------------------------------------------------------------
# Pytest collection hook -- skip everything if neo4j driver missing
# ---------------------------------------------------------------------------

pytest.importorskip("neo4j")
