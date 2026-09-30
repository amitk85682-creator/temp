"""Deterministic mocked benchmark: run with python tests/benchmark_search_latency.py."""

import importlib
import math
import os
import statistics
import sys
import time
from pathlib import Path

import psycopg2.pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POOL_DELAY_SECONDS = 0.001
SQL_ROUND_TRIP_SECONDS = 0.003
SAMPLES = 40
MOVIES = {
    "dune": (1, "Dune", "url", "file", None, None, 2021, "Sci-Fi"),
    "reacher": (2, "Reacher", "url", "file", None, None, 2022, "Action"),
}
COUNTERS = {"connections": 0, "sql_round_trips": 0}


class _ImportOnlyPool:
    def __init__(self, *_args, **_kwargs):
        pass

    def getconn(self):
        raise AssertionError("the benchmark uses a mocked connection")

    def putconn(self, _connection):
        pass

    def closeall(self):
        pass


os.environ.setdefault("TELEGRAM_BOT_TOKEN", "synthetic-benchmark-token")
os.environ.setdefault("DATABASE_URL", "postgresql://synthetic.invalid/benchmark")
os.environ.setdefault("TMDB_API_KEY", "synthetic-benchmark-key")
os.environ.setdefault("UPDATE_SECRET_CODE", "synthetic-benchmark-secret")
_pool_class = psycopg2.pool.ThreadedConnectionPool
psycopg2.pool.ThreadedConnectionPool = _ImportOnlyPool
try:
    bot = importlib.import_module("main")
finally:
    psycopg2.pool.ThreadedConnectionPool = _pool_class
bot.logger.setLevel("CRITICAL")


class _Cursor:
    def __init__(self):
        self.sql = ""
        self.params = ()

    def execute(self, sql, params):
        self.sql = sql
        self.params = params
        COUNTERS["sql_round_trips"] += 1
        time.sleep(SQL_ROUND_TRIP_SECONDS)

    def fetchall(self):
        normalized = str(self.params[0]).casefold()
        if "movie_aliases" in self.sql:
            return []
        if "similarity(" in self.sql.casefold():
            if normalized == "rechar":
                return [MOVIES["reacher"]]
            return []
        if normalized == "dune":
            return [MOVIES["dune"]]
        return []

    def close(self):
        pass


class _Connection:
    def cursor(self):
        return _Cursor()


def _get_connection():
    COUNTERS["connections"] += 1
    time.sleep(POOL_DELAY_SECONDS)
    return _Connection()


bot.get_db_connection = _get_connection
bot.close_db_connection = lambda _connection: None


def _previous_staged_search(query, limit=5):
    """Model the pre-combination exact -> prefix -> title/alias fuzzy stages."""
    normalized = bot._normalize_search_text(query)
    cache_key = ("catalog_search", normalized, int(limit))
    cached = bot.search_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    connection = bot.get_db_connection()
    cursor = connection.cursor()
    statements = (
        ("exact", False),
        ("prefix", False),
        ("fuzzy title", True),
        ("fuzzy alias", True),
    )
    results = []
    for name, fuzzy in statements:
        sql = "SELECT movie title WHERE "
        if name == "exact":
            sql += "normalized_title = %s"
        elif name == "prefix":
            sql += "normalized_title LIKE %s"
        elif name == "fuzzy title":
            sql += "SIMILARITY(normalized_title, %s)"
        else:
            sql += "SIMILARITY(movie_aliases.alias, %s)"
        cursor.execute(sql, (normalized,))
        results = cursor.fetchall()
        if results:
            break
    cursor.close()
    bot.close_db_connection(connection)
    bot.search_cache.set(cache_key, tuple(tuple(row[:8]) for row in results))
    return list(results)


def _percentile95(samples):
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _run_scenario(search_function, scenario):
    samples = []
    connections = 0
    round_trips = 0
    for index in range(SAMPLES):
        bot.search_cache.clear()
        COUNTERS["connections"] = 0
        COUNTERS["sql_round_trips"] = 0
        if scenario == "normalized_case_variation":
            search_function("Dune")
            COUNTERS["connections"] = 0
            COUNTERS["sql_round_trips"] = 0
            query = "d u n e"
        else:
            query = {
                "exact_popular_title": "Dune",
                "partial_requiring_fuzzy": "Rechar",
                "no_result": "zzzxq",
            }[scenario]
        started = time.perf_counter()
        search_function(query)
        samples.append((time.perf_counter() - started) * 1000)
        connections += COUNTERS["connections"]
        round_trips += COUNTERS["sql_round_trips"]
    return (
        statistics.median(samples),
        _percentile95(samples),
        connections / SAMPLES,
        round_trips / SAMPLES,
    )


def main():
    scenarios = (
        "exact_popular_title",
        "normalized_case_variation",
        "partial_requiring_fuzzy",
        "no_result",
    )
    print(
        "Mock delays: connection acquisition={:.1f} ms; SQL round trip={:.1f} ms; "
        "samples={}".format(
            POOL_DELAY_SECONDS * 1000,
            SQL_ROUND_TRIP_SECONDS * 1000,
            SAMPLES,
        )
    )
    print("Scenario | before median/p95 ms | after median/p95 ms | DB conns/query round trips")
    for scenario in scenarios:
        before = _run_scenario(_previous_staged_search, scenario)
        after = _run_scenario(bot.get_movies_fast_sql, scenario)
        print(
            "{} | {:.3f}/{:.3f} | {:.3f}/{:.3f} | "
            "before {:.1f}/{:.1f}; after {:.1f}/{:.1f}".format(
                scenario,
                before[0],
                before[1],
                after[0],
                after[1],
                before[2],
                before[3],
                after[2],
                after[3],
            )
        )


if __name__ == "__main__":
    main()
