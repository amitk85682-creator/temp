import pytest

import db_migrations


class MigrationCursor:
    def __init__(self, connection):
        self.connection = connection
        self.query = ''

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.query = query
        self.connection.queries.append((query, params))

    def fetchall(self):
        return [(1,)] if 'SELECT version' in self.query else []


class MigrationConnection:
    def __init__(self):
        self.queries = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return MigrationCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_migration_versions_are_unique_and_ordered():
    versions = [version for version, _ in db_migrations.MIGRATIONS]
    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))
    assert versions[-1] == 10


def test_identity_migration_adds_structural_episode_and_evidence_tables():
    connection = MigrationConnection()

    db_migrations._migration_10(connection)

    queries = "\n".join(query for query, _ in connection.queries)
    assert "CREATE TABLE IF NOT EXISTS seasons" in queries
    assert "CREATE TABLE IF NOT EXISTS episodes" in queries
    assert "CREATE TABLE IF NOT EXISTS file_episodes" in queries
    assert "CREATE TABLE IF NOT EXISTS file_seasons" in queries
    assert "CREATE TABLE IF NOT EXISTS ingestion_evidence" in queries
    assert "CREATE TABLE IF NOT EXISTS content_identity_repair_archive" in queries
    assert "ensure_file_episode_parent_match" in queries
    assert "ensure_file_season_parent_match" in queries


def test_already_applied_migrations_are_skipped(monkeypatch):
    connection = MigrationConnection()
    called = []
    monkeypatch.setattr(db_migrations.psycopg2, 'connect', lambda _url: connection)
    monkeypatch.setattr(
        db_migrations,
        'MIGRATIONS',
        ((1, lambda _conn: called.append(1)),),
    )

    db_migrations.run_migrations('synthetic://database')

    assert called == []
    assert connection.closed


def test_successful_migration_is_recorded_and_connection_closes(monkeypatch):
    connection = MigrationConnection()
    monkeypatch.setattr(db_migrations.psycopg2, 'connect', lambda _url: connection)
    monkeypatch.setattr(
        db_migrations,
        'MIGRATIONS',
        ((2, lambda _conn: None),),
    )

    db_migrations.run_migrations('synthetic://database')

    assert connection.commits == 2
    assert any('INSERT INTO schema_migrations' in query for query, _ in connection.queries)
    assert connection.closed


def test_failed_migration_rolls_back_and_is_not_recorded(monkeypatch):
    connection = MigrationConnection()
    monkeypatch.setattr(db_migrations.psycopg2, 'connect', lambda _url: connection)
    monkeypatch.setattr(
        db_migrations,
        'MIGRATIONS',
        ((2, lambda _conn: (_ for _ in ()).throw(RuntimeError('synthetic failure'))),),
    )

    with pytest.raises(RuntimeError, match='synthetic failure'):
        db_migrations.run_migrations('synthetic://database')

    assert connection.rollbacks == 1
    assert not any(
        'INSERT INTO schema_migrations' in query for query, _ in connection.queries
    )
    assert connection.closed
