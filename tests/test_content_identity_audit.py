from pathlib import Path

from content_identity_audit import classify_identity_rows


def test_episode_noise_maps_to_one_canonical_series_candidate():
    movies = [
        {
            "id": 10,
            "title": "Reacher",
            "year": 2022,
            "content_type": "Web Series",
            "imdb_id": "tt1111111",
            "tmdb_id": 100,
        },
        {
            "id": 11,
            "title": "Reacher S01E02-08",
            "year": 2022,
            "content_type": "Movie",
            "imdb_id": None,
            "tmdb_id": None,
        },
    ]

    report = classify_identity_rows(movies, {11: [{"id": 50}, {"id": 51}]})

    assert len(report) == 1
    assert report[0]["wrong_row_id"] == 11
    assert report[0]["possible_canonical_row_id"] == 10
    assert report[0]["files_affected"] == 2
    assert report[0]["confidence"] >= 0.9
    assert "archive row" in report[0]["proposed_action"]


def test_audit_does_not_suggest_merging_provider_conflicts_or_different_years():
    movies = [
        {
            "id": 1,
            "title": "The Office",
            "year": 2005,
            "content_type": "Web Series",
            "imdb_id": "tt0000001",
            "tmdb_id": 1,
        },
        {
            "id": 2,
            "title": "The Office",
            "year": 1995,
            "content_type": "Web Series",
            "imdb_id": "tt0000002",
            "tmdb_id": 2,
        },
        {
            "id": 3,
            "title": "The Office S01E02",
            "year": 2005,
            "content_type": "Movie",
            "imdb_id": "tt0000003",
            "tmdb_id": 3,
        },
    ]

    report = classify_identity_rows(movies, {})

    assert len(report) == 1
    assert report[0]["possible_canonical_row_id"] == ""
    assert report[0]["confidence"] < 0.9


def test_repair_archives_before_retiring_rows_and_remaps_episode_links():
    source = (Path(__file__).resolve().parents[1] / "content_identity_audit.py").read_text(
        encoding="utf-8"
    )
    repair = source[
        source.index("def apply_repair"):
        source.index("def restore_repair")
    ]

    assert "related_rows[\"__structured__\"] = structured" in repair
    assert "INSERT INTO file_seasons" in repair
    assert "INSERT INTO file_episodes" in repair
    assert "canonical parent is now missing, conflicting, or ambiguous" in repair
    assert repair.index("INSERT INTO content_identity_repair_archive") < repair.index(
        "DELETE FROM movies WHERE id = %s"
    )
    assert "relationship remains on" in repair


def test_repair_classification_does_not_merge_when_parent_type_is_not_series():
    movies = [
        {
            "id": 20,
            "title": "Reacher",
            "year": 2022,
            "content_type": "Movie",
            "imdb_id": None,
            "tmdb_id": None,
        },
        {
            "id": 21,
            "title": "Reacher S01E02",
            "year": 2022,
            "content_type": "Movie",
            "imdb_id": None,
            "tmdb_id": None,
        },
    ]

    report = classify_identity_rows(movies, {21: [{"id": 72}]})

    assert report[0]["possible_canonical_row_id"] == ""
    assert report[0]["confidence"] < 0.9


def test_apply_only_attempts_high_confidence_rows_and_skips_candidate_exceptions():
    source = (Path(__file__).resolve().parents[1] / "content_identity_audit.py").read_text(
        encoding="utf-8"
    )
    apply_loop = source[
        source.index("if args.apply:"):
        source.index("finally:", source.index("if args.apply:"))
    ]

    assert 'candidate["confidence"] < 0.9' in apply_loop
    assert "if conn is None or conn.closed:" in apply_loop
    assert "conn.commit()" in apply_loop
    assert "except Exception as exc:" in apply_loop
    assert 'continue' in apply_loop
    assert "conn.rollback()" in apply_loop
