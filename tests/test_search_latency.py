from pathlib import Path


MAIN_SOURCE = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(
    encoding="utf-8"
)


def test_fast_search_prioritizes_exact_and_prefix_before_fuzzy():
    search_source = MAIN_SOURCE[
        MAIN_SOURCE.index("def _get_movies_fast_sql_nocache"):
        MAIN_SOURCE.index("def get_movie_by_imdb_id", MAIN_SOURCE.index("def _get_movies_fast_sql_nocache"))
    ]
    assert "exact_prefix_sql = " in search_source
    assert "regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g') = %s" in search_source
    assert "if not results:" in search_source
    assert "LIKE %s" in search_source
    assert "SIMILARITY(" in search_source
    assert "ORDER BY CASE" in search_source
    assert search_source.index("exact_prefix_sql = ") < search_source.index("stage = \"fuzzy\"")
    assert "OR SIMILARITY" not in search_source


def test_search_bounds_google_fallback_and_joins_delivery_data():
    assert "GOOGLE_SUGGESTION_TIMEOUT_SECONDS = 1.5" in MAIN_SOURCE
    assert "get_google_title_suggestions_with_timeout" in MAIN_SOURCE
    assert "def get_movie_delivery_data(movie_id)" in MAIN_SOURCE
    assert "LEFT JOIN movie_files AS mf" in MAIN_SOURCE
