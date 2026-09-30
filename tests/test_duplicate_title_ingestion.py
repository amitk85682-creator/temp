from pathlib import Path


MAIN = (Path(__file__).resolve().parents[1] / "main.py").read_text(
    encoding="utf-8"
)


def test_batch_id_resolves_identity_collision_without_using_title():
    section = MAIN[
        MAIN.index("async def batch_id_command"):
        MAIN.index("async def", MAIN.index("async def batch_id_command") + 10)
    ]
    assert "_find_movie_by_provider_identity" in section
    assert "tmdb_id = await run_async" in section
    assert "if existing_movie_id:" in section
    assert "INSERT INTO movies" in section
    assert "title = %s" in section


def test_title_conflicts_are_not_used_as_an_upsert_key():
    assert "ON CONFLICT (title)" not in MAIN
    assert "def _find_movie_by_provider_identity" in MAIN


def test_automatic_ingestion_gates_creation_before_movie_insert():
    section = MAIN[
        MAIN.index("async def _core_movie_processor"):
        MAIN.index("# 📤 _pm_save_file", MAIN.index("async def _core_movie_processor"))
    ]

    assert "can_create_canonical_content(" in section
    assert "status=\"pending_review\"" in section
    assert section.index("can_create_canonical_content(") < section.index("INSERT INTO movies")
    assert "raw_filename: str = \"\"" in section
    assert "file_unique_id: Optional[str] = None" in section
    assert "resolve_raw_content_identity(" in section
    assert '("evidence_engine", ai_data.get("title", ""))' not in section


def test_episode_file_uses_unique_existing_series_before_any_storage_copy():
    resolver = MAIN[
        MAIN.index("def _resolve_episode_file_parent"):
        MAIN.index("async def _pm_save_file")
    ]
    saver = MAIN[
        MAIN.index("async def _pm_save_file"):
        MAIN.index("async def pm_file_listener")
    ]

    assert "require_series=True" in resolver
    assert "Multiple canonical series rows" in resolver
    assert saver.index("_resolve_episode_file_parent(") < saver.index("message.copy(")
    assert "_link_movie_file_episodes(" in saver


def test_episode_links_and_duplicate_file_upsert_are_idempotent():
    link_section = MAIN[
        MAIN.index("def _link_movie_file_episodes"):
        MAIN.index("async def _core_movie_processor")
    ]
    upsert_section = MAIN[
        MAIN.index("def upsert_movie_file"):
        MAIN.index("def generate_quality_label")
    ]

    assert "ON CONFLICT (movie_id, season_number)" in link_section
    assert "ON CONFLICT DO NOTHING" in link_section
    assert "ON CONFLICT (file_unique_id) DO UPDATE" in upsert_section
    assert "WHERE movie_files.movie_id = EXCLUDED.movie_id" in upsert_section


def test_batch18_and_superbatch_share_identity_and_episode_parent_resolution():
    batch18 = MAIN[
        MAIN.index("async def batch18_listener"):
        MAIN.index("async def batch18_done")
    ]
    superbatch = MAIN[
        MAIN.index("async def superbatch_done"):
        MAIN.index("async def _core_movie_processor")
    ]

    assert "resolve_raw_content_identity(" in batch18
    assert "_resolve_episode_file_parent(" in batch18
    assert "_core_movie_processor(" in superbatch


def test_initial_pm_and_batch18_media_are_saved_after_parent_resolution():
    pm = MAIN[
        MAIN.index("async def pm_file_listener"):
        MAIN.index("async def batch18_listener")
    ]
    batch18 = MAIN[
        MAIN.index("async def batch18_listener"):
        MAIN.index("async def batch18_done")
    ]

    assert pm.index("BATCH_SESSION.update({") < pm.index(
        "first_file_label = await _pm_save_file(message, context)"
    )
    assert "first_file_label" in pm
    assert batch18.index("BATCH_18_SESSION.update({") < batch18.index(
        "upload_status = await message.reply_text("
    )
    assert "movie_file_id = upsert_movie_file(" in batch18


def test_movie_file_upsert_never_reparents_or_replaces_by_quality():
    section = MAIN[
        MAIN.index("def upsert_movie_file"):
        MAIN.index("def generate_quality_label", MAIN.index("def upsert_movie_file"))
    ]

    assert "ON CONFLICT (file_unique_id) DO UPDATE" in section
    assert "WHERE movie_files.movie_id = EXCLUDED.movie_id" in section
    assert "RETURNING id" in section
    assert "WHERE movie_id = %s AND quality = %s" not in section


def test_adult_batch_uses_the_shared_identity_creation_gate():
    section = MAIN[
        MAIN.index("async def batch18_listener"):
        MAIN.index("async def batch18_done", MAIN.index("async def batch18_listener"))
    ]

    assert "can_create_canonical_content(" in section
    assert "status=\"pending_review\"" in section
    assert "No movie row or file was created." in section


def test_duplicate_title_search_buttons_include_year_and_keep_all_rows():
    assert "ORDER BY year DESC NULLS LAST, id DESC" in MAIN
    assert "Multiple results found for" in MAIN
    assert 'button_text = f"{title}   {year_text}"' in MAIN


def test_mini_app_search_deduplicates_by_record_id_not_title():
    routes = (Path(__file__).resolve().parents[1] / "webapp_routes.py").read_text(
        encoding="utf-8"
    )
    assert "key = ('local', m['id'])" in routes
    assert "key = ('tmdb', m['id'])" in routes
