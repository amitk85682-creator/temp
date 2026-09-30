from content_identity import (
    can_create_canonical_content,
    is_safe_canonical_title,
    parse_content_identity,
    resolve_raw_content_identity,
)


def test_parses_single_episode_and_file_attributes():
    parsed = parse_content_identity("Reacher S01E02 1080p WEB-DL x264 Hindi.mkv")

    assert parsed.canonical_title == "Reacher"
    assert parsed.content_type == "Web Series"
    assert parsed.season_number == 1
    assert parsed.episode_numbers == (2,)
    assert parsed.quality == "1080p"
    assert "Hindi" in parsed.languages


def test_parses_episode_ranges():
    for value in ("Reacher S01E02-08 1080p", "Reacher S01E02-E08"):
        parsed = parse_content_identity(value)
        assert parsed.canonical_title == "Reacher"
        assert parsed.season_number == 1
        assert parsed.episode_numbers == tuple(range(2, 9))
        assert parsed.episode_range == (2, 8)


def test_parses_alternate_episode_notation_and_season_only():
    episode = parse_content_identity("The Bear 1x02-08 WEBRip")
    season = parse_content_identity("The Bear S02 1080p")

    assert episode.canonical_title == "The Bear"
    assert episode.season_number == 1
    assert episode.episode_numbers == tuple(range(2, 9))
    assert season.canonical_title == "The Bear"
    assert season.season_number == 2
    assert season.content_type == "Web Series"


def test_parses_episode_labels_and_season_episode_variants():
    for value, season_number, episode_number in (
        ("Show S02E01", 2, 1),
        ("Show Episode 1", None, 1),
        ("Show Episode 01", None, 1),
        ("Show EP 01", None, 1),
        ("Show S01E02-E08", 1, 2),
    ):
        parsed = parse_content_identity(value)
        assert parsed.canonical_title == "Show"
        assert parsed.season_number == season_number
        assert parsed.episode_numbers[0] == episode_number


def test_episode_caption_without_parent_is_not_a_title():
    parsed = parse_content_identity("Episode :- 12")
    promoted = parse_content_identity("Join Now Reacher S01E12 1080p")

    assert parsed.canonical_title == ""
    assert parsed.episode_numbers == (12,)
    assert parsed.parse_warnings
    assert not is_safe_canonical_title("Episode :- 12")
    assert promoted.canonical_title == "Reacher"


def test_movie_quality_is_representation_metadata_and_numbered_title_survives():
    movie = parse_content_identity("Inception 1080p BluRay HEVC AAC")
    numbered_title = parse_content_identity("Blade Runner 2049")

    assert movie.canonical_title == "Inception"
    assert movie.quality == "1080p"
    assert numbered_title.canonical_title == "Blade Runner 2049"
    assert numbered_title.year is None
    assert is_safe_canonical_title(numbered_title.canonical_title)


def test_filename_with_movie_year_and_release_noise():
    parsed = parse_content_identity(
        "Dune (2021) 2160p WEB-DL Atmos Hindi-YIFY.mkv"
    )
    bare_year = parse_content_identity("Dune 2021 1080p")
    web_title = parse_content_identity("The Web")
    blade_runner = parse_content_identity("Blade Runner 2049 4K")

    assert parsed.canonical_title == "Dune"
    assert parsed.year == 2021
    assert parsed.quality == "2160p"
    assert "Hindi" in parsed.languages
    assert bare_year.canonical_title == "Dune"
    assert bare_year.year == 2021
    assert web_title.canonical_title == "The Web"
    assert blade_runner.canonical_title == "Blade Runner 2049"
    assert blade_runner.year is None


def test_language_and_known_release_group_are_removed_from_identity():
    parsed = parse_content_identity("Inception Hindi 1080p WEB-DL x264-RARBG")

    assert parsed.canonical_title == "Inception"
    assert parsed.languages == ("Hindi",)


def test_series_creation_requires_provider_or_unique_existing_parent():
    episode = parse_content_identity("Unknown Show S01E02")
    movie = parse_content_identity("Inception 1080p")

    assert not can_create_canonical_content(episode, False, False)
    assert can_create_canonical_content(episode, True, False)
    assert can_create_canonical_content(episode, False, True)
    assert not can_create_canonical_content(episode, False, True, ambiguous=True)
    assert not can_create_canonical_content(
        episode, False, True, provider_conflict=True
    )
    assert not can_create_canonical_content(
        episode, True, False, ambiguous=True
    )
    assert not can_create_canonical_content(movie, False, False)
    assert can_create_canonical_content(movie, True, False)


def test_reacher_episode_variants_resolve_only_to_the_raw_canonical_parent():
    for value in (
        "Reacher S01E02",
        "Reacher S01E02-08",
        "Reacher S01E02-E08",
        "Reacher 1x02",
        "Reacher Episode 2",
        "Reacher S02E01 1080p",
    ):
        source, parsed, error = resolve_raw_content_identity("", value)

        assert source == "filename"
        assert error is None
        assert parsed.canonical_title == "Reacher"
        assert parsed.has_episode_marker
        assert is_safe_canonical_title(parsed.canonical_title)
        assert not is_safe_canonical_title(value)


def test_ai_title_cannot_supply_raw_identity_and_conflicting_files_are_ambiguous():
    source, parsed, error = resolve_raw_content_identity("", "", "")
    assert (source, parsed) == (None, None)
    assert error

    source, parsed, error = resolve_raw_content_identity(
        "Reacher S01E02",
        "The Bear S01E02.mkv",
    )
    assert (source, parsed) == (None, None)
    assert "different content" in error
