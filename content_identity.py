"""Deterministic parsing and safety checks for media content identity."""

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass(frozen=True)
class ParsedContentIdentity:
    canonical_title: str
    year: Optional[int]
    content_type: str
    season_number: Optional[int]
    episode_numbers: Tuple[int, ...]
    episode_range: Optional[Tuple[int, int]]
    quality: Optional[str]
    languages: Tuple[str, ...]
    extra_info: Tuple[str, ...]
    confidence: float
    parse_warnings: Tuple[str, ...]
    has_episode_marker: bool


_EPISODE_PATTERNS = (
    re.compile(
        r"(?<![\w])s(?:eason)?\s*0*(?P<season>\d{1,2})\s*"
        r"e(?:p(?:isode)?)?\s*0*(?P<first>\d{1,3})"
        r"(?:\s*[-~]\s*(?:e(?:p(?:isode)?)?\s*)?0*(?P<last>\d{1,3}))?(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\w])(?P<season>\d{1,2})x(?P<first>\d{1,3})"
        r"(?:\s*[-~]\s*(?:\d{1,2}x)?(?P<last>\d{1,3}))?(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![\w])(?:episode|ep|e)\s*[:.#-]*\s*0*(?P<first>\d{1,3})"
        r"(?:\s*[-~]\s*(?:(?:episode|ep|e)\s*)?0*(?P<last>\d{1,3}))?(?!\d)",
        re.IGNORECASE,
    ),
)
_SEASON_PATTERN = re.compile(
    r"(?<![\w])s(?:eason)?\s*0*(?P<season>\d{1,2})(?![\w])",
    re.IGNORECASE,
)
_QUALITY_PATTERNS = (
    ("2160p", re.compile(r"(?<!\w)(?:2160p|4k)(?!\w)", re.IGNORECASE)),
    ("1080p", re.compile(r"(?<!\w)1080p(?!\w)", re.IGNORECASE)),
    ("720p", re.compile(r"(?<!\w)720p(?!\w)", re.IGNORECASE)),
    ("480p", re.compile(r"(?<!\w)480p(?!\w)", re.IGNORECASE)),
    ("360p", re.compile(r"(?<!\w)360p(?!\w)", re.IGNORECASE)),
)
_LANGUAGES = (
    ("Dual Audio", re.compile(r"\bdual\s*audio\b", re.IGNORECASE)),
    ("Hindi", re.compile(r"\bhindi\b", re.IGNORECASE)),
    ("English", re.compile(r"\benglish\b|\beng\b", re.IGNORECASE)),
    ("Tamil", re.compile(r"\btamil\b", re.IGNORECASE)),
    ("Telugu", re.compile(r"\btelugu\b", re.IGNORECASE)),
    ("Malayalam", re.compile(r"\bmalayalam\b", re.IGNORECASE)),
    ("Japanese", re.compile(r"\bjapanese\b", re.IGNORECASE)),
    ("Korean", re.compile(r"\bkorean\b", re.IGNORECASE)),
)
_TECHNICAL_NOISE = re.compile(
    r"(?<!\w)(?:web[- .]?dl|webrip|bluray|blu[- .]?ray|hdrip|hdtv|"
    r"hevc|h\.?265|h\.?264|x26[45]|aac|ddp?\s*\d*(?:\.\d)?|dts|"
    r"atmos|hdr10?|dv|remux|proper|repack|limited|internal|"
    r"multi[- .]?audio|dual[- .]?audio|subs?|subbed|"
    r"amzn|nf|dsnp|hmax|hulu|hindi|english|tamil|telugu|malayalam|"
    r"japanese|korean|dual\s+audio|multi\s+audio)(?!\w)",
    re.IGNORECASE,
)
_RELEASE_GROUP = re.compile(
    r"(?:\s*[-_. ]+\s*)(?:rarbg|yify|yts|etrg|evo|ntb|kogi|ion10|"
    r"tgx|psa|qxr|fgt|galaxyrg|megusta|pahe|tigole|smurf|cmrg)\b.*$",
    re.IGNORECASE,
)
_EXTENSION = re.compile(r"\.(?:mkv|mp4|avi|mov|webm|m4v|ts)$", re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)")
_LATEST_UNAMBIGUOUS_RELEASE_YEAR = 2035
_EPISODE_ONLY = re.compile(
    r"^\s*(?:(?:episode|ep|e)\s*[:.#-]*\s*\d{1,3}|"
    r"s(?:eason)?\s*\d{1,2}(?:\s*e\s*\d{1,3})?)\s*$",
    re.IGNORECASE,
)
_PROMOTIONAL_PREFIX = re.compile(
    r"^(?:join\s+now|subscribe|download|watch\s+online|powered\s+by)\b"
    r"[\s:|,-]*",
    re.IGNORECASE,
)


def _title_from_prefix(raw: str, cutoff: int) -> str:
    title = raw[:cutoff]
    title = re.sub(r"https?://\S+|@\w+|#\w+", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"[\[\]{}()]", " ", title)
    title = re.sub(r"[._]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" \t\r\n-–—|:")
    title = _PROMOTIONAL_PREFIX.sub("", title).strip()
    return title


def parse_content_identity(value: str) -> ParsedContentIdentity:
    """Parse title and representation metadata without treating file noise as identity."""
    raw = str(value or "").strip()
    text = _EXTENSION.sub("", raw)
    warnings = []
    episodes = None
    episode_match = None

    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(text)
        if match:
            episode_match = match
            break

    season_number = None
    season_match = None
    episode_numbers = ()
    episode_range = None
    if episode_match:
        groups = episode_match.groupdict()
        season_number = int(groups["season"]) if groups.get("season") else None
        first = int(groups["first"])
        last = int(groups["last"]) if groups.get("last") else first
        if last < first:
            warnings.append("Episode range ends before it starts")
        elif last - first > 250:
            warnings.append("Episode range is too large to expand")
        else:
            episodes = tuple(range(first, last + 1))
            episode_numbers = episodes
            episode_range = (first, last) if last != first else None
    else:
        season_match = _SEASON_PATTERN.search(text)
        if season_match:
            season_number = int(season_match.group("season"))

    has_episode_marker = episode_match is not None
    marker_match = episode_match or season_match
    cutoff = marker_match.start() if marker_match else len(text)

    quality = None
    for label, pattern in _QUALITY_PATTERNS:
        match = pattern.search(text)
        if match:
            if quality is None:
                quality = label
            cutoff = min(cutoff, match.start())

    technical = _TECHNICAL_NOISE.search(text)
    if technical:
        cutoff = min(cutoff, technical.start())
    release_group = _RELEASE_GROUP.search(text)
    if release_group:
        cutoff = min(cutoff, release_group.start())

    year = None
    for match in _YEAR_TOKEN.finditer(text):
        following = text[match.end():]
        before = text[match.start() - 1:match.start()] if match.start() else ""
        after = text[match.end():match.end() + 1]
        bracketed = before in {"(", "["} and after in {")", "]"}
        release_context = bool(
            re.match(
                r"\s*(?:[.)\]]|1080p|720p|2160p|4k|web|bluray)",
                following,
                re.IGNORECASE,
            )
        )
        year_value = int(match.group("year"))
        plausible_release_year = year_value <= _LATEST_UNAMBIGUOUS_RELEASE_YEAR
        if match.start() and (bracketed or (plausible_release_year and release_context)):
            year = year_value
            cutoff = min(cutoff, match.start())
            break

    canonical_title = _title_from_prefix(text, cutoff)
    if _EPISODE_ONLY.match(canonical_title):
        canonical_title = ""
    if not canonical_title:
        warnings.append("No plausible title remains after parsing metadata")
    if episode_match and not canonical_title:
        warnings.append("Episode marker has no parent title")

    languages = tuple(
        label for label, pattern in _LANGUAGES if pattern.search(text)
    )
    extra_info = []
    if season_number is not None:
        extra_info.append("S{:02d}".format(season_number))
    if has_episode_marker and episode_numbers:
        if episode_range:
            extra_info.append(
                "E{:02d}-E{:02d}".format(episode_range[0], episode_range[1])
            )
        else:
            extra_info.append("E{:02d}".format(episode_numbers[0]))

    if has_episode_marker:
        content_type = "Web Series"
    elif season_number is not None:
        content_type = "Web Series"
    else:
        content_type = "Movie"

    if canonical_title and has_episode_marker:
        confidence = 0.9
    elif canonical_title and season_number is not None:
        confidence = 0.82
    elif canonical_title:
        confidence = 0.72
    else:
        confidence = 0.0

    return ParsedContentIdentity(
        canonical_title=canonical_title,
        year=year,
        content_type=content_type,
        season_number=season_number,
        episode_numbers=episode_numbers,
        episode_range=episode_range,
        quality=quality,
        languages=languages,
        extra_info=tuple(extra_info),
        confidence=confidence,
        parse_warnings=tuple(warnings),
        has_episode_marker=has_episode_marker,
    )


def resolve_raw_content_identity(raw_caption="", raw_filename="", raw_text=""):
    """Resolve identity only from the uploaded file's human-provided evidence."""
    candidates = []
    seen = set()
    for source, value in (
        ("caption", raw_caption),
        ("filename", raw_filename),
        ("raw_text", raw_text),
    ):
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        parsed = parse_content_identity(value)
        if is_safe_canonical_title(parsed.canonical_title):
            candidates.append((source, parsed))

    if not candidates:
        return None, None, "No plausible canonical title in raw caption or filename"

    episode_candidates = [
        candidate for candidate in candidates
        if candidate[1].has_episode_marker or candidate[1].season_number is not None
    ]
    canonical_titles = {
        re.sub(r"[^a-z0-9]+", "", parsed.canonical_title.casefold())
        for _, parsed in candidates
    }
    if len(canonical_titles) != 1:
        return None, None, "Raw caption and filename identify different content"

    if episode_candidates:
        source, parsed = episode_candidates[0]
    else:
        source, parsed = candidates[0]
    return source, parsed, None


def is_safe_canonical_title(value: str) -> bool:
    """Return false when a candidate still includes representation/episode noise."""
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(title) < 2 or _EPISODE_ONLY.match(title):
        return False
    if any(pattern.search(title) for _, pattern in _QUALITY_PATTERNS):
        return False
    if any(pattern.search(title) for pattern in _EPISODE_PATTERNS):
        return False
    if _TECHNICAL_NOISE.search(title):
        return False
    if _RELEASE_GROUP.search(title):
        return False
    return True


def can_create_canonical_content(
    parsed_identity: ParsedContentIdentity,
    provider_identity: bool,
    existing_parent: bool,
    ambiguous: bool = False,
    provider_conflict: bool = False,
) -> bool:
    """Gate automatic parent creation on provider or unique catalog identity."""
    if not is_safe_canonical_title(parsed_identity.canonical_title):
        return False
    if ambiguous or provider_conflict:
        return False
    return bool(provider_identity or existing_parent)
