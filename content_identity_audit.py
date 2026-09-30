"""Dry-run-first audit and reversible repair for polluted movie identities."""

import argparse
import csv
import json
import os
import re
import sys

import psycopg2
from psycopg2 import sql

from content_identity import is_safe_canonical_title, parse_content_identity


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def classify_identity_rows(movies, files_by_movie):
    """Create explainable candidates without changing any database rows."""
    candidates = []
    for wrong in movies:
        parsed = parse_content_identity(wrong["title"])
        if is_safe_canonical_title(wrong["title"]):
            continue
        canonical_key = _normalized(parsed.canonical_title)
        if not canonical_key:
            continue

        possible = []
        for parent in movies:
            if parent["id"] == wrong["id"]:
                continue
            if _normalized(parent["title"]) != canonical_key:
                continue
            if (
                parsed.has_episode_marker
                and str(parent.get("content_type") or "").casefold()
                not in {"web series", "tv series", "tv show", "series", "anime"}
            ):
                continue
            wrong_year = int(wrong.get("year") or 0)
            parent_year = int(parent.get("year") or 0)
            if wrong_year and parent_year and wrong_year != parent_year:
                continue
            provider_conflict = any(
                wrong.get(key) and parent.get(key) and wrong[key] != parent[key]
                for key in ("imdb_id", "tmdb_id")
            )
            if provider_conflict:
                continue
            possible.append(parent)

        reasons = []
        if parsed.has_episode_marker:
            reasons.append("season/episode marker embedded in canonical title")
        elif parsed.season_number is not None:
            reasons.append("season marker embedded in canonical title")
        else:
            reasons.append("representation/technical marker embedded in title")
        if possible:
            reasons.append("exact normalized canonical parent title exists")
        if wrong.get("content_type") and possible and str(
            possible[0].get("content_type") or ""
        ).casefold() in {"web series", "tv series", "tv show", "series", "anime"}:
            reasons.append("parent is typed as a series")
        if parsed.has_episode_marker and str(
            wrong.get("content_type") or ""
        ).casefold() not in {"web series", "tv series", "tv show", "series", "anime"}:
            reasons.append("candidate content type conflicts with episode markers")
        files_affected = len(files_by_movie.get(wrong["id"], ()))

        parent = possible[0] if len(possible) == 1 else None
        confidence = 0.96 if parent and parsed.has_episode_marker else (
            0.85 if parent else 0.45
        )
        candidates.append({
            "wrong_row_id": wrong["id"],
            "wrong_title": wrong["title"],
            "possible_canonical_row_id": parent["id"] if parent else "",
            "possible_canonical_title": parent["title"] if parent else "",
            "reason": "; ".join(reasons),
            "confidence": confidence,
            "files_affected": files_affected,
            "proposed_action": (
                "archive row and move relationships; retire only if constraints permit"
                if parent and confidence >= 0.9
                else "manual review; no automatic repair proposed"
            ),
            "_wrong_movie": wrong,
            "_parent_movie": parent,
        })
    return candidates


def _load_identity_rows(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, year, content_type, imdb_id, tmdb_id
        FROM movies
        ORDER BY id
    """)
    columns = ("id", "title", "year", "content_type", "imdb_id", "tmdb_id")
    movies = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.execute("SELECT id, movie_id, quality, languages, extra_info, file_unique_id FROM movie_files")
    files_by_movie = {}
    for row in cur.fetchall():
        files_by_movie.setdefault(row[1], []).append({
            "id": row[0],
            "quality": row[2],
            "languages": row[3],
            "extra_info": row[4],
            "file_unique_id": row[5],
        })
    cur.close()
    return movies, files_by_movie


def _referencing_columns(cur):
    cur.execute("""
        SELECT ns.nspname, rel.relname, att.attname
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace ns ON ns.oid = rel.relnamespace
        JOIN LATERAL unnest(con.conkey) AS key(attnum) ON TRUE
        JOIN pg_attribute att ON att.attrelid = rel.oid AND att.attnum = key.attnum
        WHERE con.contype = 'f'
          AND con.confrelid = 'public.movies'::regclass
    """)
    columns = set(cur.fetchall())
    cur.execute("""
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE column_name = 'movie_id'
    """)
    columns.update(cur.fetchall())
    return sorted(item for item in columns if item[1] != "movies")


def apply_repair(conn, candidate):
    """Archive and remap one high-confidence identity without losing episode links."""
    if candidate["confidence"] < 0.9 or not candidate.get("_parent_movie"):
        return False, "candidate is not high confidence"
    wrong_id = candidate["wrong_row_id"]
    parent_id = candidate["possible_canonical_row_id"]
    cur = conn.cursor()
    cur.execute(
        "SELECT restored_at FROM content_identity_repair_archive WHERE wrong_movie_id = %s",
        (wrong_id,),
    )
    prior_archive = cur.fetchone()
    if prior_archive and prior_archive[0] is None:
        cur.close()
        return True, "repair was already archived"
    if prior_archive:
        cur.close()
        return False, "a previous repair was restored; review before applying again"

    cur.execute(
        "SELECT to_jsonb(m) FROM movies AS m WHERE id = %s FOR UPDATE",
        (wrong_id,),
    )
    original_movie = cur.fetchone()
    if not original_movie:
        cur.close()
        return False, "candidate row no longer exists"
    wrong_movie = candidate["_wrong_movie"]
    if (
        original_movie[0].get("title") != candidate["wrong_title"]
        or int(original_movie[0].get("year") or 0) != int(wrong_movie.get("year") or 0)
        or original_movie[0].get("imdb_id") != wrong_movie.get("imdb_id")
        or original_movie[0].get("tmdb_id") != wrong_movie.get("tmdb_id")
    ):
        cur.close()
        return False, "polluted row changed since the dry run"

    parsed_wrong = parse_content_identity(candidate["wrong_title"])
    series_types = {"web series", "tv series", "tv show", "series", "anime"}
    cur.execute(
        """
        SELECT id, title, year, content_type, imdb_id, tmdb_id
        FROM movies
        WHERE id <> %s
          AND regexp_replace(lower(title), '[^a-z0-9]', '', 'g') = %s
        """,
        (wrong_id, _normalized(parsed_wrong.canonical_title)),
    )
    possible_parents = []
    for row in cur.fetchall():
        if (
            wrong_movie.get("year") and row[2]
            and int(wrong_movie["year"]) != int(row[2])
        ):
            continue
        if (
            (parsed_wrong.has_episode_marker or parsed_wrong.season_number is not None)
            and str(row[3] or "").casefold() not in series_types
        ):
            continue
        if any(
            wrong_movie.get(key) and row[index] and wrong_movie[key] != row[index]
            for key, index in (("imdb_id", 4), ("tmdb_id", 5))
        ):
            continue
        possible_parents.append(row)
    if len(possible_parents) != 1 or possible_parents[0][0] != parent_id:
        cur.close()
        return False, "canonical parent is now missing, conflicting, or ambiguous"

    cur.execute(
        "SELECT title, year, content_type, imdb_id, tmdb_id "
        "FROM movies WHERE id = %s FOR UPDATE",
        (parent_id,),
    )
    parent = cur.fetchone()
    if not parent:
        cur.close()
        return False, "canonical parent no longer exists"
    if _normalized(parent[0]) != _normalized(candidate["possible_canonical_title"]):
        cur.close()
        return False, "canonical parent changed since the dry run"
    if (
        parent[1] and candidate["_parent_movie"].get("year")
        and int(parent[1]) != int(candidate["_parent_movie"]["year"])
    ):
        cur.close()
        return False, "canonical parent year changed since the dry run"
    if parsed_wrong.has_episode_marker or parsed_wrong.season_number is not None:
        if str(parent[2] or "").casefold() not in series_types:
            cur.close()
            return False, "canonical parent is no longer typed as a series"
    if any(
        wrong_movie.get(key) and parent[index] and wrong_movie[key] != parent[index]
        for key, index in (("imdb_id", 3), ("tmdb_id", 4))
    ):
        cur.close()
        return False, "provider identities conflict with the canonical parent"

    cur.execute("SAVEPOINT identity_repair")
    related_rows = {}
    try:
        structured = {}
        cur.execute("SELECT to_jsonb(s) FROM seasons AS s WHERE movie_id = %s", (wrong_id,))
        structured["seasons"] = [row[0] for row in cur.fetchall()]
        old_season_ids = [row["id"] for row in structured["seasons"]]
        if old_season_ids:
            cur.execute(
                "SELECT to_jsonb(e) FROM episodes AS e WHERE season_id = ANY(%s)",
                (old_season_ids,),
            )
            structured["episodes"] = [row[0] for row in cur.fetchall()]
            old_episode_ids = [row["id"] for row in structured["episodes"]]
            cur.execute(
                "SELECT to_jsonb(fs) FROM file_seasons AS fs WHERE season_id = ANY(%s)",
                (old_season_ids,),
            )
            structured["file_seasons"] = [row[0] for row in cur.fetchall()]
            if old_episode_ids:
                cur.execute(
                    "SELECT to_jsonb(fe) FROM file_episodes AS fe WHERE episode_id = ANY(%s)",
                    (old_episode_ids,),
                )
                structured["file_episodes"] = [row[0] for row in cur.fetchall()]
        related_rows["__structured__"] = structured

        for schema_name, table_name, column_name in _referencing_columns(cur):
            if table_name == "seasons":
                continue
            relation = sql.Identifier(schema_name, table_name)
            column = sql.Identifier(column_name)
            cur.execute(
                sql.SQL("SELECT to_jsonb(t) FROM {} AS t WHERE {} = %s").format(
                    relation, column
                ),
                (wrong_id,),
            )
            rows = [row[0] for row in cur.fetchall()]
            if rows:
                related_rows["{}.{}.{}".format(schema_name, table_name, column_name)] = rows
                cur.execute(
                    sql.SQL("UPDATE {} SET {} = %s WHERE {} = %s").format(
                        relation, column, column
                    ),
                    (parent_id, wrong_id),
                )

        season_map = {}
        for season in structured.get("seasons", []):
            cur.execute(
                """
                INSERT INTO seasons (movie_id, season_number)
                VALUES (%s, %s)
                ON CONFLICT (movie_id, season_number) DO NOTHING
                """,
                (parent_id, season["season_number"]),
            )
            cur.execute(
                "SELECT id FROM seasons WHERE movie_id = %s AND season_number = %s",
                (parent_id, season["season_number"]),
            )
            season_map[season["id"]] = cur.fetchone()[0]

        episode_map = {}
        for episode in structured.get("episodes", []):
            target_season_id = season_map[episode["season_id"]]
            cur.execute(
                """
                INSERT INTO episodes (season_id, episode_number)
                VALUES (%s, %s)
                ON CONFLICT (season_id, episode_number) DO NOTHING
                """,
                (target_season_id, episode["episode_number"]),
            )
            cur.execute(
                "SELECT id FROM episodes WHERE season_id = %s AND episode_number = %s",
                (target_season_id, episode["episode_number"]),
            )
            episode_map[episode["id"]] = cur.fetchone()[0]

        for link in structured.get("file_seasons", []):
            target_season_id = season_map[link["season_id"]]
            cur.execute(
                "DELETE FROM file_seasons WHERE movie_file_id = %s AND season_id = %s",
                (link["movie_file_id"], link["season_id"]),
            )
            cur.execute(
                """
                INSERT INTO file_seasons (movie_file_id, season_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
                """,
                (link["movie_file_id"], target_season_id),
            )

        for link in structured.get("file_episodes", []):
            target_episode_id = episode_map[link["episode_id"]]
            cur.execute(
                "DELETE FROM file_episodes WHERE movie_file_id = %s AND episode_id = %s",
                (link["movie_file_id"], link["episode_id"]),
            )
            cur.execute(
                """
                INSERT INTO file_episodes (movie_file_id, episode_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
                """,
                (link["movie_file_id"], target_episode_id),
            )

        cur.execute(
            """
            INSERT INTO content_identity_repair_archive
                (wrong_movie_id, canonical_movie_id, original_movie, related_rows, audit_report)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            """,
            (
                wrong_id,
                parent_id,
                json.dumps(original_movie[0], default=str),
                json.dumps(related_rows, default=str),
                json.dumps({
                    key: value for key, value in candidate.items()
                    if not key.startswith("_")
                }, default=str),
            ),
        )
        cur.execute("DELETE FROM seasons WHERE movie_id = %s", (wrong_id,))
        for schema_name, table_name, column_name in _referencing_columns(cur):
            if table_name == "seasons":
                continue
            cur.execute(
                sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE {} = %s)").format(
                    sql.Identifier(schema_name, table_name),
                    sql.Identifier(column_name),
                ),
                (wrong_id,),
            )
            if cur.fetchone()[0]:
                raise RuntimeError(
                    "relationship remains on {}.{}.{}".format(
                        schema_name, table_name, column_name
                    )
                )
        cur.execute(
            "SELECT COUNT(*) FROM movies AS m "
            "WHERE m.id = %s AND EXISTS ("
            "SELECT 1 FROM movie_files mf WHERE mf.movie_id = m.id)",
            (wrong_id,),
        )
        if cur.fetchone()[0]:
            raise RuntimeError("movie files remain attached to the polluted row")
        cur.execute("DELETE FROM movies WHERE id = %s", (wrong_id,))
        cur.execute("RELEASE SAVEPOINT identity_repair")
        cur.close()
        return True, "archived and repaired"
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            cur.close()
            raise
        cur.close()
        return False, str(exc)


def restore_repair(conn, wrong_movie_id):
    """Restore an archived movie row and its remapped relationships."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT original_movie, related_rows
        FROM content_identity_repair_archive
        WHERE wrong_movie_id = %s AND restored_at IS NULL
        FOR UPDATE
        """,
        (wrong_movie_id,),
    )
    archived = cur.fetchone()
    if not archived:
        cur.close()
        return False, "no unrestored archive entry exists"
    original_movie, related_rows = archived

    cur.execute("SELECT 1 FROM movies WHERE id = %s", (wrong_movie_id,))
    if cur.fetchone():
        cur.close()
        return False, "movie ID is already in use"

    cur.execute(
        "INSERT INTO movies SELECT * FROM jsonb_populate_record(NULL::movies, %s::jsonb)",
        (json.dumps(original_movie, default=str),),
    )
    structured = related_rows.pop("__structured__", {})
    for relation_key, rows in related_rows.items():
        schema_name, table_name, column_name = relation_key.split(".", 2)
        cur.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
             AND tc.table_name = kcu.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            """,
            (schema_name, table_name),
        )
        primary_key = [row[0] for row in cur.fetchall()]
        if not primary_key:
            raise ValueError(
                "Cannot restore relationship table without a primary key: "
                "{}.{}".format(schema_name, table_name)
            )
        relation = sql.Identifier(schema_name, table_name)
        column = sql.Identifier(column_name)
        for row in rows:
            conditions = sql.SQL(" AND ").join(
                sql.SQL("{} IS NOT DISTINCT FROM %s").format(sql.Identifier(key))
                for key in primary_key
            )
            values = tuple(row.get(key) for key in primary_key)
            cur.execute(
                sql.SQL("UPDATE {} SET {} = %s WHERE {}").format(
                    relation, column, conditions
                ),
                (wrong_movie_id,) + values,
            )
            if cur.rowcount == 0:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} SELECT * FROM jsonb_populate_record(NULL::{}.{} , %s::jsonb)"
                    ).format(
                        relation,
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                    ),
                    (json.dumps(row, default=str),),
                )

    for table_name in ("seasons", "episodes", "file_seasons", "file_episodes"):
        table_rows = structured.get(table_name, [])
        for row in table_rows:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} SELECT * FROM jsonb_populate_record(NULL::{}.{} , %s::jsonb)"
                ).format(
                    sql.Identifier("public", table_name),
                    sql.Identifier("public"),
                    sql.Identifier(table_name),
                ),
                (json.dumps(row, default=str),),
            )

    cur.execute(
        """
        UPDATE content_identity_repair_archive
        SET restored_at = CURRENT_TIMESTAMP
        WHERE wrong_movie_id = %s
        """,
        (wrong_movie_id,),
    )
    cur.close()
    return True, "archived identity restored"


def _write_report(candidates, output_path):
    fields = (
        "wrong_row_id",
        "wrong_title",
        "possible_canonical_row_id",
        "possible_canonical_title",
        "reason",
        "confidence",
        "files_affected",
        "proposed_action",
    )
    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {key: candidate.get(key, "") for key in fields}
                for candidate in candidates
            )
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: candidate.get(key, "") for key in fields}
            for candidate in candidates
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        help="CSV report path (defaults to standard output)",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--apply",
        action="store_true",
        help="apply only high-confidence repairs after archiving original rows",
    )
    actions.add_argument(
        "--restore",
        type=int,
        metavar="WRONG_MOVIE_ID",
        help="restore a movie and its relationships from the repair archive",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required; its value will not be printed")
    conn = psycopg2.connect(database_url)
    try:
        if args.restore is not None:
            restored, detail = restore_repair(conn, args.restore)
            if restored:
                conn.commit()
            else:
                conn.rollback()
            print(detail)
            return
        movies, files_by_movie = _load_identity_rows(conn)
        candidates = classify_identity_rows(movies, files_by_movie)
        _write_report(candidates, args.output)
        if args.apply:
            conn.close()
            conn = None
            for candidate in candidates:
                if candidate["confidence"] < 0.9:
                    continue
                try:
                    if conn is None or conn.closed:
                        conn = psycopg2.connect(database_url)
                    applied, detail = apply_repair(conn, candidate)
                    if applied:
                        conn.commit()
                    else:
                        conn.rollback()
                except Exception as exc:
                    if conn and not conn.closed:
                        try:
                            conn.rollback()
                        except Exception as rollback_exc:
                            print(
                                "Rollback failed for row {}: {}".format(
                                    candidate["wrong_row_id"], rollback_exc
                                ),
                                file=sys.stderr,
                            )
                            conn.close()
                            conn = None
                    elif conn:
                        conn.close()
                        conn = None
                    print(
                        "Repair skipped for row {}: {}".format(
                            candidate["wrong_row_id"], exc
                        ),
                        file=sys.stderr,
                    )
                    continue
                if not applied:
                    print(
                        "Repair skipped for row {}: {}".format(
                            candidate["wrong_row_id"], detail
                        ),
                        file=sys.stderr,
                    )
    finally:
        if conn and not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()
