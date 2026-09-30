import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import pytest
import psycopg2.pool


class _ImportOnlyPool:
    def __init__(self, *_args, **_kwargs):
        pass

    def getconn(self):
        raise AssertionError("unexpected live database access")

    def putconn(self, _connection):
        pass

    def closeall(self):
        pass


@pytest.fixture(scope="module")
def bot_module():
    required_environment = {
        "TELEGRAM_BOT_TOKEN": "synthetic-test-token",
        "DATABASE_URL": "postgresql://synthetic.invalid/test",
        "TMDB_API_KEY": "synthetic-tmdb-key",
        "UPDATE_SECRET_CODE": "synthetic-update-secret",
    }
    original_environment = {
        key: os.environ.get(key) for key in required_environment
    }
    os.environ.update(required_environment)
    original_pool = psycopg2.pool.ThreadedConnectionPool
    psycopg2.pool.ThreadedConnectionPool = _ImportOnlyPool
    try:
        yield importlib.import_module("main")
    finally:
        psycopg2.pool.ThreadedConnectionPool = original_pool
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages) + 100)


class _FakeMessage:
    def __init__(self, text="", chat_id=7, animation=False):
        self.text = text
        self.message_id = 42
        self.chat = SimpleNamespace(id=chat_id, type="private" if chat_id > 0 else "group")
        self.animation = object() if animation else None
        self.edited_text = []
        self.edited_caption = []
        self.replies = []
        self.animations = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        return self

    async def reply_animation(self, animation, caption, **kwargs):
        self.animations.append((animation, caption, kwargs))
        return SimpleNamespace(message_id=98)

    async def edit_text(self, text, **kwargs):
        self.edited_text.append((text, kwargs))
        return self


class _FakeCallback:
    def __init__(self, data, message=None, user_id=10):
        self.data = data
        self.message = message or _FakeMessage(animation=True)
        self.from_user = SimpleNamespace(
            id=user_id, first_name="Tester", username="tester"
        )
        self.answered = []
        self.edited_text = []
        self.edited_caption = []

    async def answer(self, *args, **kwargs):
        self.answered.append((args, kwargs))

    async def edit_message_text(self, text, **kwargs):
        self.edited_text.append((text, kwargs))
        return self.message

    async def edit_message_caption(self, caption, **kwargs):
        self.edited_caption.append((caption, kwargs))
        return self.message


class _FakeContext:
    def __init__(self):
        self.bot = _FakeBot()
        self.user_data = {}


def _callback_update(callback, chat_id=7):
    return SimpleNamespace(
        callback_query=callback,
        effective_chat=callback.message.chat,
        effective_user=callback.from_user,
        message=None,
    )


def _group_update(text, user_id, chat_id=-1001):
    message = _FakeMessage(text, chat_id=chat_id)
    user = SimpleNamespace(id=user_id, first_name="Group user")
    return SimpleNamespace(
        message=message,
        effective_chat=message.chat,
        effective_user=user,
        callback_query=None,
    )


def _run(coro):
    return asyncio.run(coro)


def test_animation_retry_no_results_replaces_caption_and_answers_callback(
    bot_module, monkeypatch
):
    bot = bot_module
    callback = _FakeCallback("retrysearch_Some%20%26%20%C3%89Title")
    context = _FakeContext()
    searched = []

    async def fake_run_async(function, title, **_kwargs):
        searched.append((function, title))
        return []

    async def no_suggestions(*_args, **_kwargs):
        return []

    monkeypatch.setattr(bot, "run_async", fake_run_async)
    monkeypatch.setattr(bot, "get_google_title_suggestions_with_timeout", no_suggestions)

    _run(bot.button_callback(_callback_update(callback), context))

    assert callback.answered
    assert searched[0][1] == "Some & ÉTitle"
    assert callback.edited_caption
    assert not callback.edited_text
    assert "Some &amp; ÉTitle" in callback.edited_caption[0][0]
    assert callback.edited_caption[0][1]["reply_markup"].inline_keyboard


def test_private_not_found_animation_shows_working_retry_button(bot_module, monkeypatch):
    bot = bot_module
    message = _FakeMessage("Unknown query")
    update = SimpleNamespace(
        message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=10),
        callback_query=None,
    )
    context = _FakeContext()
    context.user_data["_search_progress_message"] = SimpleNamespace(
        delete=lambda: _async_return(None)()
    )

    async def no_results(*_args, **_kwargs):
        return []

    async def suggestions(*_args, **_kwargs):
        return ["Suggested Title"]

    monkeypatch.setattr(bot, "run_async", no_results)
    monkeypatch.setattr(bot, "get_google_title_suggestions_with_timeout", suggestions)
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "track_user_message_for_deletion", lambda *_args: None)
    monkeypatch.setattr(bot, "SEARCH_ERROR_GIFS", ["synthetic-animation"])

    _run(bot.search_movies(update, context))

    assert len(message.animations) == 1
    _, caption, options = message.animations[0]
    assert "No matching title" in caption
    buttons = [
        button
        for row in options["reply_markup"].inline_keyboard
        for button in row
    ]
    retry = next(button.callback_data for button in buttons
                 if button.callback_data and button.callback_data.startswith("retrysearch_"))
    assert unquote(retry.removeprefix("retrysearch_")) == "Suggested Title"


def test_retry_search_found_opens_movie_files(bot_module, monkeypatch):
    bot = bot_module
    callback = _FakeCallback("retrysearch_Found%20Title")
    context = _FakeContext()
    sent = []

    async def fake_run_async(_function, _title, **_kwargs):
        return [(5, "Found Title", "url", "file")]

    async def fake_send_movie(*args):
        sent.append(args[2:6])

    monkeypatch.setattr(bot, "run_async", fake_run_async)
    monkeypatch.setattr(
        bot, "_select_single_search_result",
        lambda _query, movies: movies[0],
    )
    monkeypatch.setattr(bot, "send_movie_to_user", fake_send_movie)

    _run(bot.button_callback(_callback_update(callback), context))

    assert callback.answered
    assert callback.edited_caption
    assert sent and sent[0][0:2] == (5, "Found Title")


def test_private_success_uses_shared_core_without_google_or_fuzzy(
    bot_module, monkeypatch
):
    bot = bot_module
    message = _FakeMessage("Found Title")
    update = SimpleNamespace(
        message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=10),
        callback_query=None,
    )
    context = _FakeContext()
    progress = _FakeMessage("Searching")
    exact = []
    displayed = []

    async def fast_search(_function, query, **_kwargs):
        exact.append(query)
        return [(5, "Found Title", "url", "file")]

    async def unexpected_fallback(*_args, **_kwargs):
        pytest.fail("successful search must not invoke suggestions")

    async def display(*args, **kwargs):
        displayed.append((args, kwargs))

    monkeypatch.setattr(bot, "run_async", fast_search)
    monkeypatch.setattr(bot, "send_search_progress", _async_return(progress))
    monkeypatch.setattr(bot, "get_google_title_suggestions_with_timeout", unexpected_fallback)
    monkeypatch.setattr(bot, "get_movies_from_db", lambda *_args, **_kwargs: pytest.fail("legacy fuzzy route used"))
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "process_movie_exact_match", display)

    _run(bot.search_movies(update, context))

    assert exact == ["Found Title"]
    assert displayed and displayed[0][1]["status_message"] is progress


def test_retry_search_exception_answers_callback_and_shows_retry_state(
    bot_module, monkeypatch
):
    bot = bot_module
    callback = _FakeCallback("retrysearch_Broken")
    context = _FakeContext()

    async def fail_search(*_args, **_kwargs):
        raise RuntimeError("synthetic search failure")

    monkeypatch.setattr(bot, "run_async", fail_search)

    _run(bot.button_callback(_callback_update(callback), context))

    assert callback.answered
    assert callback.edited_caption
    assert "temporarily unavailable" in callback.edited_caption[0][0]


def test_request_prefill_callback_preserves_title_and_can_be_confirmed(
    bot_module, monkeypatch
):
    bot = bot_module
    title = "The Long Request Title"
    callback = _FakeCallback("request_prefill_" + bot.quote(title, safe=""))
    update = _callback_update(callback)
    context = _FakeContext()
    assert len(callback.data.encode("utf-8")) <= 64

    state = _run(bot.start_request_flow(update, context))

    assert callback.answered
    assert state == bot.CONFIRMATION
    assert context.user_data["temp_request_name"] == title
    assert callback.edited_caption
    assert "The Long Request Title" in callback.edited_caption[0][0]

    confirm = _FakeCallback("confirm_yes", message=callback.message)
    confirm.from_user = callback.from_user
    confirm_update = _callback_update(confirm)
    stored = []
    async def call_function(function, *args, **_kwargs):
        return function(*args)

    monkeypatch.setattr(bot, "run_async", call_function)
    monkeypatch.setattr(bot, "send_admin_notification", _async_return(None))
    monkeypatch.setattr(bot, "track_message_for_deletion", lambda *_args: None)
    monkeypatch.setattr(
        bot,
        "store_user_request",
        lambda *args: stored.append(args) or True,
    )
    state = _run(bot.handle_confirmation_callback(confirm_update, context))

    assert confirm.answered
    assert state == bot.ConversationHandler.END
    assert stored[0][3] == title
    assert confirm.edited_caption


def _async_return(value):
    async def result(*_args, **_kwargs):
        return value
    return result


def test_request_portal_prefills_req_and_invalid_configuration_falls_back(
    bot_module, monkeypatch
):
    bot = bot_module
    title = "A & B (2026)"
    monkeypatch.setattr(bot, "WEB_APP_URL", "not a valid URL")
    portal = bot._request_portal_url(title)
    parsed = urlparse(portal)

    assert parsed.scheme == "https"
    assert parsed.path == "/webapp"
    assert parse_qs(parsed.query)["req"] == [title]
    keyboard = bot._not_found_keyboard(title, ())
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    assert any(button.url == portal for button in buttons)
    response = bot.flask_app.test_client().get(portal.replace(
        "https://flimfybox-bot-yht0.onrender.com", ""
    ))
    assert response.status_code == 200
    frontend = (
        Path(bot.__file__).resolve().parent / "static" / "miniapp" / "app.js"
    ).read_text(encoding="utf-8")
    assert "searchInput.value = requestedTitle" in frontend
    assert "window.requestMovie(requestedTitle)" in frontend


def test_invalid_update_channel_is_not_rendered(bot_module, monkeypatch):
    bot = bot_module
    monkeypatch.setattr(bot, "UPDATE_CHANNEL_URL", "invalid-channel-url")
    keyboard = bot._not_found_keyboard("Missing Title", ())
    buttons = [button for row in keyboard.inline_keyboard for button in row]

    assert any(button.text == "🌐 Open Request Portal" for button in buttons)
    assert not any("Update Channel" in button.text for button in buttons)


def test_retry_callback_data_round_trips_without_title_truncation(bot_module):
    bot = bot_module
    title = "A Title & More"
    keyboard = bot._not_found_keyboard(title, [title])
    buttons = [button for row in keyboard.inline_keyboard for button in row]
    callbacks = [button.callback_data for button in buttons if button.callback_data]

    assert all(len(data.encode("utf-8")) <= 64 for data in callbacks)
    assert unquote(next(data.removeprefix("retrysearch_") for data in callbacks
                        if data.startswith("retrysearch_"))) == title
    assert unquote(next(data.removeprefix("request_prefill_") for data in callbacks
                        if data.startswith("request_prefill_"))) == title


def test_unsupported_message_edit_falls_back_to_new_search_message(bot_module):
    bot = bot_module
    callback = _FakeCallback("retrysearch_title", message=_FakeMessage(animation=False))
    context = _FakeContext()

    async def cannot_edit(*_args, **_kwargs):
        raise RuntimeError("synthetic unsupported edit")

    callback.edit_message_text = cannot_edit
    sent = _run(bot._replace_search_result_message(
        callback, context, "Updated state"
    ))

    assert sent.message_id == 101
    assert context.bot.messages[0]["text"] == "Updated state"


def test_group_search_found_responds_with_movie(bot_module, monkeypatch):
    bot = bot_module
    update = _group_update("Found Title", 15)
    context = _FakeContext()
    movie = (5, "Found Title", "url", "file")
    displayed = []

    async def fake_run_async(*_args, **_kwargs):
        return [movie]

    async def display(*args, **kwargs):
        displayed.append((args[2:], kwargs))

    monkeypatch.setattr(bot, "run_async", fake_run_async)
    monkeypatch.setattr(bot, "_select_single_search_result", lambda *_args: movie)
    monkeypatch.setattr(bot, "process_movie_exact_match", display)
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)

    _run(bot.handle_group_message(update, context))

    assert displayed and displayed[0][0][:2] == (5, "Found Title")
    assert displayed[0][1]["status_message"] is update.message
    assert update.message.replies[0][0].startswith('🔎 Search for "Found Title"')


def test_group_search_no_result_always_replies_compactly(bot_module, monkeypatch):
    bot = bot_module
    update = _group_update("Missing <Title>", 16)
    context = _FakeContext()

    async def no_results(*_args, **_kwargs):
        return []

    monkeypatch.setattr(bot, "run_async", no_results)
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)

    result = _run(bot.handle_group_message(update, context))

    assert result is None
    assert len(update.message.replies) == 1
    assert update.message.replies[0][0].startswith('🔎 Search for "Missing')
    text, options = update.message.edited_text[0]
    assert "No matching movie found" in text
    assert "&lt;Title&gt;" in text
    assert len(text) < 300
    assert options["reply_markup"].inline_keyboard


def test_group_search_error_sends_visible_temporary_failure(bot_module, monkeypatch):
    bot = bot_module
    update = _group_update("Broken Search", 17)
    context = _FakeContext()

    async def fail_search(*_args, **_kwargs):
        raise RuntimeError("synthetic database error")

    monkeypatch.setattr(bot, "run_async", fail_search)

    _run(bot.handle_group_message(update, context))

    assert update.message.replies[0][0].startswith('🔎 Search for "Broken Search"')
    assert "temporarily unavailable" in update.message.edited_text[0][0]


def test_concurrent_group_searches_keep_each_users_results_isolated(
    bot_module, monkeypatch
):
    bot = bot_module
    first_update = _group_update("First Search", 101)
    second_update = _group_update("Second Search", 202)
    first_context = _FakeContext()
    second_context = _FakeContext()
    rows = {
        "First Search": [(1, "First Result", "", "") , (2, "First Alternate", "", "")],
        "Second Search": [(3, "Second Result", "", ""), (4, "Second Alternate", "", "")],
    }

    async def fake_run_async(_function, query, **_kwargs):
        await asyncio.sleep(0)
        return rows[query]

    monkeypatch.setattr(bot, "run_async", fake_run_async)
    monkeypatch.setattr(bot, "_select_single_search_result", lambda *_args: None)
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)
    monkeypatch.setattr(bot, "track_message_for_deletion", lambda *_args: None)

    async def run_both():
        await asyncio.gather(
            bot.handle_group_message(first_update, first_context),
            bot.handle_group_message(second_update, second_context),
        )

    _run(run_both())

    assert first_context.user_data["search_query"] == "First Search"
    assert second_context.user_data["search_query"] == "Second Search"
    assert first_update.message.replies[0][0].startswith('🔎 Search for')
    first_markup = first_update.message.edited_text[0][1]["reply_markup"]
    second_markup = second_update.message.edited_text[0][1]["reply_markup"]
    assert any("_u101" in button.callback_data for row in first_markup.inline_keyboard for button in row)
    assert any("_u202" in button.callback_data for row in second_markup.inline_keyboard for button in row)


def test_group_result_callback_is_restricted_to_requester(bot_module):
    bot = bot_module
    message = _FakeMessage(chat_id=-1001)
    callback = _FakeCallback("movie_5_u101", message=message, user_id=202)
    context = _FakeContext()

    _run(bot.button_callback(_callback_update(callback, chat_id=-1001), context))

    assert callback.answered
    assert callback.answered[0][0][0].startswith("✋")
    assert not callback.edited_text


def test_malformed_group_selection_callback_fails_closed(bot_module):
    bot = bot_module
    message = _FakeMessage(chat_id=-1001)
    callback = _FakeCallback("movie_5", message=message, user_id=202)
    context = _FakeContext()

    _run(bot.button_callback(_callback_update(callback, chat_id=-1001), context))

    assert callback.answered
    assert "no longer available" in callback.answered[0][0][0]


def test_group_handler_is_registered_for_groups_and_supergroups(bot_module):
    bot = bot_module
    source = Path(bot.__file__).read_text(encoding="utf-8")

    assert "filters.ChatType.GROUPS, handle_group_message" in source
    assert "No matching movie found" in source


def test_search_cache_uses_normalized_key_and_skips_db_on_hit(bot_module, monkeypatch):
    bot = bot_module
    bot.search_cache.clear()
    calls = []

    class Cursor:
        def execute(self, query, params):
            calls.append((query, params))

        def fetchall(self):
            return [(7, "Spider-Man", "url", "file", None, None, 2002, "Action")]

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(bot, "get_db_connection", lambda: Connection())
    monkeypatch.setattr(bot, "close_db_connection", lambda _conn: None)

    first = bot.get_movies_fast_sql("Spider-Man", limit=5)
    second = bot.get_movies_fast_sql("spider man", limit=5)

    assert first == second
    assert len(calls) == 1
    bot.search_cache.clear()


def test_exact_match_skips_prefix_fuzzy_google_and_ai(bot_module, monkeypatch):
    bot = bot_module
    bot.search_cache.clear()
    calls = []

    class Cursor:
        def execute(self, query, params):
            calls.append(query)

        def fetchall(self):
            return [(8, "Dune", "url", "file", None, None, 2021, "Sci-Fi")]

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(bot, "get_db_connection", lambda: Connection())
    monkeypatch.setattr(bot, "close_db_connection", lambda _conn: None)

    result = bot.get_movies_fast_sql("Dune")

    assert result[0][1] == "Dune"
    assert len(calls) == 1
    assert "SIMILARITY" not in calls[0]
    assert "LIKE %s" in calls[0]
    assert "CASE" in calls[0]
    bot.search_cache.clear()


def test_fuzzy_stage_runs_only_after_exact_and_prefix_fail(bot_module, monkeypatch):
    bot = bot_module
    bot.search_cache.clear()
    calls = []

    class Cursor:
        def execute(self, query, params):
            calls.append(query)

        def fetchall(self):
            query = calls[-1]
            if "SIMILARITY" in query:
                return [(9, "Reacher", "url", "file", None, None, 2022, "Action")]
            return []

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(bot, "get_db_connection", lambda: Connection())
    monkeypatch.setattr(bot, "close_db_connection", lambda _conn: None)

    result = bot.get_movies_fast_sql("Rechar")

    assert result[0][1] == "Reacher"
    assert len(calls) == 2
    assert "SIMILARITY" not in calls[0]
    assert "SIMILARITY" in calls[1]
    bot.search_cache.clear()


def test_search_timing_reports_pool_and_sql_stages(bot_module, monkeypatch, caplog):
    bot = bot_module
    bot.search_cache.clear()

    class Cursor:
        def execute(self, _query, _params):
            pass

        def fetchall(self):
            return [(10, "Arrival", "url", "file", None, None, 2016, "Drama")]

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(bot, "get_db_connection", lambda: Connection())
    monkeypatch.setattr(bot, "close_db_connection", lambda _conn: None)
    monkeypatch.setattr(bot, "SEARCH_TIMING_ENABLED", True)

    with caplog.at_level("INFO", logger=bot.logger.name):
        result = bot.get_movies_fast_sql("Arrival")

    assert result[0][1] == "Arrival"
    assert "connection_acquire_ms=" in caplog.text
    assert "exact_prefix_sql_ms=" in caplog.text
    assert "fuzzy_title_sql_ms=0.00" in caplog.text


def test_group_progress_is_sent_before_search_starts(bot_module, monkeypatch):
    bot = bot_module
    update = _group_update("Slow Search", 333)
    context = _FakeContext()
    search_started = []

    async def wait_for_search(_function, _query, **_kwargs):
        search_started.append(bool(update.message.replies))
        return []

    monkeypatch.setattr(bot, "run_async", wait_for_search)
    monkeypatch.setattr(bot, "schedule_recommendation_event", lambda **_kwargs: None)

    _run(bot.handle_group_message(update, context))

    assert search_started == [True]
