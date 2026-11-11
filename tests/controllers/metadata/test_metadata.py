"""Tests for the MetaDataController public API."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.config_entries import ProviderAccess
from music_assistant_models.enums import ProviderFeature, ProviderSharing, ProviderType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import MediaItemPalette
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.controllers.metadata.constants import (
    CONF_ENABLE_ONLINE_METADATA,
    CONF_PREFER_SYNCED_LYRICS,
)
from music_assistant.controllers.music import MusicController
from music_assistant.controllers.webserver.helpers.auth_middleware import (
    current_user,
    get_current_user,
    impersonated_user,
)
from music_assistant.models.music_provider import MusicProvider
from tests.common import set_music_source_access
from tests.controllers.music.helpers import create_track

if TYPE_CHECKING:
    from music_assistant.controllers.metadata import MetaDataController
    from music_assistant.mass import MusicAssistant

# every test here registers image ids, which are persisted to the cache database
pytestmark = pytest.mark.usefixtures("cache_database")


async def test_get_image_palette_resolves_registered_id(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered image id resolves to its (provider, path) and yields its palette."""
    palette = MediaItemPalette(primary=(1, 2, 3))
    calls: dict[str, object] = {}

    async def fake_get_palette(_mass: object, path: str, provider: str) -> MediaItemPalette:
        calls["get_palette"] = (path, provider)
        return palette

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    image_id = metadata_controller.compute_image_id("spotify", "cover.jpg")
    assert await metadata_controller.get_image_palette(image_id) is palette
    assert calls["get_palette"] == ("cover.jpg", "spotify")


async def test_get_image_palette_rejects_imageproxy_url(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a bare image id is accepted; a full /imageproxy/<id> URL yields None."""
    fetched = False

    async def fake_get_palette(_mass: object, _path: str, _provider: str) -> MediaItemPalette:
        nonlocal fetched
        fetched = True
        return MediaItemPalette(primary=(4, 5, 6))

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    image_id = metadata_controller.compute_image_id("filesystem", "/x.jpg")
    url = f"http://mass.local/imageproxy/{image_id}?size=256"
    assert await metadata_controller.get_image_palette(url) is None
    assert fetched is False


async def test_get_image_palette_rejects_unregistered_input(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown ids and arbitrary URLs yield None and never trigger a fetch (no SSRF)."""
    fetched = False

    async def fake_get_palette(_mass: object, _path: str, _provider: str) -> MediaItemPalette:
        nonlocal fetched
        fetched = True
        return MediaItemPalette(primary=(0, 0, 0))

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", fake_get_palette)

    # an id that was never registered
    assert await metadata_controller.get_image_palette("0" * 64) is None
    # an arbitrary URL must not be resolved or fetched
    assert await metadata_controller.get_image_palette("http://169.254.169.254/latest") is None
    assert fetched is False


async def test_get_image_palette_returns_none_for_unreadable_image(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered image whose data can't be read yields None instead of raising."""

    async def boom(_mass: object, path: str, _provider: str) -> MediaItemPalette:
        raise FileNotFoundError(path)

    monkeypatch.setattr("music_assistant.controllers.metadata.images.get_palette", boom)
    image_id = metadata_controller.compute_image_id("filesystem", "missing.jpg")
    assert await metadata_controller.get_image_palette(image_id) is None


async def test_update_metadata_runs_as_the_server(
    metadata_controller: MetaDataController,
) -> None:
    """The refresh fills the household's library item from all sources, not the caller's."""
    service = User(user_id="user-service", username="ha", role=UserRole.SERVICE)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    refreshing_users: list[User | None] = []

    async def _refresh(*_: object, **__: object) -> None:
        refreshing_users.append(get_current_user())

    metadata_controller._update_track_metadata = AsyncMock(side_effect=_refresh)  # type: ignore[method-assign]
    track = create_track("spotify_theirs", "t1")
    track.item_id, track.provider = "1", "library"

    current_user_token = current_user.set(service)
    impersonated_user_token = impersonated_user.set(member)
    try:
        await metadata_controller.update_metadata(track)
        assert refreshing_users == [None]
        assert get_current_user() is member
    finally:
        impersonated_user.reset(impersonated_user_token)
        current_user.reset(current_user_token)


async def test_lyrics_come_only_from_a_source_the_user_may_see(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """A track's own provider is asked for lyrics only when it is one of the user's sources."""
    owner = User(user_id="user-owner", username="owner", role=UserRole.USER)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    theirs = MagicMock(spec=MusicProvider)
    theirs.instance_id, theirs.domain, theirs.type = "spotify_theirs", "spotify", ProviderType.MUSIC
    theirs.available, theirs.is_streaming_provider = True, True
    theirs.supported_features = {ProviderFeature.LYRICS}
    theirs.initialized = MagicMock(is_set=MagicMock(return_value=True))
    full_track = create_track("spotify_theirs", "t1")
    full_track.metadata.lyrics = "secret"
    theirs.get_track = AsyncMock(return_value=full_track)
    mass_minimal.music = MusicController(mass_minimal)
    mass_minimal._providers = {"spotify_theirs": theirs}
    set_music_source_access(
        mass_minimal,
        {"spotify_theirs": ProviderAccess(owner=owner.user_id, sharing=ProviderSharing.PRIVATE)},
    )
    track = create_track("spotify_theirs", "t1")

    for user, lyrics in ((owner, "secret"), (member, None)):
        with (
            patch(
                "music_assistant.controllers.music.controller.get_current_user", return_value=user
            ),
            patch(
                "music_assistant.controllers.music.media.base.get_current_user", return_value=user
            ),
        ):
            assert (await metadata_controller.get_track_lyrics(track))[0] == lyrics


@pytest.mark.parametrize(
    "online_metadata",
    [
        pytest.param(False, id="online-metadata-disabled"),
        pytest.param(True, id="online-metadata-enabled"),
    ],
)
async def test_provider_track_cached_plain_lyrics_allow_synced_lookup(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    online_metadata: bool,
) -> None:
    """Direct provider lookups can replace cached plain lyrics regardless of online settings."""
    _set_lyrics_config(metadata_controller, monkeypatch, online_metadata=online_metadata)
    provider = _lyrics_provider("lyrics", MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"))
    monkeypatch.setattr(
        metadata_controller.mass, "get_providers", MagicMock(return_value=[provider])
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)
    track = create_track("spotify", "t1")
    track.metadata.lyrics = "cached plain"
    original_metadata = track.metadata

    assert await metadata_controller.get_track_lyrics(track) == (None, "[00:01.00]synchronized")

    assert track.metadata is original_metadata
    assert (track.metadata.lyrics, track.metadata.lrc_lyrics) == ("cached plain", None)
    provider.get_track_metadata.assert_awaited_once()
    lookup_track = provider.get_track_metadata.await_args.args[0]
    assert (lookup_track.item_id, lookup_track.provider) == (track.item_id, track.provider)
    assert (lookup_track.metadata.lyrics, lookup_track.metadata.lrc_lyrics) == (None, None)


@pytest.mark.parametrize(
    ("online_metadata", "expected_lyrics", "expected_lookups"),
    [
        pytest.param(False, ("cached plain", None), 0, id="offline-keeps-stored-plain"),
        pytest.param(True, (None, "[00:01.00]synchronized"), 1, id="online-prefers-synchronized"),
    ],
)
async def test_library_lyrics_lookup_respects_online_metadata_setting(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    online_metadata: bool,
    expected_lyrics: tuple[str | None, str | None],
    expected_lookups: int,
) -> None:
    """Library lookup uses stored lyrics and only consults metadata providers when enabled."""
    _set_lyrics_config(metadata_controller, monkeypatch, online_metadata=online_metadata)
    provider = _lyrics_provider("lyrics", MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"))
    monkeypatch.setattr(
        metadata_controller.mass, "get_providers", MagicMock(return_value=[provider])
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)
    stored_track = create_track("library", "1")
    stored_track.provider = "library"
    stored_track.metadata.lyrics = "cached plain"
    get_track = AsyncMock(return_value=stored_track)
    monkeypatch.setattr(metadata_controller.mass.music.tracks, "get", get_track)
    refresh = AsyncMock()
    monkeypatch.setattr(metadata_controller, "_update_track_metadata", refresh)
    caller_track = create_track("library", "1")
    caller_track.provider = "library"
    original_metadata = caller_track.metadata

    assert await metadata_controller.get_track_lyrics(caller_track) == expected_lyrics

    assert caller_track.metadata is original_metadata
    assert (caller_track.metadata.lyrics, caller_track.metadata.lrc_lyrics) == (None, None)
    assert stored_track.metadata.lyrics == "cached plain"
    get_track.assert_awaited_once_with("1", "library", allow_update_metadata=False)
    refresh.assert_awaited_once_with(stored_track, force_refresh=False)
    assert provider.get_track_metadata.await_count == expected_lookups
    for call in provider.get_track_metadata.await_args_list:
        lookup_track = call.args[0]
        assert (lookup_track.item_id, lookup_track.provider) == ("1", "library")
        assert (lookup_track.metadata.lyrics, lookup_track.metadata.lrc_lyrics) == (None, None)


async def test_remote_lyrics_lookup_keeps_first_plain_fallback(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first plain result is returned if later remote sources have no synchronized lyrics."""
    _set_lyrics_config(metadata_controller, monkeypatch)
    first = _lyrics_provider("first", MediaItemMetadata(lyrics="first plain"))
    second = _lyrics_provider("second", MediaItemMetadata(lyrics="second plain"), priority=1)
    monkeypatch.setattr(
        metadata_controller.mass, "get_providers", MagicMock(return_value=[first, second])
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)

    assert await metadata_controller.get_track_lyrics(create_track("spotify", "t1")) == (
        "first plain",
        None,
    )
    first.get_track_metadata.assert_awaited_once()
    second.get_track_metadata.assert_awaited_once()
    for provider in (first, second):
        lookup_track = provider.get_track_metadata.await_args.args[0]
        assert (lookup_track.metadata.lyrics, lookup_track.metadata.lrc_lyrics) == (None, None)


async def test_remote_lyrics_lookup_continues_after_provider_failure(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing lyrics provider does not prevent lookup from later providers."""
    failed = _lyrics_provider("failed", RuntimeError("provider failed"))
    succeeding = _lyrics_provider(
        "succeeding", MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"), priority=1
    )
    monkeypatch.setattr(
        metadata_controller.mass,
        "get_providers",
        MagicMock(return_value=[failed, succeeding]),
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)

    assert await metadata_controller.get_track_lyrics(create_track("spotify", "t1")) == (
        None,
        "[00:01.00]synchronized",
    )
    failed.get_track_metadata.assert_awaited_once()
    succeeding.get_track_metadata.assert_awaited_once()


async def test_remote_lyrics_lookup_skips_unsupported_and_empty_sources(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported sources and empty results do not stop the ordered lookup."""
    unsupported = MagicMock()
    unsupported.priority = 0
    unsupported.supported_features = {ProviderFeature.TRACK_METADATA}
    unsupported.get_track_metadata = AsyncMock()
    empty = _lyrics_provider("empty", MediaItemMetadata(), priority=1)
    succeeding = _lyrics_provider(
        "succeeding", MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"), priority=2
    )
    monkeypatch.setattr(
        metadata_controller.mass,
        "get_providers",
        MagicMock(return_value=[unsupported, empty, succeeding]),
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)

    assert await metadata_controller.get_track_lyrics(create_track("spotify", "t1")) == (
        None,
        "[00:01.00]synchronized",
    )
    unsupported.get_track_metadata.assert_not_awaited()
    empty.get_track_metadata.assert_awaited_once()
    succeeding.get_track_metadata.assert_awaited_once()


async def test_embedded_lrc_in_plain_lyrics_stops_remote_lookup(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LRC-formatted text in the plain lyrics field counts as synchronized."""
    _set_lyrics_config(metadata_controller, monkeypatch)
    first = _lyrics_provider("first", MediaItemMetadata(lyrics="[00:01.00]embedded synchronized"))
    second = _lyrics_provider(
        "second", MediaItemMetadata(lrc_lyrics="[00:02.00]later synchronized"), priority=1
    )
    monkeypatch.setattr(
        metadata_controller.mass, "get_providers", MagicMock(return_value=[first, second])
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)

    assert await metadata_controller.get_track_lyrics(create_track("spotify", "t1")) == (
        "[00:01.00]embedded synchronized",
        "[00:01.00]embedded synchronized",
    )
    first.get_track_metadata.assert_awaited_once()
    second.get_track_metadata.assert_not_awaited()


async def test_remote_lyrics_lookup_can_stop_at_first_plain_result(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling the preference returns the first plain result without further lookups."""
    _set_lyrics_config(metadata_controller, monkeypatch, prefer_synced=False, online_metadata=True)
    first = _lyrics_provider("first", MediaItemMetadata(lyrics="plain"))
    second = _lyrics_provider(
        "second", MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"), priority=1
    )
    monkeypatch.setattr(
        metadata_controller.mass, "get_providers", MagicMock(return_value=[first, second])
    )
    metadata_controller.mass.music = MusicController(metadata_controller.mass)

    assert await metadata_controller.get_track_lyrics(create_track("spotify", "t1")) == (
        "plain",
        None,
    )
    first.get_track_metadata.assert_awaited_once()
    second.get_track_metadata.assert_not_awaited()


async def test_lyrics_of_a_library_track_refresh_the_stored_item(
    metadata_controller: MetaDataController, mass_minimal: MusicAssistant
) -> None:
    """The caller's copy of a library track is never written back, the stored item is used."""
    owner = User(user_id="user-owner", username="owner", role=UserRole.USER)
    member = User(user_id="user-member", username="member", role=UserRole.USER)
    mass_minimal.music = MusicController(mass_minimal)
    mass_minimal.streams = MagicMock()
    mass_minimal.streams.audio_analysis.get_track_audio_metadata = AsyncMock(return_value=None)
    mass_minimal.webserver = MagicMock()
    mass_minimal.webserver.auth.get_user = AsyncMock(return_value=None)
    mass_minimal.webserver.auth.list_users = AsyncMock(return_value=[])
    # the minimal server runs no cache database
    mass_minimal.cache.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mass_minimal.cache.set = AsyncMock()  # type: ignore[method-assign]
    await mass_minimal.music._setup_database()
    try:
        set_music_source_access(
            mass_minimal,
            {
                "spotify_theirs": ProviderAccess(
                    owner=owner.user_id, sharing=ProviderSharing.PRIVATE
                )
            },
        )
        with patch.object(mass_minimal, "metadata", MagicMock()):
            stored = await mass_minimal.music.tracks.add_item_to_library(
                create_track("spotify_theirs", "t1")
            )
        refreshing_users: list[User | None] = []

        async def _refresh(*_: object, **__: object) -> None:
            refreshing_users.append(get_current_user())

        refresh = AsyncMock(side_effect=_refresh)
        metadata_controller._update_track_metadata = refresh  # type: ignore[method-assign]
        crafted = create_track("spotify_theirs", "t1", name="Crafted")
        crafted.item_id, crafted.provider = stored.item_id, "library"

        for user in (member, owner):
            user_token = current_user.set(user)
            try:
                if user is member:
                    # the only source of the track is hidden from the member
                    with pytest.raises(MediaNotFoundError):
                        await metadata_controller.get_track_lyrics(crafted)
                    refresh.assert_not_awaited()
                else:
                    assert await metadata_controller.get_track_lyrics(crafted) == (None, None)
            finally:
                current_user.reset(user_token)
        assert refresh.await_args is not None
        refreshed = refresh.await_args.args[0]
        assert (refreshed.item_id, refreshed.name) == (stored.item_id, "Test Track")
        # the refresh fills the household's item, so it runs as the server, not as the caller
        assert refreshing_users == [None]
        assert get_current_user() is None
    finally:
        await mass_minimal.music.close()


def _lyrics_provider(
    name: str, result: MediaItemMetadata | Exception | None, *, priority: int = 0
) -> MagicMock:
    """Return a remote lyrics provider with a fixed result or lookup failure."""
    provider = MagicMock()
    provider.priority = priority
    provider.supported_features = {ProviderFeature.LYRICS}
    provider.is_local = False
    provider.name = name
    if isinstance(result, Exception):
        provider.get_track_metadata = AsyncMock(side_effect=result)
    else:
        provider.get_track_metadata = AsyncMock(return_value=result)
    return provider


def _set_lyrics_config(
    controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefer_synced: bool = True,
    online_metadata: bool = False,
) -> None:
    """Configure synchronized-lyrics preference and library online lookup permissions."""
    values = {
        CONF_PREFER_SYNCED_LYRICS: prefer_synced,
        CONF_ENABLE_ONLINE_METADATA: online_metadata,
    }
    monkeypatch.setattr(
        controller.config, "get_value", lambda key, *_args, **_kwargs: values.get(key, False)
    )
