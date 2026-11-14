"""Tests for the on-demand lyrics lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ProviderFeature
from music_assistant_models.media_items import ProviderMapping
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.controllers.music import MusicController
from music_assistant.models.music_provider import MusicProvider
from tests.controllers.music.helpers import create_track

if TYPE_CHECKING:
    from music_assistant.controllers.metadata import MetaDataController


PLAIN_LYRICS = "plain lyrics"
SYNCED_LYRICS = "[00:01.00]synchronized lyrics"
METADATA_LYRICS = "metadata lyrics"
PROVIDER_PLAIN_LYRICS = "provider plain lyrics"
METADATA_SYNCED_LYRICS = "[00:02.00]metadata synchronized lyrics"


def _set_prefer_synced_lyrics(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    """Set the on-demand synchronized lyrics preference for a test."""
    monkeypatch.setattr(
        metadata_controller.config,
        "get_value",
        lambda key, default=None: enabled if key == "prefer_synced_lyrics" else default,
    )


def _metadata_provider(name: str, *results: MediaItemMetadata | Exception) -> MagicMock:
    """Build a metadata provider returning the given results in order."""
    provider = MagicMock()
    provider.name = name
    provider.priority = 50
    provider.supported_features = {ProviderFeature.LYRICS}
    provider.get_track_metadata = AsyncMock(side_effect=results)
    return provider


def _setup_metadata_providers(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    *providers: MagicMock,
    visible_provider: MagicMock | None = None,
) -> MagicMock:
    """Set up provider access for an on-demand lookup."""
    mass = metadata_controller.mass
    mass.music = MusicController(mass)
    get_visible_provider = MagicMock(return_value=visible_provider)
    monkeypatch.setattr(mass.music, "get_visible_provider", get_visible_provider)
    monkeypatch.setattr(mass, "get_providers", MagicMock(return_value=list(providers)))
    return get_visible_provider


async def test_plain_lyrics_remain_the_fallback_when_preference_is_disabled(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the preference, the first available lyrics are returned as before."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, False)
    plain_provider = _metadata_provider("plain", MediaItemMetadata(lyrics=PLAIN_LYRICS))
    synced_provider = _metadata_provider("synced", MediaItemMetadata(lrc_lyrics=SYNCED_LYRICS))
    _setup_metadata_providers(metadata_controller, monkeypatch, plain_provider, synced_provider)

    result = await metadata_controller.get_track_lyrics(create_track("spotify_test", "track"))

    assert result == (PLAIN_LYRICS, None)
    plain_provider.get_track_metadata.assert_awaited_once()
    synced_provider.get_track_metadata.assert_not_awaited()


async def test_preference_continues_after_plain_lyrics_and_keeps_them_as_fallback(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later synchronized result wins while the first plain lyrics remain available."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    plain_provider = _metadata_provider("plain", MediaItemMetadata(lyrics=PLAIN_LYRICS))
    synced_provider = _metadata_provider("synced", MediaItemMetadata(lrc_lyrics=SYNCED_LYRICS))
    _setup_metadata_providers(metadata_controller, monkeypatch, plain_provider, synced_provider)

    result = await metadata_controller.get_track_lyrics(create_track("spotify_test", "track"))

    assert result == (PLAIN_LYRICS, SYNCED_LYRICS)
    plain_provider.get_track_metadata.assert_awaited_once()
    synced_provider.get_track_metadata.assert_awaited_once()


async def test_first_plain_lyrics_are_returned_when_no_provider_has_synced_lyrics(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earliest plain result remains the fallback if the search finds no LRC."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    first_lyrics = "first plain lyrics"
    later_lyrics = "later plain lyrics"
    first_provider = _metadata_provider("first", MediaItemMetadata(lyrics=first_lyrics))
    next_provider = _metadata_provider("next", MediaItemMetadata(lyrics=later_lyrics))
    _setup_metadata_providers(metadata_controller, monkeypatch, first_provider, next_provider)

    result = await metadata_controller.get_track_lyrics(create_track("spotify_test", "track"))

    assert result == (first_lyrics, None)
    first_provider.get_track_metadata.assert_awaited_once()
    next_provider.get_track_metadata.assert_awaited_once()


@pytest.mark.parametrize(
    "metadata",
    [
        MediaItemMetadata(lrc_lyrics=SYNCED_LYRICS),
        MediaItemMetadata(lyrics=SYNCED_LYRICS),
    ],
    ids=["lrc-field", "timestamped-plain-field"],
)
async def test_synchronized_lyrics_stop_provider_search(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    metadata: MediaItemMetadata,
) -> None:
    """Both synchronized fields are recognized, including LRC text in the plain field."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    first_provider = _metadata_provider("first", metadata)
    later_lyrics = "later lyrics"
    next_provider = _metadata_provider("next", MediaItemMetadata(lyrics=later_lyrics))
    _setup_metadata_providers(metadata_controller, monkeypatch, first_provider, next_provider)

    result = await metadata_controller.get_track_lyrics(create_track("spotify_test", "track"))

    assert result == (
        SYNCED_LYRICS if metadata.lyrics else None,
        SYNCED_LYRICS,
    )
    next_provider.get_track_metadata.assert_not_awaited()


async def test_provider_failure_does_not_prevent_later_synchronized_lyrics(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed lyrics provider is skipped while the ordered search continues."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    broken_provider = _metadata_provider("broken", RuntimeError("provider unavailable"))
    synced_provider = _metadata_provider("synced", MediaItemMetadata(lrc_lyrics=SYNCED_LYRICS))
    _setup_metadata_providers(metadata_controller, monkeypatch, broken_provider, synced_provider)

    assert await metadata_controller.get_track_lyrics(create_track("spotify_test", "track")) == (
        None,
        SYNCED_LYRICS,
    )
    synced_provider.get_track_metadata.assert_awaited_once()


async def test_provider_search_does_not_mutate_track_lyrics(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing plain lyrics are retained on the caller while providers receive a clean copy."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    synced_provider = _metadata_provider("synced", MediaItemMetadata(lrc_lyrics=SYNCED_LYRICS))
    _setup_metadata_providers(metadata_controller, monkeypatch, synced_provider)
    track = create_track("spotify_test", "track")
    cached_lyrics = "cached plain lyrics"
    track.metadata.lyrics = cached_lyrics

    result = await metadata_controller.get_track_lyrics(track)

    provider_track = synced_provider.get_track_metadata.await_args.args[0]
    assert provider_track is not track
    assert provider_track.metadata.lyrics is None
    assert track.metadata.lyrics == cached_lyrics
    assert result == (cached_lyrics, SYNCED_LYRICS)


async def test_cached_synchronized_lyrics_stop_source_lookup(
    metadata_controller: MetaDataController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synchronized lyrics already on the supplied track are immediately authoritative."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    provider = _metadata_provider("metadata", MediaItemMetadata(lyrics="other lyrics"))
    visible_provider = _setup_metadata_providers(metadata_controller, monkeypatch, provider)
    track = create_track("spotify_test", "track")
    cached_synced_lyrics = "[00:01.00]cached synchronized lyrics"
    track.metadata.lrc_lyrics = cached_synced_lyrics

    assert await metadata_controller.get_track_lyrics(track) == (
        None,
        cached_synced_lyrics,
    )
    visible_provider.assert_not_called()
    provider.get_track_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    ("prefer_synced", "expected"),
    [(False, (None, None)), (True, (METADATA_LYRICS, None))],
    ids=["default-stops", "preference-searches"],
)
async def test_library_without_lyrics_only_searches_when_preference_is_enabled(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    prefer_synced: bool,
    expected: tuple[str | None, str | None],
) -> None:
    """A library refresh without lyrics stops by default and searches when opted in."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, prefer_synced)
    provider = _metadata_provider("metadata", MediaItemMetadata(lyrics=METADATA_LYRICS))
    _setup_metadata_providers(metadata_controller, monkeypatch, provider)
    library_track = create_track("spotify_test", "track")
    library_track.provider = "library"
    library_track.item_id = "library-track"
    get_track = AsyncMock(return_value=library_track)
    monkeypatch.setattr(metadata_controller.mass.music.tracks, "get", get_track)
    refresh = AsyncMock()
    monkeypatch.setattr(metadata_controller, "_update_track_metadata", refresh)

    assert await metadata_controller.get_track_lyrics(library_track) == expected

    get_track.assert_awaited_once_with("library-track", "library", allow_update_metadata=False)
    refresh.assert_awaited_once_with(library_track, force_refresh=False)
    if prefer_synced:
        provider.get_track_metadata.assert_awaited_once()
    else:
        provider.get_track_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    ("prefer_synced", "expected"),
    [
        (False, (PROVIDER_PLAIN_LYRICS, None)),
        (True, (PROVIDER_PLAIN_LYRICS, METADATA_SYNCED_LYRICS)),
    ],
    ids=["default-stops", "preference-searches"],
)
async def test_plain_track_provider_lyrics_only_stop_search_when_preference_is_disabled(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    prefer_synced: bool,
    expected: tuple[str | None, str | None],
) -> None:
    """Own-provider lyrics stop by default and remain fallback when searching for sync."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, prefer_synced)
    track_provider = MagicMock(spec=MusicProvider)
    track_provider.supported_features = {ProviderFeature.LYRICS}
    provider_track = create_track("spotify_test", "track", name="Fetched track")
    provider_track.metadata.lyrics = PROVIDER_PLAIN_LYRICS
    get_provider_item = AsyncMock(return_value=provider_track)
    metadata_provider = _metadata_provider(
        "metadata", MediaItemMetadata(lrc_lyrics=METADATA_SYNCED_LYRICS)
    )
    _setup_metadata_providers(
        metadata_controller,
        monkeypatch,
        metadata_provider,
        visible_provider=track_provider,
    )
    mass = metadata_controller.mass
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", get_provider_item)

    result = await metadata_controller.get_track_lyrics(create_track("spotify_test", "track"))

    assert result == expected
    get_provider_item.assert_awaited_once_with("track", "spotify_test")
    if prefer_synced:
        metadata_provider.get_track_metadata.assert_awaited_once()
        searched_track = metadata_provider.get_track_metadata.await_args.args[0]
        assert searched_track.name == "Fetched track"
        assert searched_track.metadata.lyrics is None
        assert provider_track.metadata.lyrics == PROVIDER_PLAIN_LYRICS
    else:
        metadata_provider.get_track_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    ("local_lyrics", "local_lrc_lyrics"),
    [("local tag lyrics", None), (None, "plain sidecar lyrics")],
    ids=["embedded-tag", "plain-sidecar"],
)
async def test_filesystem_lyrics_override_remote_results_for_library_track(
    metadata_controller: MetaDataController,
    monkeypatch: pytest.MonkeyPatch,
    local_lyrics: str | None,
    local_lrc_lyrics: str | None,
) -> None:
    """Visible local tags or sidecars remain authoritative for mapped library tracks."""
    _set_prefer_synced_lyrics(metadata_controller, monkeypatch, True)
    filesystem = MagicMock(spec=MusicProvider)
    filesystem.domain = "filesystem_local"
    filesystem.instance_id = "filesystem_local_test"
    filesystem.name = "Local files"
    remote = _metadata_provider(
        "remote", MediaItemMetadata(lrc_lyrics="[00:02.00]remote synchronized lyrics")
    )
    _setup_metadata_providers(
        metadata_controller,
        monkeypatch,
        remote,
        visible_provider=filesystem,
    )
    mass = metadata_controller.mass

    track = create_track("spotify_test", "track")
    track.provider = "library"
    track.item_id = "library-track"
    track.provider_mappings = {
        ProviderMapping(
            item_id="local-track",
            provider_domain="filesystem_local",
            provider_instance="filesystem_local_test",
        )
    }
    cached_lyrics = "cached plain lyrics"
    track.metadata.lyrics = cached_lyrics
    local_track = create_track("filesystem_local_test", "local-track")
    local_track.metadata.lyrics = local_lyrics
    local_track.metadata.lrc_lyrics = local_lrc_lyrics
    get_provider_item = AsyncMock(return_value=local_track)
    monkeypatch.setattr(mass.music.tracks, "get_provider_item", get_provider_item)
    monkeypatch.setattr(mass.music.tracks, "get", AsyncMock(return_value=track))
    monkeypatch.setattr(metadata_controller, "_update_track_metadata", AsyncMock())

    assert await metadata_controller.get_track_lyrics(track) == (local_lyrics, local_lrc_lyrics)
    get_provider_item.assert_awaited_once()
    remote.get_track_metadata.assert_not_awaited()
    assert track.metadata.lyrics == cached_lyrics
