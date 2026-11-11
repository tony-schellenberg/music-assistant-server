"""Tests for metadata lyrics selection and source precedence."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from music_assistant_models.enums import ProviderFeature
from music_assistant_models.media_items import ProviderMapping
from music_assistant_models.media_items.metadata import MediaItemMetadata

from music_assistant.controllers.metadata.lyrics import (
    LyricsSelector,
    iter_music_lyrics_candidates,
    track_without_deferred_lyrics,
)
from music_assistant.models.music_provider import MusicProvider
from tests.controllers.music.helpers import create_track


def test_selector_keeps_first_plain_fallback_until_synced_lyrics_are_found() -> None:
    """The first plain lyrics remain a fallback while later sources are considered."""
    selector = LyricsSelector(prefer_synced=True)
    selector.consider("first plain", None)
    selector.consider("second plain", None)

    assert selector.final_lyrics is None
    fallback = selector.best_lyrics
    assert fallback == ("first plain", None)

    selector.consider(None, "[00:01.00]synchronized")
    selector.consider("later plain", None)

    assert selector.is_final
    selected = selector.best_lyrics
    assert selected == (None, "[00:01.00]synchronized")


def test_selector_accepts_embedded_lrc_and_plain_lyrics_when_preference_is_disabled() -> None:
    """Embedded LRC is synchronized and disabled preference selects the first result."""
    synced = LyricsSelector(prefer_synced=True)
    synced.consider("[00:01.00]embedded synchronized", None)
    assert synced.is_final

    plain = LyricsSelector(prefer_synced=False)
    plain.consider("plain lyrics", None)
    plain.consider(None, "[00:01.00]later synchronized")
    assert plain.best_lyrics == ("plain lyrics", None)


def test_selector_ignores_empty_lyrics() -> None:
    """Empty results do not complete selection or replace a fallback."""
    selector = LyricsSelector(prefer_synced=True)
    selector.consider(None, None)
    selector.consider("first plain", None)
    selector.consider(None, None)

    assert not selector.is_final
    assert selector.best_lyrics == ("first plain", None)


def test_consider_metadata_returns_whether_selection_is_final() -> None:
    """Selection reports when synchronized lyrics complete the search."""
    selector = LyricsSelector(prefer_synced=True)

    assert not selector.consider_metadata(MediaItemMetadata(lyrics="plain lyrics"))
    assert selector.consider_metadata(MediaItemMetadata(lrc_lyrics="[00:01.00]synced"))
    assert selector.consider_metadata(MediaItemMetadata())


def test_lookup_track_hides_deferred_lyrics_without_mutating_original() -> None:
    """Providers can be given a lyrics-free copy while synchronized lookup continues."""
    track = create_track("spotify", "track")
    track.metadata = MediaItemMetadata(
        lyrics="plain lyrics", lrc_lyrics="original LRC", copyright="copyright"
    )
    selector = LyricsSelector(prefer_synced=True)
    selector.consider("plain lyrics", None)

    lookup_track = track_without_deferred_lyrics(track, selector)

    assert lookup_track is not track
    assert (lookup_track.item_id, lookup_track.provider) == (track.item_id, track.provider)
    assert lookup_track.metadata is not track.metadata
    assert (lookup_track.metadata.lyrics, lookup_track.metadata.lrc_lyrics) == (None, None)
    assert lookup_track.metadata.copyright == "copyright"
    assert (track.metadata.lyrics, track.metadata.lrc_lyrics) == ("plain lyrics", "original LRC")


def test_metadata_with_lyrics_returns_a_new_metadata_object() -> None:
    """Applying selected lyrics returns a copy and leaves the input unchanged."""
    metadata = MediaItemMetadata(lyrics="original lyrics", copyright="copyright")
    selector = LyricsSelector(prefer_synced=True)
    selector.consider_metadata(MediaItemMetadata(lrc_lyrics="[00:01.00]synchronized"))

    updated = selector.metadata_with_lyrics(metadata)

    assert updated is not metadata
    assert updated.lyrics is None
    assert updated.lrc_lyrics == "[00:01.00]synchronized"
    assert updated.copyright == "copyright"
    assert metadata.lyrics == "original lyrics"
    assert metadata.lrc_lyrics is None


def test_metadata_with_lyrics_can_include_plain_fallback() -> None:
    """Fallback lyrics are applied only when explicitly requested."""
    selector = LyricsSelector(prefer_synced=True)
    selector.consider_metadata(MediaItemMetadata(lyrics="plain fallback"))
    metadata = MediaItemMetadata(copyright="copyright")

    assert selector.metadata_with_lyrics(metadata) is metadata
    updated = selector.metadata_with_lyrics(metadata, include_deferred=True)

    assert updated.lyrics == "plain fallback"
    assert updated.copyright == "copyright"


async def test_filesystem_lyrics_take_precedence_over_cached_library_lyrics() -> None:
    """A local file's plain lyrics are authoritative over synchronized cached lyrics."""
    mass = MagicMock()
    local_provider = MagicMock(spec=MusicProvider)
    local_provider.domain = "filesystem_local"
    local_provider.instance_id = "local_1"
    local_provider.is_streaming_provider = False
    local_provider.supported_features = set()
    mass.music.get_visible_provider.return_value = local_provider
    local_track = create_track("local_1", "local-track")
    local_track.metadata.lyrics = "local plain lyrics"
    mass.music.tracks.get_provider_item = AsyncMock(return_value=local_track)

    library_track = create_track("library", "library-track")
    library_track.provider = "library"
    library_track.provider_mappings = {
        ProviderMapping(
            item_id="local-track",
            provider_domain="filesystem_local",
            provider_instance="local_1",
        )
    }
    library_track.metadata.lrc_lyrics = "[00:01.00]cached synchronized lyrics"
    selector = LyricsSelector(prefer_synced=True)

    async for metadata, authoritative in iter_music_lyrics_candidates(mass, library_track):
        selector.consider(metadata.lyrics, metadata.lrc_lyrics, authoritative=authoritative)

    assert selector.best_lyrics == ("local plain lyrics", None)
    mass.music.tracks.get_provider_item.assert_awaited_once_with("local-track", "local_1")


async def test_direct_track_sources_include_cached_and_fresh_provider_metadata() -> None:
    """Direct tracks yield cached metadata before fresh provider metadata."""
    mass = MagicMock()
    provider = MagicMock(spec=MusicProvider)
    provider.domain = "spotify"
    provider.supported_features = {ProviderFeature.LYRICS}
    mass.music.get_visible_provider.return_value = provider
    fresh_track = create_track("spotify", "track")
    fresh_track.metadata.lyrics = "fresh lyrics"
    mass.music.tracks.get_provider_item = AsyncMock(return_value=fresh_track)

    track = create_track("spotify", "track")
    track.metadata.lyrics = "cached lyrics"
    sources = [
        (metadata.lyrics, authoritative)
        async for metadata, authoritative in iter_music_lyrics_candidates(mass, track)
    ]

    assert sources == [("cached lyrics", False), ("fresh lyrics", False)]
    mass.music.tracks.get_provider_item.assert_awaited_once_with("track", "spotify")


async def test_direct_filesystem_lyrics_are_authoritative() -> None:
    """Direct filesystem tracks expose cached and fresh metadata as authoritative."""
    mass = MagicMock()
    provider = MagicMock(spec=MusicProvider)
    provider.domain = "filesystem_local"
    mass.music.get_visible_provider.return_value = provider
    fresh_track = create_track("local_1", "track")
    fresh_track.metadata.lyrics = "file lyrics"
    mass.music.tracks.get_provider_item = AsyncMock(return_value=fresh_track)

    track = create_track("local_1", "track")
    track.metadata.lyrics = "cached file lyrics"
    sources = [
        (metadata.lyrics, authoritative)
        async for metadata, authoritative in iter_music_lyrics_candidates(mass, track)
    ]

    assert sources == [("cached file lyrics", True), ("file lyrics", True)]


async def test_library_mapping_misses_do_not_yield_library_items() -> None:
    """A library fallback from provider lookup is not fresh mapping metadata."""
    mass = MagicMock()
    provider = MagicMock(spec=MusicProvider)
    provider.domain = "spotify"
    provider.is_streaming_provider = True
    provider.supported_features = {ProviderFeature.LYRICS}
    mass.music.get_visible_provider.return_value = provider
    fallback = create_track("library", "library-track")
    fallback.metadata.lyrics = "stale library lyrics"
    mass.music.tracks.get_provider_item = AsyncMock(return_value=fallback)

    track = create_track("library", "library-track")
    track.provider = "library"
    track.provider_mappings = {
        ProviderMapping(
            item_id="spotify-track",
            provider_domain="spotify",
            provider_instance="spotify_1",
        )
    }
    track.metadata.lyrics = "stored library lyrics"
    sources = [
        metadata.lyrics
        async for metadata, _authoritative in iter_music_lyrics_candidates(mass, track)
    ]

    assert sources == ["stored library lyrics"]
