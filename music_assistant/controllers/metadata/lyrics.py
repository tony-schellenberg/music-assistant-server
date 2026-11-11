"""Helpers for selecting lyrics from metadata sources."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.enums import ProviderFeature

from music_assistant.helpers.lyrics import extract_lrc_lyrics
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from music_assistant_models.media_items import MediaItemMetadata, Track

    from music_assistant import MusicAssistant


class LyricsSelector:
    """
    Select lyrics from candidates supplied in precedence order.

    Callers supply filesystem lyrics first, then stored lyrics, other music sources,
    and finally metadata sources. Direct provider tracks use their cached lyrics
    before freshly fetched lyrics from that provider.

    With synchronized lyrics preferred, remote plain lyrics remain a fallback while
    later sources are checked. Authoritative sources (filesystem music and local
    metadata) can finalize plain lyrics, but never replace an already-final result.
    Without that preference, the first nonempty candidate is final.
    """

    def __init__(self, prefer_synced: bool) -> None:
        """Initialize selection with the remote-source synchronization preference."""
        self.prefer_synced = prefer_synced
        self._deferred: tuple[str | None, str | None] | None = None
        self._final: tuple[str | None, str | None] | None = None

    @property
    def is_final(self) -> bool:
        """Return whether a preferred lyric result has been found."""
        return self._final is not None

    @property
    def should_search_for_synced(self) -> bool:
        """Return whether further sources may provide preferred synchronized lyrics."""
        return self.prefer_synced and self._final is None

    @property
    def best_lyrics(self) -> tuple[str | None, str | None] | None:
        """Return final lyrics, or the first deferred plain lyrics if none are final."""
        return self._final if self._final is not None else self._deferred

    @property
    def final_lyrics(self) -> tuple[str | None, str | None] | None:
        """Return final lyrics, excluding a plain result still being deferred."""
        return self._final

    def consider(
        self,
        lyrics: str | None,
        lrc_lyrics: str | None,
        *,
        authoritative: bool = False,
    ) -> None:
        """Consider lyrics from a source, retaining only the preferred result."""
        if self.is_final:
            return
        if not lyrics and not lrc_lyrics:
            return
        candidate = (lyrics, lrc_lyrics)
        if authoritative or not self.prefer_synced or lrc_lyrics or extract_lrc_lyrics(lyrics):
            self._final = candidate
        elif self._deferred is None:
            self._deferred = candidate

    def consider_metadata(
        self, metadata: MediaItemMetadata, *, authoritative: bool = False
    ) -> bool:
        """Consider lyrics from a metadata result and return whether selection is complete."""
        self.consider(metadata.lyrics, metadata.lrc_lyrics, authoritative=authoritative)
        return self.is_final

    def apply_selected_lyrics(
        self, metadata: MediaItemMetadata, *, include_deferred: bool = False
    ) -> MediaItemMetadata:
        """
        Return metadata with selected lyrics, leaving it unchanged if none qualify.

        Unlike track_without_deferred_lyrics, this does not hide existing lyrics.

        :param metadata: Metadata to copy when applying a selected result.
        :param include_deferred: Allow the plain fallback when no final result exists.
            If false, existing lyrics are still kept when no final result exists.
        """
        lyrics = self.best_lyrics if include_deferred else self.final_lyrics
        if lyrics is None:
            return metadata
        return replace(metadata, lyrics=lyrics[0], lrc_lyrics=lyrics[1])


def track_without_deferred_lyrics(track: Track, selector: LyricsSelector) -> Track:
    """Hide deferred plain lyrics from providers that short-circuit on track lyrics."""
    if not selector.should_search_for_synced:
        return track
    return replace(track, metadata=replace(track.metadata, lyrics=None, lrc_lyrics=None))


def is_filesystem_provider(provider: object) -> bool:
    """Return whether a provider is a filesystem music source."""
    return isinstance(provider, MusicProvider) and provider.domain.startswith("filesystem_")


async def iter_music_lyrics_candidates(
    mass: MusicAssistant, track: Track
) -> AsyncIterator[tuple[MediaItemMetadata, bool]]:
    """Yield music-source metadata and whether each source is authoritative."""
    if track.provider == "library":
        async for metadata in _iter_library_music_metadata(mass, track, filesystem=True):
            yield metadata, True
        # Stored lyrics have no source provenance, so consult local mappings first.
        yield track.metadata, False
        async for metadata in _iter_library_music_metadata(mass, track, filesystem=False):
            yield metadata, False
        return

    provider = mass.music.get_visible_provider(track.provider)
    is_filesystem = is_filesystem_provider(provider)
    fetch_provider_metadata = is_filesystem or (
        isinstance(provider, MusicProvider)
        and ProviderFeature.LYRICS in provider.supported_features
    )
    async for metadata in _iter_provider_track_metadata(
        mass, track, fetch_provider_metadata=fetch_provider_metadata
    ):
        yield metadata, is_filesystem


async def _iter_provider_track_metadata(
    mass: MusicAssistant, track: Track, *, fetch_provider_metadata: bool
) -> AsyncIterator[MediaItemMetadata]:
    """Yield cached and freshly fetched metadata for a provider track."""
    yield track.metadata
    if not fetch_provider_metadata:
        return
    full_track = await mass.music.tracks.get_provider_item(track.item_id, track.provider)
    if full_track.provider != "library":
        yield full_track.metadata


async def _iter_library_music_metadata(
    mass: MusicAssistant, track: Track, *, filesystem: bool
) -> AsyncIterator[MediaItemMetadata]:
    """Yield full track metadata from visible mappings of the requested source type."""
    mappings = sorted(track.provider_mappings, key=lambda mapping: mapping.priority, reverse=True)
    seen_sources: set[str] = set()
    for mapping in mappings:
        provider = mass.music.get_visible_provider(mapping.provider_instance)
        if (
            not isinstance(provider, MusicProvider)
            or is_filesystem_provider(provider) != filesystem
        ):
            continue
        if not filesystem and ProviderFeature.LYRICS not in provider.supported_features:
            continue
        source_key = provider.domain if provider.is_streaming_provider else provider.instance_id
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        full_track = await mass.music.tracks.get_provider_item(
            mapping.item_id, mapping.provider_instance
        )
        if full_track.provider != "library":
            yield full_track.metadata
