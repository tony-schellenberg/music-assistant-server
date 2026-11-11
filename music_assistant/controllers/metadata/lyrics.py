"""Helpers for selecting lyrics from metadata sources."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from music_assistant_models.enums import ProviderFeature

from music_assistant.helpers.lyrics import extract_lrc_lyrics
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from music_assistant_models.media_items import MediaItemMetadata, Track

    from music_assistant import MusicAssistant


class LyricsSelector:
    """Select the first authoritative or preferred lyric result."""

    def __init__(self, prefer_synced: bool) -> None:
        """Initialize selection with the remote-source synchronization preference."""
        self.prefer_synced = prefer_synced
        self._fallback: tuple[str | None, str | None] | None = None
        self._selected: tuple[str | None, str | None] | None = None

    @property
    def should_continue(self) -> bool:
        """Return whether another source may yield preferred synchronized lyrics."""
        return self.prefer_synced and self._selected is None

    @property
    def result(self) -> tuple[str | None, str | None] | None:
        """Return the selected lyrics, or the first plain fallback found."""
        return self._selected if self._selected is not None else self._fallback

    def add(
        self,
        lyrics: str | None,
        lrc_lyrics: str | None,
        *,
        authoritative: bool = False,
    ) -> bool:
        """Add lyrics from the next source and report whether lookup can stop."""
        if self._selected is not None:
            return True
        if not lyrics and not lrc_lyrics:
            return False
        candidate = (lyrics, lrc_lyrics)
        if authoritative or not self.prefer_synced or lrc_lyrics or extract_lrc_lyrics(lyrics):
            self._selected = candidate
            return True
        if self._fallback is None:
            self._fallback = candidate
        return False


def track_without_lyrics(track: Track) -> Track:
    """Return a copy of a track without lyrics, for providers that short-circuit on them."""
    result = deepcopy(track)
    result.metadata.lyrics = None
    result.metadata.lrc_lyrics = None
    return result


async def iter_music_lyrics(
    mass: MusicAssistant, track: Track
) -> AsyncIterator[tuple[MediaItemMetadata, bool]]:
    """Yield visible music-source lyrics in precedence order, with local lyrics marked."""
    if track.provider != "library":
        provider = mass.music.get_visible_provider(track.provider)
        is_filesystem = isinstance(provider, MusicProvider) and provider.domain.startswith(
            "filesystem_"
        )
        yield track.metadata, is_filesystem
        if isinstance(provider, MusicProvider) and (
            is_filesystem or ProviderFeature.LYRICS in provider.supported_features
        ):
            full_track = await mass.music.tracks.get_provider_item(track.item_id, track.provider)
            if full_track.provider != "library":
                yield full_track.metadata, is_filesystem
        return

    mappings = sorted(track.provider_mappings, key=lambda mapping: mapping.priority, reverse=True)
    seen_sources: set[str] = set()
    for filesystem_only in (True, False):
        if not filesystem_only:
            # Stored lyrics have no source provenance, so consult local mappings first.
            yield track.metadata, False
        for mapping in mappings:
            provider = mass.music.get_visible_provider(mapping.provider_instance)
            if not isinstance(provider, MusicProvider):
                continue
            is_filesystem = provider.domain.startswith("filesystem_")
            if is_filesystem != filesystem_only or (
                not is_filesystem and ProviderFeature.LYRICS not in provider.supported_features
            ):
                continue
            source_key = provider.domain if provider.is_streaming_provider else provider.instance_id
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            full_track = await mass.music.tracks.get_provider_item(
                mapping.item_id, mapping.provider_instance
            )
            if full_track.provider != "library":
                yield full_track.metadata, is_filesystem
