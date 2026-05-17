import logging
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.core.cache import cache
from django.db.models import Q
from django.utils import timezone

from app.models import Item, MediaTypes, Sources, UserMessage
from app.providers import services, tmdb

logger = logging.getLogger(__name__)


@shared_task(name="Cleanup user messages")
def cleanup_user_messages():
    """Delete shown user messages older than the configured retention window."""
    cutoff = timezone.now() - timedelta(days=settings.USER_MESSAGE_RETENTION_DAYS)
    deleted_count, _ = UserMessage.objects.filter(
        shown_at__isnull=False,
        shown_at__lt=cutoff,
    ).delete()

    logger.info("Deleted %s old shown user messages.", deleted_count)

    return deleted_count


@shared_task(name="Refresh TMDB metadata")
def refresh_tmdb_metadata():
    """Refresh tracked TMDB items whose metadata changed upstream.

    Uses the TMDB /changes endpoint to detect TV shows and movies that have been
    updated recently, then re-fetches and stores their metadata locally. Avoids
    a full resync of every tracked item.
    """
    # Look back over the configured cadence plus a 1-day overlap for missed
    # runs. TMDB caps the /changes window at 14 days; the settings validator
    # keeps REFRESH_TMDB_METADATA_DAYS in range, so this stays valid.
    cadence_days = settings.REFRESH_TMDB_METADATA_DAYS or 3
    window_days = min(cadence_days + 1, 14)

    try:
        changed_tv_ids = tmdb.tv_changes(days=window_days)
    except services.ProviderAPIError:
        logger.warning("Failed to fetch TMDB TV changes; skipping TV refresh.")
        changed_tv_ids = set()

    try:
        changed_movie_ids = tmdb.movie_changes(days=window_days)
    except services.ProviderAPIError:
        logger.warning("Failed to fetch TMDB movie changes; skipping movie refresh.")
        changed_movie_ids = set()

    refreshed_count = 0

    if changed_tv_ids:
        # Skip orphan items by requiring at least one related TV (or Season)
        # tracking row — matches the calendar selector's filtering pattern.
        tv_items = (
            Item.objects.filter(
                source=Sources.TMDB.value,
                media_type=MediaTypes.TV.value,
                media_id__in=changed_tv_ids,
            )
            .filter(Q(tv__isnull=False) | Q(season__isnull=False))
            .distinct()
        )
        for tv_item in tv_items:
            if _refresh_tv_item(tv_item):
                refreshed_count += 1

    if changed_movie_ids:
        movie_items = Item.objects.filter(
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            media_id__in=changed_movie_ids,
            movie__isnull=False,
        ).distinct()
        for movie_item in movie_items:
            if _refresh_movie_item(movie_item):
                refreshed_count += 1

    logger.info(
        "Refreshed metadata for %d TMDB items "
        "(%d TV / %d movie ids reported as changed).",
        refreshed_count,
        len(changed_tv_ids),
        len(changed_movie_ids),
    )
    return refreshed_count


def _tmdb_cache_key(media_type, media_id, season_number=None):
    key = f"{Sources.TMDB.value}_{media_type}_{media_id}"
    if season_number is not None:
        key += f"_{season_number}"
    return key


def _update_item_fields(item, title, image):
    """Update an Item's title/image only if changed."""
    fields = []
    if title and item.title != title:
        item.title = title
        fields.append("title")
    if image and item.image != image:
        item.image = image
        fields.append("image")
    if fields:
        item.save(update_fields=fields)
    return bool(fields)


def _refresh_movie_item(item):
    """Refresh a tracked movie Item's metadata from TMDB."""
    cache.delete(_tmdb_cache_key(MediaTypes.MOVIE.value, item.media_id))
    try:
        metadata = tmdb.movie(item.media_id)
    except services.ProviderAPIError:
        logger.warning("Failed to refresh movie %s", item)
        return False

    _update_item_fields(item, metadata.get("title"), metadata.get("image"))
    item.fetch_releases(delay=True)
    logger.info("Refreshed movie metadata for %s", item)
    return True


def _refresh_tv_item(item):
    """Refresh a tracked TV Item's metadata, plus any tracked seasons/episodes."""
    season_items = list(
        Item.objects.filter(
            source=Sources.TMDB.value,
            media_id=item.media_id,
            media_type=MediaTypes.SEASON.value,
        ),
    )
    season_numbers = sorted(
        {s.season_number for s in season_items if s.season_number is not None},
    )

    cache.delete(_tmdb_cache_key(MediaTypes.TV.value, item.media_id))
    for sn in season_numbers:
        cache.delete(_tmdb_cache_key(MediaTypes.SEASON.value, item.media_id, sn))

    try:
        if season_numbers:
            data = tmdb.tv_with_seasons(item.media_id, season_numbers)
        else:
            data = tmdb.tv(item.media_id)
    except services.ProviderAPIError:
        logger.warning("Failed to refresh TV %s", item)
        return False

    tv_title = data.get("title")
    tv_image = data.get("image")
    _update_item_fields(item, tv_title, tv_image)

    for season_item in season_items:
        season_metadata = data.get(f"season/{season_item.season_number}")
        if not season_metadata:
            continue
        _update_item_fields(
            season_item,
            tv_title,
            season_metadata.get("image") or tv_image,
        )
        _refresh_episode_items(season_item, season_metadata, tv_title)

    item.fetch_releases(delay=True)
    logger.info("Refreshed TV metadata for %s", item)
    return True


def _refresh_episode_items(season_item, season_metadata, tv_title):
    """Update title/image on tracked Episode Items under a season."""
    episodes_metadata = tmdb.process_episodes(season_metadata, [])
    existing = {
        ep.episode_number: ep
        for ep in Item.objects.filter(
            source=Sources.TMDB.value,
            media_type=MediaTypes.EPISODE.value,
            media_id=season_item.media_id,
            season_number=season_item.season_number,
        )
    }

    to_update = []
    for ep_data in episodes_metadata:
        ep_item = existing.get(ep_data["episode_number"])
        if ep_item is None:
            continue
        ep_item.title = tv_title or ep_item.title
        ep_item.image = ep_data["image"] or ep_item.image
        to_update.append(ep_item)

    if to_update:
        Item.objects.bulk_update(to_update, ["title", "image"], batch_size=100)
