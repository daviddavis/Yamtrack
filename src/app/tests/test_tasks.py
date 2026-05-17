from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from app.models import (
    TV,
    Item,
    MediaTypes,
    Movie,
    Season,
    Sources,
    Status,
    UserMessage,
    UserMessageLevel,
)
from app.providers import services
from app.tasks import cleanup_user_messages, refresh_tmdb_metadata


def _make_provider_error():
    """Build a ProviderAPIError with the minimum mocked response shape."""
    error = MagicMock()
    error.response.status_code = 500
    error.response.text = "boom"
    return services.ProviderAPIError(
        provider=Sources.TMDB.value,
        error=error,
        details="boom",
    )


class CleanupUserMessagesTaskTests(TestCase):
    """Test cleanup of old shown user messages."""

    def setUp(self):
        """Create a user for task tests."""
        self.user = get_user_model().objects.create_user(
            username="test",
        )

    @override_settings(USER_MESSAGE_RETENTION_DAYS=30)
    def test_cleanup_user_messages_deletes_only_old_shown_messages(self):
        """Delete only shown messages older than the retention window."""
        now = timezone.now()
        old_shown = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="old shown",
            shown_at=now - timedelta(days=31),
        )
        recent_shown = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="recent shown",
            shown_at=now - timedelta(days=5),
        )
        unseen = UserMessage.objects.create(
            user=self.user,
            level=UserMessageLevel.INFO,
            message="unseen",
        )

        deleted_count = cleanup_user_messages()

        self.assertEqual(deleted_count, 1)
        self.assertFalse(UserMessage.objects.filter(id=old_shown.id).exists())
        self.assertTrue(UserMessage.objects.filter(id=recent_shown.id).exists())
        self.assertTrue(UserMessage.objects.filter(id=unseen.id).exists())


class RefreshTmdbMetadataTaskTests(TestCase):
    """Test the weekly TMDB metadata refresh task."""

    def setUp(self):
        """Create tracked TMDB items used across tests."""
        self.user = get_user_model().objects.create_user(username="refresh_user")

        self.tv_item = Item.objects.create(
            media_id="1396",
            source=Sources.TMDB.value,
            media_type=MediaTypes.TV.value,
            title="Old TV Title",
            image="http://example.com/old_tv.jpg",
        )
        tv_object = TV.objects.create(
            item=self.tv_item,
            user=self.user,
            status=Status.PLANNING.value,
        )
        self.season_item = Item.objects.create(
            media_id="1396",
            source=Sources.TMDB.value,
            media_type=MediaTypes.SEASON.value,
            season_number=1,
            title="Old TV Title",
            image="http://example.com/old_tv.jpg",
        )
        Season.objects.create(
            item=self.season_item,
            related_tv=tv_object,
            user=self.user,
            status=Status.PLANNING.value,
        )
        self.episode_item = Item.objects.create(
            media_id="1396",
            source=Sources.TMDB.value,
            media_type=MediaTypes.EPISODE.value,
            season_number=1,
            episode_number=1,
            title="Old TV Title",
            image="http://example.com/old_episode.jpg",
        )
        self.movie_item = Item.objects.create(
            media_id="238",
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            title="Old Movie Title",
            image="http://example.com/old_movie.jpg",
        )
        Movie.objects.create(
            item=self.movie_item,
            user=self.user,
            status=Status.PLANNING.value,
        )
        # An item that should NOT be refreshed because TMDB does not report it.
        self.untouched_item = Item.objects.create(
            media_id="999999",
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            title="Untouched",
            image="http://example.com/untouched.jpg",
        )
        Movie.objects.create(
            item=self.untouched_item,
            user=self.user,
            status=Status.PLANNING.value,
        )

    @patch("app.tasks.Item.fetch_releases")
    @patch("app.tasks.tmdb.process_episodes")
    @patch("app.tasks.tmdb.tv_with_seasons")
    @patch("app.tasks.tmdb.movie")
    @patch("app.tasks.tmdb.movie_changes")
    @patch("app.tasks.tmdb.tv_changes")
    def test_refresh_tmdb_metadata_updates_changed_items(
        self,
        mock_tv_changes,
        mock_movie_changes,
        mock_movie,
        mock_tv_with_seasons,
        mock_process_episodes,
        mock_fetch_releases,
    ):
        """Only items in the TMDB changes feed get refreshed."""
        mock_tv_changes.return_value = {"1396"}
        mock_movie_changes.return_value = {"238"}
        mock_tv_with_seasons.return_value = {
            "title": "New TV Title",
            "image": "http://example.com/new_tv.jpg",
            "season/1": {
                "media_id": "1396",
                "season_number": 1,
                "image": "http://example.com/new_season.jpg",
                "episodes": [],
            },
        }
        mock_process_episodes.return_value = [
            {"episode_number": 1, "image": "http://example.com/new_episode.jpg"},
        ]
        mock_movie.return_value = {
            "title": "New Movie Title",
            "image": "http://example.com/new_movie.jpg",
        }

        refreshed = refresh_tmdb_metadata()

        self.assertEqual(refreshed, 2)

        self.tv_item.refresh_from_db()
        self.assertEqual(self.tv_item.title, "New TV Title")
        self.assertEqual(self.tv_item.image, "http://example.com/new_tv.jpg")

        self.season_item.refresh_from_db()
        self.assertEqual(self.season_item.title, "New TV Title")
        self.assertEqual(self.season_item.image, "http://example.com/new_season.jpg")

        self.episode_item.refresh_from_db()
        self.assertEqual(self.episode_item.title, "New TV Title")
        self.assertEqual(self.episode_item.image, "http://example.com/new_episode.jpg")

        self.movie_item.refresh_from_db()
        self.assertEqual(self.movie_item.title, "New Movie Title")
        self.assertEqual(self.movie_item.image, "http://example.com/new_movie.jpg")

        self.untouched_item.refresh_from_db()
        self.assertEqual(self.untouched_item.title, "Untouched")

        mock_tv_with_seasons.assert_called_once_with("1396", [1])
        mock_movie.assert_called_once_with("238")
        # fetch_releases is triggered for each refreshed item.
        self.assertEqual(mock_fetch_releases.call_count, 2)

    @patch("app.tasks.tmdb.movie_changes")
    @patch("app.tasks.tmdb.tv_changes")
    def test_refresh_tmdb_metadata_handles_changes_api_error(
        self,
        mock_tv_changes,
        mock_movie_changes,
    ):
        """If the TMDB changes endpoint fails, the task returns 0 without raising."""
        mock_tv_changes.side_effect = _make_provider_error()
        mock_movie_changes.side_effect = _make_provider_error()

        self.assertEqual(refresh_tmdb_metadata(), 0)

    @patch("app.tasks.Item.fetch_releases")
    @patch("app.tasks.tmdb.movie")
    @patch("app.tasks.tmdb.movie_changes")
    @patch("app.tasks.tmdb.tv_changes")
    def test_refresh_tmdb_metadata_skips_item_on_provider_error(
        self,
        mock_tv_changes,
        mock_movie_changes,
        mock_movie,
        mock_fetch_releases,
    ):
        """A provider error on one item does not stop processing of others."""
        mock_tv_changes.return_value = set()
        mock_movie_changes.return_value = {"238"}
        mock_movie.side_effect = _make_provider_error()

        refreshed = refresh_tmdb_metadata()

        self.assertEqual(refreshed, 0)
        self.movie_item.refresh_from_db()
        self.assertEqual(self.movie_item.title, "Old Movie Title")
        mock_fetch_releases.assert_not_called()

    @patch("app.tasks.Item.fetch_releases")
    @patch("app.tasks.tmdb.movie")
    @patch("app.tasks.tmdb.tv_with_seasons")
    @patch("app.tasks.tmdb.movie_changes")
    @patch("app.tasks.tmdb.tv_changes")
    def test_refresh_tmdb_metadata_skips_orphan_items(
        self,
        mock_tv_changes,
        mock_movie_changes,
        mock_tv_with_seasons,
        mock_movie,
        mock_fetch_releases,
    ):
        """Items with no tracking rows (orphans) are not refreshed."""
        orphan_tv = Item.objects.create(
            media_id="2000",
            source=Sources.TMDB.value,
            media_type=MediaTypes.TV.value,
            title="Orphan TV",
            image="http://example.com/orphan_tv.jpg",
        )
        orphan_movie = Item.objects.create(
            media_id="3000",
            source=Sources.TMDB.value,
            media_type=MediaTypes.MOVIE.value,
            title="Orphan Movie",
            image="http://example.com/orphan_movie.jpg",
        )

        mock_tv_changes.return_value = {orphan_tv.media_id}
        mock_movie_changes.return_value = {orphan_movie.media_id}

        refreshed = refresh_tmdb_metadata()

        self.assertEqual(refreshed, 0)
        mock_tv_with_seasons.assert_not_called()
        mock_movie.assert_not_called()
        mock_fetch_releases.assert_not_called()
