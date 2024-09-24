from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SlackAPIWrapper(BaseModel):
    """Wrapper for Slack API."""

    token: Optional[str] = None  # Use SecretStr
    """Token for Slack API"""

    workspace_url: Optional[str] = None
    """URL of the Slack workspace"""

    slack_client: Any = None
    """Client for Slack API"""

    def __init__(self, **kwargs: Any):
        """Validate environment."""
        kwargs["token"] = get_from_dict_or_env(kwargs, "token", "SLACK_TOKEN", "")
        kwargs["workspace_url"] = get_from_dict_or_env(
            kwargs, "workspace_url", "SLACK_WORKSPACE_URL", ""
        )

        try:
            from slack_sdk import WebClient
        except ImportError:
            raise ImportError(
                "The 'slack_sdk' package is not installed. "
                "Please install it using 'pip install slack_sdk'."
            )
        client = WebClient(token=kwargs["api_key"])
        kwargs["slack_client"] = client

        super().__init__(**kwargs)

    def get_channel_details_map(
        self, types: str = "public_channel"
    ) -> Dict[str, Dict[str, str]]:
        """
        Get a dictionary mapping channel names to their respective details.
        Details include the channel ID, name, and type(public, private).

        Args:
            types (str): Comma-separated list of types of channels to get details from.
                Defaults to "public_channel".

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping channel names to their
            respective details.
        """
        try:
            channels = self.slack_client.conversations_list(types=types)
            return {
                channel["name"]: {
                    "id": channel.get("id"),
                    "name": channel.get("name"),
                    "type": channel.get("type"),
                }
                for channel in channels
            }
        except Exception as e:
            logger.error(f"Error getting channel details map from Slack: {e}")

    def get_user_details_map(self) -> Dict[str, Dict[str, str]]:
        """Get a dictionary mapping user IDs to their respective details."""
        try:
            users = self.slack_client.users_list()
            return {
                user["id"]: {
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "email": user.get("profile").get("email"),
                }
                for user in users
            }
        except Exception as e:
            logger.error(f"Error getting user details map from Slack: {e}")
            return {}

    def get_messages(self, channel: str, limit: int = 1000) -> Any:
        """Get messages from a channel."""
        try:
            messages = self.slack_client.conversations_history(
                channel=channel, limit=limit
            )
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from Slack: {e}")
            return None
