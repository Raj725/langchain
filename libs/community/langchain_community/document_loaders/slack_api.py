import os
from typing import Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.slack import SlackAPIWrapper, logger


class SlackAPILoader(BaseLoader):
    """Load from a `Slack` API."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        workspace_url: Optional[str] = None,
        channel_name: Optional[str] = None,
        load_auth: Optional[bool] = False,
        message_limit: Optional[int] = 1000,
        channel_types: Optional[str] = "public_channel,private_channel",
    ):
        """
        Initialize the SlackAPILoader.

        Args:
            token (Optional[str]): The Slack API token.
            workspace_url (Optional[str]): The Slack workspace URL.
                Including the URL will turn sources into links. Defaults to None.
            channel_name (Optional[str]): The channel to load data from.
                When not provided, loads data from all channels. Defaults to None.
            load_auth (Optional[bool]): Whether to load authorized identities.
                Defaults to False.
            channel_types (Optional[str]): Comma-separated list of types of channels to
                load data from. Defaults to "public_channel,private_channel".
            message_limit (Optional[int]): The maximum number of messages to load.
                Defaults to 1000.
        """
        self.token = token
        # Get workspace url from environment variable if not provided
        self.workspace_url = workspace_url or os.getenv("SLACK_WORKSPACE_URL", "")
        self.channel_name = channel_name
        self.load_auth = load_auth
        self.message_limit = message_limit
        self.channel_types = channel_types
        self.client = SlackAPIWrapper(token=token, workspace_url=workspace_url)
        # Get channel details map(channel name to channel details)
        self.channel_details_map = self.client.get_channel_details_map(channel_types)
        # Get user details map(user ID to user details)
        self.user_details_map = self.client.get_user_details_map()

    def lazy_load(self) -> Iterator[Document]:
        """
        Load and return documents from the Slack API.
        If a channel_name is specified, only messages from that channel are loaded.

        Yields:
            Document: A document object representing the parsed blob.
        """
        # Get the list of channels to load data from
        channels = (
            [self.channel_name]
            if self.channel_name
            else self.channel_details_map.keys()
        )
        # Get messages from each channel
        for channel_name in channels:
            channel_id = self.channel_details_map.get(channel_name, {}).get("id")
            if not channel_id:
                logger.warning(f"Channel ID not found for channel: {channel_name}")
                continue
            messages = self.client.get_messages(
                channel=channel_id, limit=self.message_limit
            )
            for message in messages:
                yield self._convert_message_to_document(message, channel_name)

    def _convert_message_to_document(
        self, message: dict, channel_name: str
    ) -> Document:
        """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
        text = message.get("text", "")
        metadata = self._get_message_metadata(message, channel_name)
        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _get_message_metadata(self, message: dict, channel_name: str) -> dict:
        """Create and return metadata for a given message and channel."""
        timestamp = message.get("ts", "")
        user = message.get("user", "")
        source = self._get_message_source(channel_name, user, timestamp)
        return {
            "source": source,
            "channel": channel_name,
            "timestamp": timestamp,
            "user": user,
        }

    def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
        """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
        if self.workspace_url:
            channel_id = self.channel_details_map.get(channel_name, {}).get("id")
            return (
                f"{self.workspace_url}/archives/{channel_id}"
                + f"/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"
