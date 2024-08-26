import json
import logging
import os
import platform
import threading
import time
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStoreRetriever
from requests import Response, request
from requests.exceptions import RequestException

from langchain_community.chains.pebblo_retrieval.models import (
    App,
    AuthContext,
    Context,
    Framework,
    Prompt,
    Qa,
    Runtime,
    SemanticContext,
)

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"

_DEFAULT_CLASSIFIER_URL = "http://localhost:8000"
_DEFAULT_PEBBLO_CLOUD_URL = "https://api.daxa.ai"
_POLICY_REFRESH_INTERVAL_SEC = 30


class Routes(str, Enum):
    """Routes available for the Pebblo API as enumerator."""

    retrieval_app_discover = "/v1/app/discover"
    prompt = "/v1/prompt"
    prompt_governance = "/v1/prompt/governance"
    app_policy = "/v1/app/policy"


class PolicySource(str, Enum):
    """Policy source enumerator."""

    FILE = "file"
    CLOUD = "cloud"


def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(
        name="langchain", version=runtime_env.get("library_version", None)
    )
    uname = platform.uname()
    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=runtime_env.get("platform", "unknown"),
        os=uname.system,
        os_version=uname.version,
        ip=get_ip(),
        language=runtime_env.get("runtime", "unknown"),
        language_version=runtime_env.get("runtime_version", "unknown"),
    )

    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        runtime.runtime = "Mac OSX"

    logger.debug(f"framework {framework}")
    logger.debug(f"runtime {runtime}")
    return framework, runtime


def get_ip() -> str:
    """Fetch local runtime ip address.

    Returns:
        str: IP address
    """
    import socket  # lazy imports

    host = socket.gethostname()
    try:
        public_ip = socket.gethostbyname(host)
    except Exception:
        public_ip = socket.gethostbyname("localhost")
    return public_ip


class PebbloRetrievalAPIWrapper(BaseModel):
    """Wrapper for Pebblo Retrieval API."""

    api_key: Optional[str]  # Use SecretStr
    """API key for Pebblo Cloud"""
    classifier_location: str = "local"
    """Location of the classifier, local or cloud. Defaults to 'local'"""
    classifier_url: Optional[str]
    """URL of the Pebblo Classifier"""
    cloud_url: Optional[str]
    """URL of the Pebblo Cloud"""
    app_name: str
    """Name of the app"""
    policy_source: PolicySource = PolicySource.FILE
    """Source of the policy, file or cloud"""

    def __init__(self, **kwargs: Any):
        """Validate that api key in environment."""
        kwargs["api_key"] = get_from_dict_or_env(
            kwargs, "api_key", "PEBBLO_API_KEY", ""
        )
        kwargs["classifier_url"] = get_from_dict_or_env(
            kwargs, "classifier_url", "PEBBLO_CLASSIFIER_URL", _DEFAULT_CLASSIFIER_URL
        )
        kwargs["cloud_url"] = get_from_dict_or_env(
            kwargs, "cloud_url", "PEBBLO_CLOUD_URL", _DEFAULT_PEBBLO_CLOUD_URL
        )
        super().__init__(**kwargs)
        self._start_policy_refresh_thread()

    def send_app_discover(self, app: App) -> None:
        """
        Send app discovery request to Pebblo server & cloud.

        Args:
            app (App): App instance to be discovered.
        """
        pebblo_resp = None
        payload = app.dict(exclude_unset=True)

        if self.classifier_location == "local":
            # Send app details to local classifier
            headers = self._make_headers()
            app_discover_url = f"{self.classifier_url}{Routes.retrieval_app_discover}"
            pebblo_resp = self.make_request("POST", app_discover_url, headers, payload)

        if self.api_key:
            # Send app details to Pebblo cloud if api_key is present
            headers = self._make_headers(cloud_request=True)
            if pebblo_resp:
                pebblo_server_version = json.loads(pebblo_resp.text).get(
                    "pebblo_server_version"
                )
                payload.update({"pebblo_server_version": pebblo_server_version})

            payload.update({"pebblo_client_version": PLUGIN_VERSION})
            pebblo_cloud_url = f"{self.cloud_url}{Routes.retrieval_app_discover}"
            _ = self.make_request("POST", pebblo_cloud_url, headers, payload)

    def send_prompt(
        self,
        app_name: str,
        retriever: VectorStoreRetriever,
        question: str,
        answer: str,
        auth_context: Optional[AuthContext],
        docs: List[Document],
        prompt_entities: Dict[str, Any],
        prompt_time: str,
        prompt_gov_enabled: bool = False,
    ) -> None:
        """
        Send prompt to Pebblo server for classification.
        Then send prompt to Daxa cloud(If api_key is present).

        Args:
            app_name (str): Name of the app.
            retriever (VectorStoreRetriever): Retriever instance.
            question (str): Question asked in the prompt.
            answer (str): Answer generated by the model.
            auth_context (Optional[AuthContext]): Authentication context.
            docs (List[Document]): List of documents retrieved.
            prompt_entities (Dict[str, Any]): Entities present in the prompt.
            prompt_time (str): Time when the prompt was generated.
            prompt_gov_enabled (bool): Whether prompt governance is enabled.
        """
        pebblo_resp = None
        payload = self.build_prompt_qa_payload(
            app_name,
            retriever,
            question,
            answer,
            auth_context,
            docs,
            prompt_entities,
            prompt_time,
            prompt_gov_enabled,
        )

        if self.classifier_location == "local":
            # Send prompt to local classifier
            headers = self._make_headers()
            prompt_url = f"{self.classifier_url}{Routes.prompt}"
            pebblo_resp = self.make_request("POST", prompt_url, headers, payload)

        if self.api_key:
            # Send prompt to Pebblo cloud if api_key is present
            if self.classifier_location == "local":
                # If classifier location is local, then response, context and prompt
                # should be fetched from pebblo_resp and replaced in payload.
                pebblo_resp = pebblo_resp.json() if pebblo_resp else None
                self.update_cloud_payload(payload, pebblo_resp)

            headers = self._make_headers(cloud_request=True)
            pebblo_cloud_prompt_url = f"{self.cloud_url}{Routes.prompt}"
            _ = self.make_request("POST", pebblo_cloud_prompt_url, headers, payload)
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending prompt to Pebblo cloud.")
            raise NameError("API key is missing for sending prompt to Pebblo cloud.")

    def check_prompt_validity(self, question: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the validity of the given prompt using a remote classification service.

        This method sends a prompt to a remote classifier service and return entities
        present in prompt or not.

        Args:
            question (str): The prompt question to be validated.

        Returns:
            bool: True if the prompt is valid (does not contain deny list entities),
            False otherwise.
            dict: The entities present in the prompt
        """
        prompt_payload = {"prompt": question}
        prompt_entities: dict = {"entities": {}, "entityCount": 0}
        is_valid_prompt: bool = True
        if self.classifier_location == "local":
            headers = self._make_headers()
            prompt_gov_api_url = f"{self.classifier_url}{Routes.prompt_governance}"
            pebblo_resp = self.make_request(
                "POST", prompt_gov_api_url, headers, prompt_payload
            )
            if pebblo_resp:
                logger.debug(f"pebblo_resp.json() {pebblo_resp.json()}")
                prompt_entities["entities"] = pebblo_resp.json().get("entities", {})
                prompt_entities["entityCount"] = pebblo_resp.json().get(
                    "entityCount", 0
                )
        return is_valid_prompt, prompt_entities

    def get_semantic_context(self) -> Optional[SemanticContext]:
        """Get the semantic context from the local cache."""
        return self.policy_cache[1] if self.policy_cache else None

    def _start_policy_refresh_thread(self) -> None:
        """Start a thread to fetch policy from the Pebblo cloud."""
        policy_thread = threading.Thread(target=self._fetch_policy, daemon=True)
        policy_thread.start()

    def _fetch_policy(self) -> None:
        """Fetch policy from the Pebblo cloud or a file at regular intervals."""
        while True:
            try:
                policy = None
                logger.info(f"Fetching policy from source: {self.policy_source}")
                if self.policy_source == "file":
                    # Read policy from a file
                    policy = self.get_policy_from_file(self.app_name)
                else:
                    # Fetch policy from the Pebblo cloud
                    policy = self.get_policy_from_api(self.app_name)
                # Update the local cache with the fetched policy
                self._update_local_policy_cache(policy)
            except Exception as e:
                logger.warning(f"Failed to fetch policy: {e}")
            # Sleep for the refresh interval
            time.sleep(_POLICY_REFRESH_INTERVAL_SEC)

    def get_policy_from_api(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the policy for an app from the Pebblo Cloud.

        Args:
            app_name (str): Name of the app.

        Returns:
            Optional[Dict[str, Any]]: Policy for the app.
        """
        policy = None
        policy_url = f"{self.cloud_url}{Routes.app_policy}"
        logger.warning(f"Getting policy from Pebblo Cloud: {policy_url}")
        headers = self._make_headers(cloud_request=True)
        payload = {"app_name": app_name}
        response = self.make_request("POST", policy_url, headers, payload)
        if response and response.status_code == HTTPStatus.OK:
            policy = response.json()
        return policy

    @staticmethod
    def get_policy_from_file(app_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the policy for an app from the policy.json file.

        Args:
            app_name (str): Name of the app.

        Returns:
            Optional[Dict[str, Any]]: Policy for the app.
        """

        # read the policy file from current directory
        policy_file = "policy.json"
        logger.warning(f"Reading policy file: {policy_file}")
        if os.path.exists(policy_file):
            with open(policy_file, "r") as f:
                policy = json.load(f)
                return policy.get(app_name)
        return None

    def _update_local_policy_cache(self, policy: dict) -> None:
        """Update the local cache with the fetched policy."""
        # Generate semantic context from the first policy
        semantic_context = self._generate_semantic_context(policy)
        self.policy_cache = (policy, semantic_context)

    def _generate_semantic_context(self, policy: Dict[str, Any]) -> SemanticContext:
        semantic_context = dict()
        if policy.get("entity"):
            entities_to_deny = self.get_entities_from_groups(
                policy["entity"]["deny_groups"]
            )
            entities_to_deny.extend(policy["entity"]["deny"])
            semantic_context["pebblo_semantic_entities"] = {"deny": entities_to_deny}
        if policy.get("semantics"):
            topics_to_deny = self.get_topics_from_groups(
                policy["semantics"]["deny_groups"]
            )
            topics_to_deny.extend(policy["semantics"]["deny"])
            semantic_context["pebblo_semantic_topics"] = {"deny": topics_to_deny}
        return SemanticContext(**semantic_context)

    @staticmethod
    def get_entities_from_groups(entity_groups: List[str]) -> List[str]:
        """
        Get the entities from entity groups from the Pebblo Cloud.

        Args:
            entity_groups (List[str]): List of entity groups.

        Returns:
            List[str]: List of entities.
        """
        entities = []
        for entity_group in entity_groups:
            # temporary return the entity group as entity
            entities.append(entity_group)
        return entities

    @staticmethod
    def get_topics_from_groups(topic_groups: List[str]) -> List[str]:
        """
        Get the topics from topic groups from the Pebblo Cloud.

        Args:
            topic_groups (List[str]): List of topic groups.

        Returns:
            List[str]: List of topics.
        """
        topics = []
        for topic_group in topic_groups:
            # temporary return the topic group as topic
            topics.append(topic_group)
        return topics

    def _make_headers(self, cloud_request: bool = False) -> dict:
        """
        Generate headers for the request.

        args:
            cloud_request (bool): flag indicating whether the request is for Pebblo
            cloud.
        returns:
            dict: Headers for the request.

        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if cloud_request:
            # Add API key for Pebblo cloud request
            if self.api_key:
                headers.update({"x-api-key": self.api_key})
            else:
                logger.warning("API key is missing for Pebblo cloud request.")
        return headers

    @staticmethod
    def make_request(
        method: str,
        url: str,
        headers: dict,
        payload: Optional[dict] = None,
        timeout: int = 20,
    ) -> Optional[Response]:
        """
        Make a request to the Pebblo server/cloud API.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.).
            url (str): URL for the request.
            headers (dict): Headers for the request.
            payload (Optional[dict]): Payload for the request (for POST, PUT, etc.).
            timeout (int): Timeout for the request in seconds.

        Returns:
            Optional[Response]: Response object if the request is successful.
        """
        try:
            response = request(
                method=method, url=url, headers=headers, json=payload, timeout=timeout
            )
            logger.debug(
                "Request: method %s, url %s, len %s response status %s",
                method,
                response.request.url,
                str(len(response.request.body if response.request.body else [])),
                str(response.status_code),
            )

            if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
                logger.warning(f"Pebblo Server: Error {response.status_code}")
            elif response.status_code >= HTTPStatus.BAD_REQUEST:
                logger.warning(f"Pebblo received an invalid payload: {response.text}")
            elif response.status_code != HTTPStatus.OK:
                logger.warning(
                    f"Pebblo returned an unexpected response code: "
                    f"{response.status_code}"
                )

            return response
        except RequestException:
            logger.warning("Unable to reach server %s", url)
        except Exception as e:
            logger.warning("An Exception caught in make_request: %s", e)
        return None

    @staticmethod
    def update_cloud_payload(payload: dict, pebblo_resp: Optional[dict]) -> None:
        """
        Update the payload with response, prompt and context from Pebblo response.

        Args:
            payload (dict): Payload to be updated.
            pebblo_resp (Optional[dict]): Response from Pebblo server.
        """
        if pebblo_resp:
            # Update response, prompt and context from pebblo response
            response = payload.get("response", {})
            response.update(pebblo_resp.get("retrieval_data", {}).get("response", {}))
            response.pop("data", None)
            prompt = payload.get("prompt", {})
            prompt.update(pebblo_resp.get("retrieval_data", {}).get("prompt", {}))
            prompt.pop("data", None)
            context = payload.get("context", [])
            for context_data in context:
                context_data.pop("doc", None)
        else:
            payload["response"] = {}
            payload["prompt"] = {}
            payload["context"] = []

    def build_prompt_qa_payload(
        self,
        app_name: str,
        retriever: VectorStoreRetriever,
        question: str,
        answer: str,
        auth_context: Optional[AuthContext],
        docs: List[Document],
        prompt_entities: Dict[str, Any],
        prompt_time: str,
        prompt_gov_enabled: bool = False,
    ) -> dict:
        """
        Build the QA payload for the prompt.

         Args:
            app_name (str): Name of the app.
            retriever (VectorStoreRetriever): Retriever instance.
            question (str): Question asked in the prompt.
            answer (str): Answer generated by the model.
            auth_context (Optional[AuthContext]): Authentication context.
            docs (List[Document]): List of documents retrieved.
            prompt_entities (Dict[str, Any]): Entities present in the prompt.
            prompt_time (str): Time when the prompt was generated.
            prompt_gov_enabled (bool): Whether prompt governance is enabled.

        Returns:
            dict: The QA payload for the prompt.
        """
        qa = Qa(
            name=app_name,
            context=[
                Context(
                    retrieved_from=doc.metadata.get(
                        "full_path", doc.metadata.get("source")
                    ),
                    doc=doc.page_content,
                    vector_db=retriever.vectorstore.__class__.__name__,
                    pb_checksum=doc.metadata.get("pb_checksum"),
                )
                for doc in docs
                if isinstance(doc, Document)
            ],
            prompt=Prompt(
                data=question,
                entities=prompt_entities.get("entities", {}),
                entityCount=prompt_entities.get("entityCount", 0),
                prompt_gov_enabled=prompt_gov_enabled,
            ),
            response=Prompt(data=answer),
            prompt_time=prompt_time,
            user=auth_context.user_id if auth_context else "unknown",
            user_identities=auth_context.user_auth
            if auth_context and hasattr(auth_context, "user_auth")
            else [],
            classifier_location=self.classifier_location,
        )
        return qa.dict(exclude_unset=True)
