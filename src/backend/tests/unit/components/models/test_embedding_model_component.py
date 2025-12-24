from unittest.mock import MagicMock, patch

import pytest
from lfx.base.models.ollama_constants import OLLAMA_EMBEDDING_MODELS
from lfx.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES
from lfx.base.models.watsonx_constants import WATSONX_EMBEDDING_MODEL_NAMES
from lfx.components.models.embedding_model import (
    EmbeddingModelComponent,
)

from tests.base import ComponentTestBaseWithClient


@pytest.mark.usefixtures("client")
class TestEmbeddingModelComponent(ComponentTestBaseWithClient):
    @pytest.fixture
    def component_class(self):
        return EmbeddingModelComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "provider": "QueryRouter",
            "model": "baai/bge-m3",
            "api_key": "test-api-key",
            "chunk_size": 1000,
            "max_retries": 3,
            "show_progress_bar": False,
        }

    @pytest.fixture
    def file_names_mapping(self):
        """Return the file names mapping for version-specific files."""

    @patch.object(EmbeddingModelComponent, "_fetch_queryrouter_embedding_models")
    async def test_update_build_config_queryrouter(self, mock_fetch_models, component_class, default_kwargs):
        mock_models = ["baai/bge-m3", "baai/bge-large-en-v1.5"]
        mock_fetch_models.return_value = mock_models
        component = component_class(**default_kwargs)
        build_config = {
            "model": {"options": [], "value": ""},
            "api_key": {"display_name": "API Key", "required": True, "show": True, "value": ""},
            "api_base": {"display_name": "API Base URL", "value": ""},
            "project_id": {"show": False},
        }
        updated_config = component.update_build_config(build_config, "QueryRouter", "provider")
        assert updated_config["model"]["options"] == mock_models
        assert updated_config["model"]["value"] == mock_models[0]
        assert updated_config["api_key"]["display_name"] == "QueryRouter API Key"
        assert updated_config["api_key"]["value"] == "QUERYROUTER_API_KEY"
        assert updated_config["api_key"]["required"] is True
        assert updated_config["api_key"]["show"] is True
        assert updated_config["api_base"]["display_name"] == "QueryRouter API Base URL"
        assert updated_config["project_id"]["show"] is False

    async def test_update_build_config_openai(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        build_config = {
            "model": {"options": [], "value": ""},
            "api_key": {"display_name": "API Key", "required": True, "show": True, "value": ""},
            "api_base": {"display_name": "API Base URL", "value": ""},
            "project_id": {"show": False},
        }
        updated_config = component.update_build_config(build_config, "OpenAI", "provider")
        assert updated_config["model"]["options"] == OPENAI_EMBEDDING_MODEL_NAMES
        assert updated_config["model"]["value"] == OPENAI_EMBEDDING_MODEL_NAMES[0]
        assert updated_config["api_key"]["display_name"] == "OpenAI API Key"
        assert updated_config["api_key"]["required"] is True
        assert updated_config["api_key"]["show"] is True
        assert updated_config["api_base"]["display_name"] == "OpenAI API Base URL"
        assert updated_config["project_id"]["show"] is False

    async def test_update_build_config_ollama(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        build_config = {
            "model": {"options": [], "value": ""},
            "api_key": {"display_name": "API Key", "required": True, "show": True, "value": ""},
            "api_base": {"display_name": "API Base URL", "value": ""},
            "project_id": {"show": False},
        }
        updated_config = component.update_build_config(build_config, "Ollama", "provider")
        assert updated_config["model"]["options"] == OLLAMA_EMBEDDING_MODELS
        assert updated_config["model"]["value"] == OLLAMA_EMBEDDING_MODELS[0]
        assert updated_config["api_key"]["display_name"] == "API Key (Optional)"
        assert updated_config["api_key"]["required"] is False
        assert updated_config["api_key"]["show"] is False
        assert updated_config["api_base"]["display_name"] == "Ollama Base URL"
        assert updated_config["api_base"]["value"] == "http://localhost:11434"
        assert updated_config["project_id"]["show"] is False

    async def test_update_build_config_watsonx(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        build_config = {
            "model": {"options": [], "value": ""},
            "api_key": {"display_name": "API Key", "required": True, "show": True, "value": ""},
            "api_base": {"display_name": "API Base URL", "value": ""},
            "project_id": {"show": False},
        }
        updated_config = component.update_build_config(build_config, "WatsonX", "provider")
        assert updated_config["model"]["options"] == WATSONX_EMBEDDING_MODEL_NAMES
        assert updated_config["model"]["value"] == WATSONX_EMBEDDING_MODEL_NAMES[0]
        assert updated_config["api_key"]["display_name"] == "Watson AI API Key"
        assert updated_config["api_key"]["required"] is True
        assert updated_config["api_key"]["show"] is True
        assert updated_config["api_base"]["display_name"] == "Watson AI URL"
        assert updated_config["api_base"]["value"] == "https://us-south.ml.cloud.ibm.com"
        assert updated_config["project_id"]["show"] is True

    @patch("lfx.components.models.embedding_model.OpenAIEmbeddings")
    async def test_build_embeddings_queryrouter(self, mock_openai_embeddings, component_class, default_kwargs):
        mock_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_instance

        component = component_class(**default_kwargs)
        component.provider = "QueryRouter"
        component.model = "baai/bge-m3"
        component.api_key = "test-key"
        component.chunk_size = 1000
        component.max_retries = 3
        component.show_progress_bar = False

        embeddings = component.build_embeddings()

        mock_openai_embeddings.assert_called_once_with(
            model="baai/bge-m3",
            dimensions=None,
            base_url="https://api.queryrouter.ru/v1",
            api_key="test-key",
            chunk_size=1000,
            max_retries=3,
            timeout=None,
            show_progress_bar=False,
            model_kwargs={},
        )
        assert embeddings == mock_instance

    @patch("lfx.components.models.embedding_model.OpenAIEmbeddings")
    async def test_build_embeddings_openai(self, mock_openai_embeddings, component_class, default_kwargs):
        mock_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_instance

        kwargs = default_kwargs.copy()
        kwargs["provider"] = "OpenAI"
        kwargs["model"] = "text-embedding-3-small"
        component = component_class(**kwargs)
        component.api_key = "test-key"
        component.chunk_size = 1000
        component.max_retries = 3
        component.show_progress_bar = False

        embeddings = component.build_embeddings()

        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            dimensions=None,
            base_url=None,
            api_key="test-key",
            chunk_size=1000,
            max_retries=3,
            timeout=None,
            show_progress_bar=False,
            model_kwargs={},
        )
        assert embeddings == mock_instance

    @patch("langchain_ollama.OllamaEmbeddings")
    async def test_build_embeddings_ollama(self, mock_ollama_embeddings, component_class, default_kwargs):
        mock_instance = MagicMock()
        mock_ollama_embeddings.return_value = mock_instance

        kwargs = default_kwargs.copy()
        kwargs["provider"] = "Ollama"
        kwargs["model"] = "nomic-embed-text"
        component = component_class(**kwargs)
        component.api_base = "http://localhost:11434"

        embeddings = component.build_embeddings()

        mock_ollama_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )
        assert embeddings == mock_instance

    @patch("langchain_ibm.WatsonxEmbeddings")
    async def test_build_embeddings_watsonx(self, mock_watsonx_embeddings, component_class, default_kwargs):
        mock_instance = MagicMock()
        mock_watsonx_embeddings.return_value = mock_instance

        kwargs = default_kwargs.copy()
        kwargs["provider"] = "WatsonX"
        kwargs["model"] = "ibm/granite-embedding-125m-english"
        component = component_class(**kwargs)
        component.project_id = "test-project-id"

        embeddings = component.build_embeddings()

        mock_watsonx_embeddings.assert_called_once_with(
            model_id="ibm/granite-embedding-125m-english",
            url="https://us-south.ml.cloud.ibm.com",
            apikey="test-api-key",
            project_id="test-project-id",
        )
        assert embeddings == mock_instance

    async def test_build_embeddings_queryrouter_missing_api_key(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        component.provider = "QueryRouter"
        component.api_key = None

        with pytest.raises(ValueError, match="QueryRouter API key is required when using QueryRouter provider"):
            component.build_embeddings()

    async def test_build_embeddings_queryrouter_missing_model(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        component.provider = "QueryRouter"
        component.model = ""

        with pytest.raises(ValueError, match="Please select a model"):
            component.build_embeddings()

    async def test_build_embeddings_watsonx_missing_project_id(self, component_class, default_kwargs):
        kwargs = default_kwargs.copy()
        kwargs["provider"] = "WatsonX"
        component = component_class(**kwargs)
        component.project_id = None

        with pytest.raises(ValueError, match="Project ID is required for WatsonX"):
            component.build_embeddings()

    async def test_build_embeddings_openai_missing_api_key(self, component_class, default_kwargs):
        kwargs = default_kwargs.copy()
        kwargs["provider"] = "OpenAI"
        component = component_class(**kwargs)
        component.api_key = None

        with pytest.raises(ValueError, match="OpenAI API key is required when using OpenAI provider"):
            component.build_embeddings()

    async def test_build_embeddings_watsonx_missing_api_key(self, component_class, default_kwargs):
        kwargs = default_kwargs.copy()
        kwargs["provider"] = "WatsonX"
        kwargs["api_key"] = None
        component = component_class(**kwargs)
        component.api_key = None
        component.project_id = "test-project"

        with pytest.raises(ValueError, match="Watson AI API key is required when using WatsonX provider"):
            component.build_embeddings()

    async def test_build_embeddings_unknown_provider(self, component_class, default_kwargs):
        component = component_class(**default_kwargs)
        component.provider = "Unknown"

        with pytest.raises(ValueError, match="Unknown provider: Unknown"):
            component.build_embeddings()

    @patch("httpx.get")
    async def test_fetch_queryrouter_embedding_models_success(self, mock_get, component_class, default_kwargs):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1", "slug": "vendor/model-1"},
                {"id": "model-2", "slug": "vendor/model-2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        component = component_class(**default_kwargs)
        models = component._fetch_queryrouter_embedding_models()

        assert models == ["vendor/model-1", "vendor/model-2"]

    @patch("httpx.get")
    async def test_fetch_queryrouter_embedding_models_fallback_on_error(
        self, mock_get, component_class, default_kwargs
    ):
        import httpx

        mock_get.side_effect = httpx.RequestError("Connection error")

        component = component_class(**default_kwargs)
        models = component._fetch_queryrouter_embedding_models()

        assert models == []

    @patch.object(EmbeddingModelComponent, "_fetch_queryrouter_embedding_models")
    async def test_update_build_config_queryrouter_empty_models(
        self, mock_fetch_models, component_class, default_kwargs
    ):
        mock_fetch_models.return_value = []
        component = component_class(**default_kwargs)
        build_config = {
            "model": {"options": [], "value": ""},
            "api_key": {"display_name": "API Key", "required": True, "show": True, "value": ""},
            "api_base": {"display_name": "API Base URL", "value": ""},
            "project_id": {"show": False},
        }
        updated_config = component.update_build_config(build_config, "QueryRouter", "provider")
        assert updated_config["model"]["options"] == []
        assert updated_config["model"]["value"] == ""
