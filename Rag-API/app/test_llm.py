# test_llm_factory.py

import pytest
import importlib
from unittest.mock import MagicMock, patch, Mock

from providers.llm_factory import (
    BaseLLM,
    OpenAILLM,
    GeminiLLM,
    GroqLLM,
    LLMFactory,
)
from config.core_config import config_manager, LLMProvider
import logging

# Test BaseLLM abstract class
class TestBaseLLM:
    def test_abstract_methods(self):
        """Test that BaseLLM cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseLLM(model_name="test", api_key="dummy")

    def test_concrete_subclass(self):
        """Test concrete subclass implementation"""
        class TestLLM(BaseLLM):
            def initialize(self): pass
            def generate(self, prompt): return "response"
            def chat(self, messages): return "chat response"

        llm = TestLLM(model_name="test", api_key="dummy")
        assert isinstance(llm, BaseLLM)

# Common fixtures
@pytest.fixture
def mock_config():
    with patch.object(config_manager, 'get_api_key', return_value="dummy_key"):
        yield

# Test OpenAILLM
class TestOpenAILLM:
    def test_initialize_success(self, mock_config):
        """Test successful OpenAI client initialization"""
        # Mock the OpenAI class at import time
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            llm = OpenAILLM(model_name="gpt-4", api_key="dummy")
            llm.initialize()
            
            # Verify the client was created with correct API key
            mock_openai_class.assert_called_once_with(api_key="dummy")
            assert llm.client == mock_client

    def test_generate_success(self, mock_config):
        """Test successful text generation"""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            # Mock the completions response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].text = "  test response  "
            mock_client.completions.create.return_value = mock_response
            
            llm = OpenAILLM(model_name="gpt-4", api_key="dummy")
            llm.initialize()
            response = llm.generate("test prompt")
            
            assert response == "test response"
            mock_client.completions.create.assert_called_once()

    def test_chat_success(self, mock_config):
        """Test successful chat completion"""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            # Mock the chat completions response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_client.chat.completions.create.return_value = mock_response
            
            llm = OpenAILLM(model_name="gpt-4", api_key="dummy")
            llm.initialize()
            messages = [{"role": "user", "content": "test"}]
            response = llm.chat(messages)
            
            assert response == "test response"
            mock_client.chat.completions.create.assert_called_once()

    def test_missing_package(self):
        """Test behavior when OpenAI package is missing"""
        # Mock the import to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            llm = OpenAILLM(model_name="gpt-4", api_key="dummy")
            with pytest.raises(ImportError, match="openai package is required"):
                llm.initialize()

# Test GeminiLLM
class TestGeminiLLM:
    def test_initialize_success(self, mock_config):
        """Test successful Gemini client initialization"""
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            llm = GeminiLLM(model_name="gemini-pro", api_key="dummy")
            llm.initialize()
            
            mock_configure.assert_called_once_with(api_key="dummy")
            mock_model_class.assert_called_once_with("gemini-pro")
            assert llm.client == mock_model

    def test_generate_success(self, mock_config):
        """Test successful content generation"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_response = MagicMock()
            mock_response.text = "gemini response"
            mock_model.generate_content.return_value = mock_response
            
            llm = GeminiLLM(model_name="gemini-pro", api_key="dummy")
            llm.initialize()
            response = llm.generate("test prompt")
            
            assert response == "gemini response"
            mock_model.generate_content.assert_called_once()

    def test_chat_success(self, mock_config):
        """Test successful chat session"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_response = MagicMock()
            mock_response.text = "gemini response"
            mock_chat.send_message.return_value = mock_response
            
            llm = GeminiLLM(model_name="gemini-pro", api_key="dummy")
            llm.initialize()
            messages = [{"role": "user", "content": "test"}]
            response = llm.chat(messages)
            
            assert response == "gemini response"
            mock_model.start_chat.assert_called_once()

    def test_missing_package(self):
        """Test behavior when google-generativeai package is missing"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'google.generativeai'")):
            llm = GeminiLLM(model_name="gemini-pro", api_key="dummy")
            with pytest.raises(ImportError, match="google-generativeai package is required"):
                llm.initialize()

# Test GroqLLM
class TestGroqLLM:
    def test_initialize_success(self, mock_config):
        """Test successful Groq client initialization"""
        with patch('groq.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            
            llm = GroqLLM(model_name="llama3-8b", api_key="dummy")
            llm.initialize()
            
            mock_groq_class.assert_called_once_with(api_key="dummy")
            assert llm.client == mock_client

    def test_chat_completion(self, mock_config):
        """Test successful chat completion"""
        with patch('groq.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "groq response"
            mock_client.chat.completions.create.return_value = mock_response
            
            llm = GroqLLM(model_name="llama3-8b", api_key="dummy")
            llm.initialize()
            messages = [{"role": "user", "content": "test"}]
            response = llm.chat(messages)
            
            assert response == "groq response"
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_success(self, mock_config):
        """Test successful text generation (uses chat internally)"""
        with patch('groq.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "groq response"
            mock_client.chat.completions.create.return_value = mock_response
            
            llm = GroqLLM(model_name="llama3-8b", api_key="dummy")
            llm.initialize()
            response = llm.generate("test prompt")
            
            assert response == "groq response"
            mock_client.chat.completions.create.assert_called_once()

    def test_missing_package(self):
        """Test behavior when groq package is missing"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'groq'")):
            llm = GroqLLM(model_name="llama3-8b", api_key="dummy")
            with pytest.raises(ImportError, match="groq package is required"):
                llm.initialize()

# Test LLMFactory
class TestLLMFactory:
    @pytest.mark.parametrize("provider,expected_type", [
        (LLMProvider.OPENAI, OpenAILLM),
        (LLMProvider.GEMINI, GeminiLLM),
        (LLMProvider.GROQ, GroqLLM)
    ])
    def test_create_llm(self, provider, expected_type, mock_config):
        """Test LLM creation for different providers"""
        with patch.object(expected_type, 'initialize'):
            llm = LLMFactory.create_llm(provider, "test-model")
            assert isinstance(llm, expected_type)

    def test_invalid_provider(self, mock_config):
        """Test error handling for invalid provider"""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create_llm("invalid_provider", "test-model")

    def test_provider_models(self):
        """Test default models configuration"""
        models = LLMFactory.get_default_models()
        assert "gpt-4o" in models[LLMProvider.OPENAI.value]
        assert "gemini-1.5-pro" in models[LLMProvider.GEMINI.value]
        assert "llama3-8b-8192" in models[LLMProvider.GROQ.value]

    def test_api_key_missing(self):
        """Test error handling when API key is missing"""
        with patch.object(config_manager, 'get_api_key', return_value=None):
            with pytest.raises(ValueError, match="API key not found"):
                LLMFactory.create_llm(LLMProvider.OPENAI, "test-model")

    def test_get_available_providers(self):
        """Test getting list of available providers"""
        providers = LLMFactory.get_available_providers()
        expected_providers = [LLMProvider.OPENAI.value, LLMProvider.GEMINI.value, LLMProvider.GROQ.value]
        assert all(provider in providers for provider in expected_providers)

# Integration tests with error handling
class TestLLMIntegration:
    def test_openai_error_handling(self, mock_config):
        """Test OpenAI API error handling"""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            llm = OpenAILLM(model_name="gpt-4", api_key="dummy")
            llm.initialize()
            
            with pytest.raises(Exception, match="API Error"):
                llm.chat([{"role": "user", "content": "test"}])

    def test_gemini_error_handling(self, mock_config):
        """Test Gemini API error handling"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.generate_content.side_effect = Exception("Gemini Error")
            
            llm = GeminiLLM(model_name="gemini-pro", api_key="dummy")
            llm.initialize()
            
            with pytest.raises(Exception, match="Gemini Error"):
                llm.generate("test prompt")

    def test_groq_error_handling(self, mock_config):
        """Test Groq API error handling"""
        with patch('groq.Groq') as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Groq Error")
            
            llm = GroqLLM(model_name="llama3-8b", api_key="dummy")
            llm.initialize()
            
            with pytest.raises(Exception, match="Groq Error"):
                llm.chat([{"role": "user", "content": "test"}])
