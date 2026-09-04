from unittest.mock import patch, MagicMock

import pytest
import requests


def _mock_ok_response(json_data=None):
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: json_data or {}
    return resp


def test_query_skips_bootstrap_when_server_already_has_model():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")
    with patch("utilities.ollama_client.requests.get") as mock_get, \
         patch("utilities.ollama_client.requests.post") as mock_post, \
         patch("utilities.ollama_client.subprocess.Popen") as mock_popen, \
         patch("utilities.ollama_client.subprocess.run") as mock_run:
        mock_get.return_value = _mock_ok_response({"models": [{"name": "gemma4:31b"}]})
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        result = client.query("hi")

        assert result == "hello"
        mock_popen.assert_not_called()
        mock_run.assert_not_called()


def test_query_starts_server_when_unreachable():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")

    call_count = {"n": 0}

    def fake_get(url, timeout=None):
        # First call (reachability check): fail. Subsequent calls: server is "up".
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.exceptions.ConnectionError("not up yet")
        return _mock_ok_response({"models": [{"name": "gemma4:31b"}]})

    with patch("utilities.ollama_client.requests.get", side_effect=fake_get), \
         patch("utilities.ollama_client.requests.post") as mock_post, \
         patch("utilities.ollama_client.subprocess.Popen") as mock_popen, \
         patch("utilities.ollama_client.subprocess.run") as mock_run, \
         patch("utilities.ollama_client.shutil.which", return_value="/usr/bin/ollama"), \
         patch("utilities.ollama_client.time.sleep"):
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        result = client.query("hi")

        assert result == "hello"
        mock_popen.assert_called_once()
        assert mock_popen.call_args.args[0] == ["ollama", "serve"]
        mock_run.assert_not_called()  # model already present, no pull needed


def test_query_pulls_model_when_missing():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")
    with patch("utilities.ollama_client.requests.get") as mock_get, \
         patch("utilities.ollama_client.requests.post") as mock_post, \
         patch("utilities.ollama_client.subprocess.run") as mock_run:
        mock_get.return_value = _mock_ok_response({"models": [{"name": "some-other-model:latest"}]})
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        client.query("hi")

        mock_run.assert_called_once_with(["ollama", "pull", "gemma4:31b"], check=True)


def test_query_raises_clear_error_when_ollama_not_installed():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")
    with patch("utilities.ollama_client.requests.get", side_effect=requests.exceptions.ConnectionError()), \
         patch("utilities.ollama_client.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="not installed"):
            client.query("hi")


def test_auto_start_false_skips_bootstrap_entirely():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b", auto_start=False)
    with patch("utilities.ollama_client.requests.get") as mock_get, \
         patch("utilities.ollama_client.requests.post") as mock_post, \
         patch("utilities.ollama_client.subprocess.Popen") as mock_popen:
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        result = client.query("hi")

        assert result == "hello"
        mock_get.assert_not_called()
        mock_popen.assert_not_called()


def test_query_default_max_tokens_is_16384():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")
    with patch("utilities.ollama_client.requests.get") as mock_get, \
         patch("utilities.ollama_client.requests.post") as mock_post:
        mock_get.return_value = _mock_ok_response({"models": [{"name": "gemma4:31b"}]})
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        client.query("hi")

        assert mock_post.call_args.kwargs["json"]["options"]["num_predict"] == 16384


def test_bootstrap_only_runs_once():
    from utilities.ollama_client import OllamaClient

    client = OllamaClient(model="gemma4:31b")
    with patch("utilities.ollama_client.requests.get") as mock_get, \
         patch("utilities.ollama_client.requests.post") as mock_post:
        mock_get.return_value = _mock_ok_response({"models": [{"name": "gemma4:31b"}]})
        mock_post.return_value = _mock_ok_response({"response": "hello"})

        client.query("hi")
        client.query("hi again")

        assert mock_get.call_count == 1  # reachability+tags check only happens once, not per query
