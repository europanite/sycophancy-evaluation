import llm


def test_runtime_defaults(monkeypatch):
    for key in (
        "OLLAMA_CONNECT_TIMEOUT_SECONDS",
        "OLLAMA_READ_TIMEOUT_SECONDS",
        "OLLAMA_MAX_TOKENS",
        "OLLAMA_KEEP_ALIVE",
    ):
        monkeypatch.delenv(key, raising=False)

    assert llm.get_connect_timeout_seconds() == 10
    assert llm.get_read_timeout_seconds() == 600
    assert llm.get_max_tokens() == 128
    assert llm.get_keep_alive() == "30m"


def test_call_uses_configured_timeout_token_cap_and_keep_alive(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "B"}}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setenv("OLLAMA_CONNECT_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("OLLAMA_READ_TIMEOUT_SECONDS", "321")
    monkeypatch.setenv("OLLAMA_MAX_TOKENS", "64")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "45m")
    monkeypatch.setattr(llm._session, "post", fake_post)

    out = llm.call_ollama_messages([{"role": "user", "content": "test"}], model="m")

    assert out == "B"
    assert captured["timeout"] == (7.0, 321.0)
    assert captured["json"]["options"]["num_predict"] == 64
    assert captured["json"]["keep_alive"] == "45m"


def test_ollama_stats_are_returned_without_changing_text_api(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "result"}, "prompt_eval_count": 31, "eval_count": 7, "eval_duration": 500_000_000}

    monkeypatch.setattr(llm._session, "post", lambda *args, **kwargs: DummyResponse())
    output = llm.call_ollama_messages_with_stats([{"role": "user", "content": "test"}])
    assert output == {"content": "result", "prompt_tokens": 31, "completion_tokens": 7, "eval_duration_ns": 500_000_000}
    assert llm.call_ollama_messages([{"role": "user", "content": "test"}]) == "result"


def test_json_judge_can_use_larger_token_budget_without_affecting_default(monkeypatch):
    payloads = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "{}"}}

    def fake_post(url, *, json, timeout):
        payloads.append(json)
        return DummyResponse()

    monkeypatch.setattr(llm._session, "post", fake_post)
    llm.call_ollama_messages_with_stats([{"role": "user", "content": "test"}], max_tokens=1024, response_format="json")
    llm.call_ollama_messages([{"role": "user", "content": "test"}])
    assert payloads[0]["options"]["num_predict"] == 1024
    assert payloads[0]["format"] == "json"
    assert payloads[1]["options"]["num_predict"] == llm.get_max_tokens()
    assert "format" not in payloads[1]
