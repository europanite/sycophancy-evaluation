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
