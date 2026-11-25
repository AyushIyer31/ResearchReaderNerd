import json
import os

import anthropic

from researchreadernerd import cli

from .conftest import MINIMAL_PDF, TEXT_PDF


def test_writes_report_and_json(tmp_path, monkeypatch, fake_ollama, capsys):
    monkeypatch.chdir(tmp_path)  # keep the project's real .env out of tests
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(TEXT_PDF)
    monkeypatch.setattr(cli, "make_client", lambda model: fake_ollama.client)

    out = tmp_path / "out"
    assert cli.main([str(pdf), "-o", str(out)]) == 0
    assert "local model, no API cost" in capsys.readouterr().err
    assert (out / "paper.md").read_text().startswith("# A Sample Paper")
    assert "<h1>A Sample Paper</h1>" in (out / "paper.html").read_text()
    assert json.loads((out / "paper.json").read_text())["proposed_method"] == "SampleNet"


def test_claude_model_uses_the_anthropic_client(tmp_path, monkeypatch, fake_api):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(MINIMAL_PDF)
    monkeypatch.setattr(cli, "make_client", lambda model: fake_api.client)

    assert cli.main([str(pdf), "-o", str(tmp_path / "out"), "--model", "claude-haiku-5-5"]) == 0
    assert fake_api.requests[0]["model"] == "claude-haiku-5-5"


def test_missing_credentials_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep the project's real .env out of tests
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(MINIMAL_PDF)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(cli, "make_client", lambda model: anthropic.Anthropic(api_key=None, auth_token=None))

    assert cli.main([str(pdf), "-o", str(tmp_path / "out"), "--model", "claude-haiku-5-5"]) == 1
    assert "No Anthropic credentials found" in capsys.readouterr().err


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport RRN_TEST_A='one'\nRRN_TEST_B=two\nRRN_TEST_C=from-file\n")
    monkeypatch.delenv("RRN_TEST_A", raising=False)
    monkeypatch.delenv("RRN_TEST_B", raising=False)
    monkeypatch.setenv("RRN_TEST_C", "already-set")

    cli.load_dotenv(env)
    assert os.environ["RRN_TEST_A"] == "one"
    assert os.environ["RRN_TEST_B"] == "two"
    assert os.environ["RRN_TEST_C"] == "already-set"
