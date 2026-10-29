import json
import time

import pytest
from fastapi.testclient import TestClient

from researchreadernerd.analyze import Prompts
from researchreadernerd.storage import LocalStorage
from researchreadernerd.web import create_app

from .conftest import MINIMAL_PDF


@pytest.fixture
def app_client(tmp_path, monkeypatch, fake_api):
    monkeypatch.setattr("researchreadernerd.paper._download", lambda url: MINIMAL_PDF)
    reports = tmp_path / "reports"
    return TestClient(create_app(LocalStorage(reports), client_factory=lambda: fake_api.client)), reports


def run_body(paper="1706.03762", **prompt_overrides):
    return {"paper": paper, "effort": "low", "prompts": {**Prompts().to_dict(), **prompt_overrides}}


def wait_for(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/runs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_index_and_config(app_client):
    client, _ = app_client
    assert "ResearchReaderNerd" in client.get("/").text
    config = client.get("/api/config").json()
    assert [s["id"] for s in config["stages"]] == ["fetch", "analyze", "plan", "related", "catalogs", "rank", "write"]
    keys = [p["key"] for s in config["stages"] for p in s["prompts"]]
    assert set(keys) == set(config["defaults"]) == set(Prompts().to_dict())


def test_full_run_uses_edited_prompts_and_saves_them(app_client, fake_api):
    client, reports = app_client
    job = client.post("/api/runs", json=run_body(user="Only list benchmarks.")).json()
    assert job["label"] == "arXiv 1706.03762"

    job = wait_for(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert fake_api.requests[0]["messages"][0]["content"][1]["text"] == "Only list benchmarks."
    assert fake_api.requests[0]["output_config"]["effort"] == "low"

    run = json.loads((reports / "1706.03762.run.json").read_text())
    assert run["prompts"]["user"] == "Only list benchmarks."
    assert run["prompts_are_default"] is False

    listed = client.get("/api/reports").json()
    assert listed[0]["name"] == "1706.03762" and listed[0]["custom_prompts"] is True
    report = client.get("/api/reports/1706.03762").json()
    assert "<h1>A Sample Paper</h1>" in report["html"]
    assert report["run"]["input_tokens"] == 40  # analyze, plan, extract, rank: 10 each from the fake API
    assert [c["step"] for c in report["run"]["calls"]] == ["analyze", "plan", "extract", "rank"]
    assert "Benchmarks to consider" in report["html"]


def test_failed_run_reports_plain_error(app_client, fake_api):
    client, _ = app_client
    fake_api.reply = {**fake_api.reply, "stop_reason": "max_tokens"}
    job = wait_for(client, client.post("/api/runs", json=run_body()).json()["id"])
    assert job["status"] == "error"
    assert "token limit" in job["error"]
    assert job["stage"] == "analyze"


def test_rejects_non_arxiv_input(app_client):
    client, _ = app_client
    res = client.post("/api/runs", json=run_body(paper="/etc/passwd"))
    assert res.status_code == 400


def test_rejects_bad_report_names(app_client):
    client, _ = app_client
    assert client.get("/api/reports/..%2F..%2Fsecrets").status_code == 404
    assert client.get("/api/reports/missing").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404


def upload(client, data=MINIMAL_PDF, filename="My Paper (v2).pdf"):
    return client.post("/api/uploads", params={"filename": filename}, content=data,
                       headers={"Content-Type": "application/pdf"})


def test_upload_then_run(app_client, fake_api):
    client, reports = app_client
    uploaded = upload(client).json()
    assert uploaded["name"] == "My-Paper-v2"

    body = {**run_body(), "paper": None, "upload_id": uploaded["upload_id"]}
    job = wait_for(client, client.post("/api/runs", json=body).json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["label"] == "My Paper (v2).pdf"
    assert job["report"] == "My-Paper-v2"
    assert fake_api.requests[0]["messages"][0]["content"][0]["title"] == "My-Paper-v2"
    run = json.loads((reports / "My-Paper-v2.run.json").read_text())
    assert run["source"] == "Uploaded file: My Paper (v2).pdf"
    assert client.get("/api/reports").json()[0]["name"] == "My-Paper-v2"


def test_upload_rejects_non_pdf(app_client):
    client, _ = app_client
    res = upload(client, data=b"hello", filename="notes.pdf")
    assert res.status_code == 400 and "not a PDF" in res.json()["detail"]


def test_run_needs_exactly_one_source(app_client):
    client, _ = app_client
    uploaded = upload(client).json()
    both = {**run_body(), "upload_id": uploaded["upload_id"]}
    neither = {**run_body(), "paper": None}
    assert client.post("/api/runs", json=both).status_code == 400
    assert client.post("/api/runs", json=neither).status_code == 400


def test_unknown_upload_id(app_client):
    client, _ = app_client
    for bad in ("0" * 32, "../../etc/passwd"):
        res = client.post("/api/runs", json={**run_body(), "paper": None, "upload_id": bad})
        assert res.status_code == 400 and "Upload the PDF again" in res.json()["detail"]


def make_client(tmp_path, fake_api, **kwargs):
    return TestClient(create_app(LocalStorage(tmp_path / "r"), client_factory=lambda: fake_api.client, **kwargs))


def test_health_is_open_even_with_origin_secret(tmp_path, fake_api):
    client = make_client(tmp_path, fake_api, origin_secret="s3cret")
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 403
    assert client.get("/api/config", headers={"x-origin-secret": "wrong"}).status_code == 403
    assert client.get("/api/config", headers={"x-origin-secret": "s3cret"}).status_code == 200


def test_daily_run_limit(tmp_path, monkeypatch, fake_api):
    monkeypatch.setattr("researchreadernerd.paper._download", lambda url: MINIMAL_PDF)
    client = make_client(tmp_path, fake_api, daily_run_limit=1)
    first = client.post("/api/runs", json=run_body())
    assert first.status_code == 200
    wait_for(client, first.json()["id"])
    second = client.post("/api/runs", json=run_body())
    assert second.status_code == 429 and "daily limit of 1" in second.json()["detail"]


def test_run_survives_in_storage_and_restart_is_reported(tmp_path, monkeypatch, fake_api):
    monkeypatch.setattr("researchreadernerd.paper._download", lambda url: MINIMAL_PDF)
    storage = LocalStorage(tmp_path / "r")
    first = TestClient(create_app(storage, client_factory=lambda: fake_api.client))
    done = wait_for(first, first.post("/api/runs", json=run_body()).json()["id"])

    # A new server process (as after a deploy) can still answer for a finished run...
    second = TestClient(create_app(storage, client_factory=lambda: fake_api.client))
    assert second.get(f"/api/runs/{done['id']}").json()["status"] == "done"

    # ...a run another server is still working on (as during a canary deploy) shows as running...
    key = f"jobs/{done['id'][:8]}/{done['id']}.json"
    record = json.loads(storage.read_text(key))
    record.update(status="running", report=None, updated=time.time())
    storage.write_text(key, json.dumps(record))
    assert second.get(f"/api/runs/{done['id']}").json()["status"] == "running"

    # ...and one whose server stopped long ago is reported as interrupted.
    record.update(updated=time.time() - 20 * 60)
    storage.write_text(key, json.dumps(record))
    res = second.get(f"/api/runs/{done['id']}").json()
    assert res["status"] == "error" and "restarted" in res["error"]


def basic(password, user="me"):
    import base64
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_password_protects_everything_but_health(tmp_path, fake_api):
    client = make_client(tmp_path, fake_api, basic_auth_password="open sesame")
    assert client.get("/health").status_code == 200
    res = client.get("/")
    assert res.status_code == 401 and res.headers["www-authenticate"].startswith("Basic")
    assert client.get("/", headers=basic("wrong")).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic !!notbase64"}).status_code == 401
    assert client.get("/api/config", headers=basic("open sesame")).status_code == 200


def test_password_or_origin_secret_either_works(tmp_path, fake_api):
    client = make_client(tmp_path, fake_api, basic_auth_password="pw", origin_secret="s3cret")
    assert client.get("/api/config", headers={"x-origin-secret": "s3cret"}).status_code == 200
    assert client.get("/api/config", headers=basic("pw")).status_code == 200
    assert client.get("/api/config").status_code == 401
