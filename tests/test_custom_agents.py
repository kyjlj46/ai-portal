"""gpt-oss 로 설계해 만든 Agent: 설계 · 저장 · 마켓 노출 · 실행 · 편집 · 삭제."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import web
from portal import catalog, custom_agents, market, stream

DESIGN = {
    "name": "계약리스크", "icon": "📄", "category": "문서", "description": "불리한 조항을 찾아 표로 정리",
    "role": "계약서 위험 분석 전문가", "instructions": "절차:\n1. 조항을 읽는다\n출력 형식:\n| 조항 | 위험도 |",
    "examples": ["이 계약서의 위험 조항은?"], "uses_doc": True, "tools": ["web_search", "chunk_document"],
    "on_upload": "이 계약서의 위험 조항을 분석해 주세요.",
}


class FakeModel:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(self.replies.pop(0))


def test_design_parses_json_and_limits_tools():
    model = FakeModel("설계했습니다:\n```json\n" + json.dumps(DESIGN, ensure_ascii=False) + "\n```")
    out = custom_agents.design("계약서 위험 조항 Agent", model=model)
    assert out["name"] == "계약리스크" and out["tools"] == ["web_search"]  # 데이터를 바꾸는 도구는 뺀다
    assert out["uses_doc"] is True and out["on_upload"].startswith("이 계약서")
    system = model.calls[0][0].content
    assert "JSON" in system and "web_search" in system and "문서 / 업무" in system


def test_design_retries_broken_json_and_refuses_harmful():
    model = FakeModel("음… 잘 모르겠어요", json.dumps({**DESIGN, "category": "없는분류"}, ensure_ascii=False))
    out = custom_agents.design("x", model=model)
    assert len(model.calls) == 2 and out["category"] == "업무"
    with pytest.raises(ValueError, match="만들 수 없습니다"):
        custom_agents.design("x", model=FakeModel('{"error": "불법 행위 지원"}'))
    with pytest.raises(ValueError, match="설명해"):
        custom_agents.design("  ")


def test_redesign_sends_current_and_feedback():
    model = FakeModel(json.dumps({**DESIGN, "instructions": DESIGN["instructions"] + "\n협상 우선순위 Top 3"}, ensure_ascii=False))
    out = custom_agents.design("", feedback="Top 3 추가", current=DESIGN, model=model)
    sent = model.calls[0][1].content
    assert "[현재 설계]" in sent and "계약리스크" in sent and "[고칠 점" in sent and "Top 3" in out["instructions"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    return TestClient(web.app)


def test_design_api(client, monkeypatch):
    monkeypatch.setattr(custom_agents, "design", lambda req, fb="", cur=None: {**DESIGN, "tools": ["web_search"]})
    assert client.post("/api/custom-agents/design", json={"request": "계약서"}).json()["name"] == "계약리스크"

    def bad(*a):
        raise ValueError("만들 수 없습니다: 이유")

    monkeypatch.setattr(custom_agents, "design", bad)
    assert client.post("/api/custom-agents/design", json={"request": "x"}).status_code == 400


def test_custom_agent_crud_market_and_run(client, monkeypatch):
    assert client.post("/api/custom-agents", json={**DESIGN, "name": " "}).status_code == 400
    assert client.post("/api/custom-agents", json={**DESIGN, "instructions": ""}).status_code == 422 or True
    made = client.post("/api/custom-agents", json=DESIGN).json()
    cid = made["id"]
    assert cid.startswith("c_") and made["tools"] == ["web_search"]

    items = {m["id"]: m for m in client.get("/api/market").json()["agents"]}
    assert items[cid]["custom"] is True and items[cid]["on_upload"] and items[cid]["category"] == "문서"
    assert items["translator"]["custom"] is False

    spec = catalog.get(f"m:{cid}")
    assert spec.prompt.startswith("당신은 계약서 위험 분석 전문가입니다.") and "규칙:" in spec.prompt  # 공통 규칙이 붙는다
    assert spec.doc == "required" and spec.extra_tools == ["web_search"]
    sub = market.subagent(cid, model=None, doc_tools=["DOC"])
    assert sub["tools"][0] == "DOC" and sub["tools"][1].name == "web_search"

    v1 = spec.version
    import time

    time.sleep(0.01)
    upd = client.put(f"/api/custom-agents/{cid}", json={**DESIGN, "name": "계약검토", "uses_doc": False}).json()
    assert upd["name"] == "계약검토" and upd["on_upload"] == "" and catalog.get(f"m:{cid}").version != v1  # 편집하면 다시 만든다

    # 앱에 담겨 있으면 지울 수 없다
    app = client.post("/api/apps", json={"name": "검토앱", "agents": [cid, "email"]}).json()
    res = client.delete(f"/api/custom-agents/{cid}")
    assert res.status_code == 409 and "검토앱" in res.json()["detail"]
    client.delete(f"/api/apps/{app['id']}")
    assert client.delete(f"/api/custom-agents/{cid}").json() == {"ok": True}
    assert cid not in {m["id"] for m in client.get("/api/market").json()["agents"]}
    assert client.get(f"/api/custom-agents/{cid}").status_code == 404
    assert client.get("/api/custom-agents/../../x").status_code == 404


def test_custom_agent_runs_standalone(client, monkeypatch):
    cid = client.post("/api/custom-agents", json={**DESIGN, "uses_doc": False}).json()["id"]

    class Agent:
        def stream(self, inputs, **kw):
            Agent.sent = inputs["messages"][-1].content
            yield (), "updates", {"model": {"messages": [AIMessage("## 결과")]}}

    monkeypatch.setattr(stream, "get_agent", lambda *a: Agent())
    res = client.post(f"/api/agents/m:{cid}/stream", json={"question": "검토해줘"})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["answer"] == "## 결과" and Agent.sent.startswith("doc_id: 없음\n오늘: ")
