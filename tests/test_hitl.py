"""질의자 확인 (HITL): 에이전트가 ask_user 로 물으면 멈추고, 답을 받으면 멈춘 자리에서 이어서 한다."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

import web
from portal import hitl, stream


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    return TestClient(web.app)


def ask(q="누구에게 보낼까요?", options=("팀장", "전체 팀"), cid="a1"):
    return AIMessage("", tool_calls=[{"name": "ask_user", "args": {"question": q, "options": list(options), "why": "받는 사람"}, "id": cid}])


class AskingAgent:
    """처음엔 질의자에게 묻고, 답(ToolMessage)을 받으면 그 답으로 결과를 쓴다."""

    def __init__(self):
        self.inputs, self.configs = [], []

    def stream(self, inputs, config=None, **kw):
        self.inputs.append(inputs["messages"])
        self.configs.append(config)
        last = inputs["messages"][-1]
        if isinstance(last, ToolMessage) and last.name == "ask_user":
            yield (), "updates", {"model": {"messages": [AIMessage(f"# 메일\n받는 사람: {last.content}")]}}
        else:
            yield (), "updates", {"model": {"messages": [ask()]}}


def test_ask_user_tool_answers_by_config():
    assert "확인 대기" in hitl.ask_user.invoke({"question": "q"})
    closed = hitl.ask_user.invoke({"question": "q"}, config={"configurable": {"ask_closed": True}})
    assert "더 이상 질의자에게 물을 수 없습니다" in closed and "가정" in closed


def test_main_agent_pauses_and_resumes_where_it_stopped(monkeypatch):
    agent = AskingAgent()
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)
    history: list = []
    out = list(stream.ask_stream(None, "회의 결과 메일 써줘", history, agent_id="m:email"))
    clar = out[-1]
    assert clar["type"] == "clarify" and clar["question"] == "누구에게 보낼까요?" and clar["options"] == ["팀장", "전체 팀"]
    assert clar["why"] == "받는 사람" and clar["asks_left"] == hitl.MAX_ASKS - 1
    assert [it["name"] for s in clar["flow"] for it in s["items"]][-1] == "질의자 확인"
    assert history == []  # 끝나지 않은 턴은 기록에 남기지 않는다

    out = list(stream.ask_stream(None, "", history, agent_id="m:email", resume={"id": clar["id"], "answer": "팀장"}))
    final = out[-1]
    assert final["type"] == "final" and "받는 사람: 팀장" in final["answer"]
    resumed = agent.inputs[1]  # 처음부터 다시가 아니라: 원래 요청 + ask_user 호출 + 답
    assert "회의 결과 메일 써줘" in resumed[0].content and resumed[-1].content == "팀장" and resumed[-1].tool_call_id == "a1"
    names = [[it["name"] for it in s["items"]] for s in final["flow"]]
    assert names[1] == ["질의자 확인"] and names[2][0].endswith("· 이어서")
    assert final["flow"][1]["items"][0]["note"] == "팀장"
    assert "[질의자 확인]" in history[0].content and "답: 팀장" in history[0].content  # 다음 질문의 맥락

    again = list(stream.ask_stream(None, "", history, agent_id="m:email", resume={"id": clar["id"], "answer": "x"}))
    assert again[-1]["type"] == "error" and "만료" in again[-1]["message"]  # 한 번만 이어서 할 수 있다


def test_questions_are_capped(monkeypatch):
    agent = AskingAgent()
    agent.stream_orig = agent.stream

    def always_ask(inputs, config=None, **kw):  # 계속 묻는 에이전트
        agent.configs.append(config)
        yield (), "updates", {"model": {"messages": [ask(cid=f"a{len(agent.configs)}")]}}
        if config["configurable"]["ask_closed"]:
            yield (), "updates", {"tools": {"messages": [ToolMessage("가정하고 진행", tool_call_id=f"a{len(agent.configs)}", name="ask_user")]}}
            yield (), "updates", {"model": {"messages": [AIMessage("가정: 팀장\n# 메일")]}}

    agent.stream = always_ask
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)
    ev = list(stream.ask_stream(None, "메일", None, agent_id="m:email"))[-1]
    for _ in range(hitl.MAX_ASKS - 1):
        ev = list(stream.ask_stream(None, "", None, agent_id="m:email", resume={"id": ev["id"], "answer": ""}))[-1]
        assert ev["type"] == "clarify"
    final = list(stream.ask_stream(None, "", None, agent_id="m:email", resume={"id": ev["id"], "answer": ""}))[-1]
    assert final["type"] == "final" and final["answer"].startswith("가정: 팀장")
    assert [c["configurable"]["ask_closed"] for c in agent.configs] == [False] * hitl.MAX_ASKS + [True]


def test_empty_answer_means_proceed_on_assumptions(monkeypatch):
    agent = AskingAgent()
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)
    clar = list(stream.ask_stream(None, "메일", None, agent_id="m:email"))[-1]
    final = list(stream.ask_stream(None, "", None, agent_id="m:email", resume={"id": clar["id"], "answer": " "}))[-1]
    assert hitl.SKIP_ANSWER in final["answer"] and final["flow"][1]["items"][0]["note"] == "알아서 진행"


# ─────────────────────────── 앱 '순서대로' ───────────────────────────

APP = {"name": "회의 후속", "icon": "🤝", "description": "", "instructions": "", "doc": "none",
       "agents": ["minutes", "action_items", "email"], "run": "pipeline"}


class Step:
    seen: list = []

    def __init__(self, name):
        self.name = name

    def invoke(self, inputs, config=None):
        msgs = inputs["messages"]
        Step.seen.append((self.name, msgs))
        if self.name == "m:email" and not isinstance(msgs[-1], ToolMessage) and not config["configurable"]["ask_closed"]:
            return {"messages": [*msgs, ask(), ToolMessage("확인 대기", tool_call_id="a1", name="ask_user"), AIMessage("확인 대기")]}
        tail = f" → {msgs[-1].content}" if isinstance(msgs[-1], ToolMessage) else ""
        return {"messages": [*msgs, AIMessage(f"{self.name} 결과{tail}")]}


def test_pipeline_pauses_at_step_and_resumes_without_rerunning_done_steps(client, monkeypatch):
    app = client.post("/api/apps", json=APP).json()
    Step.seen = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Step(agent_id))
    out = list(stream.ask_stream(None, "메모", None, agent_id=f"app:{app['id']}"))
    clar = out[-1]
    assert clar["type"] == "clarify" and "이메일" in clar["who"]
    assert [[it["name"] for it in s["items"]] for s in clar["flow"]][-1] == ["질의자 확인"]

    Step.seen = []
    final = list(stream.ask_stream(None, "", None, agent_id=f"app:{app['id']}", resume={"id": clar["id"], "answer": "전체 팀"}))[-1]
    assert [n for n, _ in Step.seen] == ["m:email"]  # 끝난 단계는 다시 돌리지 않는다
    msgs = Step.seen[0][1]
    assert msgs[-1].content == "전체 팀" and "[앞 단계 결과: ✅ 할 일 추출]" in msgs[0].content  # 멈춘 대화에서 이어서
    assert final["type"] == "final" and "m:email 결과 → 전체 팀" in final["answer"] and "m:minutes 결과" in final["answer"]
    names = [it["name"] for s in final["flow"] for it in s["items"]]
    assert names == ["회의록 정리", "할 일 추출", "질의자 확인", "이메일 작성"]


def test_web_resume_endpoint(client, monkeypatch):
    agent = AskingAgent()
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)

    def events(res):
        return [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]

    clar = events(client.post("/api/agents/m:email/stream", json={"question": "메일 써줘", "conversation_id": "c1"}))[-1]
    assert clar["type"] == "clarify"
    final = events(client.post("/api/agents/m:email/stream", json={"conversation_id": "c1", "resume": {"id": clar["id"], "answer": "팀장"}}))[-1]
    assert final["type"] == "final" and "팀장" in final["answer"]
    assert client.post("/api/agents/m:email/stream", json={"resume": {"id": "x", "answer": "a" * 4001}}).status_code == 400
