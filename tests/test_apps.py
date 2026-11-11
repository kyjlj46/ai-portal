"""Agent 마켓 · 커스텀 앱(DeepAgent 하위 에이전트 조합)."""

import json

import pytest
from fastapi.testclient import TestClient

import web
from rag import apps, catalog, market, stream


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(web.app)


APP = {"name": "회의 후속 처리", "icon": "🤝", "description": "회의록·할 일·메일", "instructions": "순서대로",
       "doc": "none", "agents": ["minutes", "action_items", "email"], "examples": ["메모를 붙여넣으세요", " "]}


def test_market_lists_ten_new_agents_and_building_blocks(client):
    data = client.get("/api/market").json()
    items = {m["id"]: m for m in data["agents"]}
    new = [m for m in data["agents"] if m["new"]]
    assert len(new) == 14
    assert {"translator", "proofreader", "email", "minutes", "action_items", "keywords", "sentiment", "swot", "slides", "sql",
            "quick_summary", "chunker", "ocr", "web_search"} == {m["id"] for m in new}
    assert items["quick_summary"]["on_upload"] and items["chunker"]["on_upload"] and items["ocr"]["on_upload"]
    assert items["web_search"]["on_upload"] == "" and items["web_search"]["category"] == "리서치"
    assert items["doc_search"]["route"] == "qa" and items["summary"]["route"] == "summary"  # 기존 기능도 부품으로
    assert items["sql"]["uses_doc"] is False
    assert set(data["categories"]) >= {m["category"] for m in data["agents"]}
    assert "prompt" not in items["translator"]


def test_market_agent_runs_standalone():
    spec = catalog.get("m:translator")
    assert spec.mode == "chat" and spec.doc == "optional" and "번역" in spec.prompt and spec.group == "마켓"
    assert catalog.get("m:sql").doc == "none"
    with pytest.raises(KeyError):
        catalog.get("m:nope")


def test_app_crud_and_validation(client):
    assert client.post("/api/apps", json={**APP, "name": " "}).status_code == 400
    assert client.post("/api/apps", json={**APP, "agents": []}).status_code == 400
    assert client.post("/api/apps", json={**APP, "agents": ["minutes", "minutes"]}).status_code == 400
    assert client.post("/api/apps", json={**APP, "agents": ["nope"]}).status_code == 400
    assert client.post("/api/apps", json={**APP, "agents": [m.id for m in market.MARKET[:9]]}).status_code == 400  # 8개 초과
    assert client.post("/api/apps", json={**APP, "doc": "maybe"}).status_code == 400

    app = client.post("/api/apps", json=APP).json()
    assert len(app["id"]) == 10 and app["examples"] == ["메모를 붙여넣으세요"]
    assert [a["id"] for a in client.get("/api/apps").json()] == [app["id"]]
    upd = client.put(f"/api/apps/{app['id']}", json={**APP, "name": "회의 정리", "agents": ["minutes", "email"]}).json()
    assert upd["name"] == "회의 정리" and upd["agents"] == ["minutes", "email"] and upd["created"] == app["created"]
    assert client.get(f"/api/apps/{app['id']}").json()["name"] == "회의 정리"
    assert client.put("/api/apps/0123456789", json=APP).status_code == 404
    assert client.get("/api/apps/../../etc").status_code == 404
    assert client.delete(f"/api/apps/{app['id']}").json() == {"ok": True}
    assert client.get("/api/apps").json() == []
    assert client.delete(f"/api/apps/{app['id']}").status_code == 404


def test_app_spec_orchestrates_team(client):
    app = client.post("/api/apps", json=APP).json()
    spec = catalog.get(f"app:{app['id']}")
    assert spec.subagents == ["minutes", "action_items", "email"] and spec.doc == "none" and spec.mode == "chat"
    assert "회의 후속 처리" in spec.prompt and "(id: action_items)" in spec.prompt and "순서대로" in spec.prompt
    assert spec.prompt.index("minutes") < spec.prompt.index("action_items") < spec.prompt.index("(id: email)")
    with pytest.raises(KeyError):
        catalog.get("app:0123456789")


def test_build_agent_attaches_market_subagents(client, monkeypatch):
    from rag import agent

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    app = client.post("/api/apps", json={**APP, "agents": ["doc_search", "sql"]}).json()
    seen = {}
    monkeypatch.setattr(agent, "create_deep_agent", lambda **kw: seen.update(kw) or "graph")
    assert agent.build_agent(f"app:{app['id']}") == "graph"
    subs = {s["name"]: s for s in seen["subagents"]}
    assert set(subs) == {"doc_search", "sql"}
    assert subs["doc_search"]["tools"] and subs["sql"]["tools"] == []  # 문서 도구는 필요한 에이전트에만
    assert "doc_id" in subs["doc_search"]["description"]
    assert seen["system_prompt"].startswith("당신은 '회의 후속 처리' 앱의 총괄 에이전트")


def test_editing_app_rebuilds_cached_agent(client, monkeypatch):
    app = client.post("/api/apps", json=APP).json()
    built = []
    monkeypatch.setattr("rag.agent.build_agent", lambda agent_id: built.append(agent_id) or object())
    stream._agents.clear()
    a1 = stream.get_agent(f"app:{app['id']}")
    assert stream.get_agent(f"app:{app['id']}") is a1 and len(built) == 1  # 재사용
    import time

    time.sleep(0.01)
    client.put(f"/api/apps/{app['id']}", json={**APP, "agents": ["email"]})
    assert stream.get_agent(f"app:{app['id']}") is not a1 and len(built) == 2  # 편집 후 새로 만듦
    assert len([k for k in stream._agents if k.startswith(f"app:{app['id']}@")]) == 1  # 예전 것은 버림
    stream._agents.clear()


def test_app_stream_gets_today_and_conversation_cleanup(client, monkeypatch):
    import json

    from langchain_core.messages import AIMessage

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()  # 총괄 에이전트 모드

    class Fake:
        inputs = None

        def stream(self, inputs, **kw):
            Fake.inputs = inputs
            call = AIMessage("", tool_calls=[{"name": "task", "args": {"subagent_type": "minutes", "description": "정리"}, "id": "t1"}])
            yield (), "updates", {"model": {"messages": [call]}}  # 팀에 맡긴 뒤
            yield (), "updates", {"model": {"messages": [AIMessage("# 회의록\n- 결과")]}}

    monkeypatch.setattr(stream, "get_agent", lambda *a: Fake())
    res = client.post(f"/api/agents/app:{app['id']}/stream", json={"question": "메모: 오픈 연기", "conversation_id": "c7"})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["type"] == "final" and events[-1]["agent"] == f"app:{app['id']}"
    sent = Fake.inputs["messages"][-1].content
    assert sent.startswith("doc_id: 없음\n오늘: ") and "메모: 오픈 연기" in sent
    assert f"app:{app['id']}:c7" in stream._conversations
    client.delete("/api/conversations/c7")
    assert f"app:{app['id']}:c7" not in stream._conversations
    assert client.post("/api/agents/app:0123456789/stream", json={"question": "q"}).status_code == 404


def test_task_step_shows_market_agent_name():
    ev = stream._describe_call("task", {"subagent_type": "minutes", "description": "메모 정리"}, False)
    assert ev["text"].startswith("🗒️ 회의록 정리 에게 위임")


def test_apps_dir_is_inside_data_dir(tmp_path):
    from rag import store

    app = apps.save(APP)
    assert (store.DATA_DIR / "apps" / f"{app['id']}.json").exists()


def test_run_and_output_validation(client):
    assert client.post("/api/apps", json={**APP, "run": "magic"}).status_code == 400
    assert client.post("/api/apps", json={**APP, "output": "some"}).status_code == 400
    app = client.post("/api/apps", json=APP).json()
    assert app["run"] == "pipeline" and app["output"] == "all"  # 기본: 순서대로 · 모든 단계
    spec = catalog.get(f"app:{app['id']}")
    assert spec.run == "pipeline" and spec.instructions == "순서대로"


class StepAgent:
    """마켓 에이전트 흉내: 받은 요청을 기록하고 자기 이름으로 답한다."""

    seen: list = []

    def __init__(self, name):
        self.name = name

    def invoke(self, inputs, config=None):
        from langchain_core.messages import AIMessage

        StepAgent.seen.append((self.name, inputs["messages"][-1].content))
        return {"messages": [*inputs["messages"], AIMessage(f"# {self.name} 결과\n```\n{self.name} 본문 내용입니다\n```")]}


def _run(app_id, question, history=None):
    return list(stream.ask_stream(None, question, history, agent_id=f"app:{app_id}"))


def test_pipeline_runs_every_agent_in_order_and_passes_results(client, monkeypatch):
    app = client.post("/api/apps", json=APP).json()
    StepAgent.seen = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: StepAgent(agent_id))
    out = _run(app["id"], "메모: 오픈 연기")
    assert [n for n, _ in StepAgent.seen] == ["m:minutes", "m:action_items", "m:email"]
    first, third = StepAgent.seen[0][1], StepAgent.seen[2][1]
    assert "요청: 메모: 오픈 연기" in first and "[앱 지시사항]\n순서대로" in first and "오늘: " in first
    assert "1/3 단계 '회의록 정리'" in first and "다른 단계(할 일 추출, 이메일 작성)" in first
    assert "[앞 단계 결과: 🗒️ 회의록 정리]" in third and "[앞 단계 결과: ✅ 할 일 추출]" in third  # 앞 결과를 넘김
    delegates = [e["text"] for e in out if e.get("kind") == "delegate"]
    assert delegates == ["1/3 🗒️ 회의록 정리 작업 중…", "2/3 ✅ 할 일 추출 작업 중…", "3/3 📧 이메일 작성 작업 중…"]
    final = out[-1]
    assert final["type"] == "final" and final["agent"] == f"app:{app['id']}"
    ans = final["answer"]
    assert ans.index("## 🗒️ 회의록 정리") < ans.index("## ✅ 할 일 추출") < ans.index("## 📧 이메일 작성")
    assert "### m:minutes 결과" in ans  # 단계 안의 제목은 한 단계 아래로
    assert "```" not in ans and "m:email 본문 내용입니다" in ans  # 글을 감싼 코드 블록은 푼다


def test_pipeline_last_output_folds_earlier_steps_and_follow_up(client, monkeypatch):
    app = client.post("/api/apps", json={**APP, "output": "last"}).json()
    StepAgent.seen = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: StepAgent(agent_id))
    history: list = []
    ans = _run(app["id"], "메모", history)[-1]["answer"]
    assert ans.startswith("# m:email 결과")
    assert "<details><summary>중간 단계 결과 보기 (2개)</summary>" in ans and "#### 🗒️ 회의록 정리" in ans
    StepAgent.seen = []
    _run(app["id"], "메일을 더 짧게", history)  # 이어지는 요청: 직전 결과를 고치게 한다
    assert "[직전 결과 — 위 요청이 수정 요청이면" in StepAgent.seen[0][1] and "m:email 결과" in StepAgent.seen[0][1]


def test_auto_mode_uses_deepagent_orchestrator(client, monkeypatch):
    from langchain_core.messages import AIMessage

    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()

    class Orchestrator:
        def stream(self, inputs, **kw):
            call = AIMessage("", tool_calls=[{"name": "task", "args": {"subagent_type": "minutes", "description": "정리"}, "id": "t1"}])
            yield (), "updates", {"model": {"messages": [call]}}  # 팀에 맡긴 뒤
            yield (), "updates", {"model": {"messages": [AIMessage("총괄 결과\n```markdown\n# 메일\n본문\n```")]}}

    used = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: used.append(agent_id) or Orchestrator())
    final = _run(app["id"], "메모")[-1]
    assert used == [f"app:{app['id']}"] and final["answer"] == "총괄 결과\n# 메일\n본문"


def test_auto_mode_falls_back_to_pipeline_when_orchestrator_fails(client, monkeypatch):
    """총괄 에이전트가 하위 에이전트 호출을 JSON 글로만 남기면(실제 사례) 그 글을 답으로 보여주지 않고 '순서대로' 로 처리한다."""
    from langchain_core.messages import AIMessage

    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()
    stub = json.dumps({"description": "Translate the following English text into Korean…"})

    class Orchestrator:
        def stream(self, inputs, **kw):
            yield (), "updates", {"model": {"messages": [AIMessage(stub)]}}

    StepAgent.seen = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Orchestrator() if agent_id.startswith("app:") else StepAgent(agent_id))
    out = _run(app["id"], "메모")
    final = out[-1]
    assert [n for n, _ in StepAgent.seen] == ["m:minutes", "m:action_items", "m:email"]  # 팀 전원이 순서대로
    assert "description" not in final["answer"] and "## 🗒️ 회의록 정리" in final["answer"]
    assert "호출 내용(JSON)만 글로 남겼습니다" in final["warnings"][0]
    assert any("순서대로" in e.get("text", "") for e in out if e["type"] == "step")


def test_pipeline_step_failure_is_reported_and_others_continue(client, monkeypatch):
    app = client.post("/api/apps", json=APP).json()

    class Broken:
        calls = 0

        def invoke(self, inputs, config=None):
            Broken.calls += 1
            raise TimeoutError("read timed out")

    StepAgent.seen = []
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Broken() if agent_id == "m:action_items" else StepAgent(agent_id))
    out = _run(app["id"], "메모")
    final = out[-1]
    assert final["type"] == "final" and [n for n, _ in StepAgent.seen] == ["m:minutes", "m:email"]  # 나머지는 계속
    assert "✅ 할 일 추출 단계 실패 — 모델 응답이 너무 오래 걸려" in final["warnings"][0]
    assert "> ⚠ 이 단계는 실패했습니다" in final["answer"]
    assert any("⚠ 할 일 추출 실패" in e.get("text", "") for e in out)

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Broken())
    err = _run(app["id"], "메모")[-1]
    assert err["type"] == "error" and err["message"].startswith("모든 단계가 실패했습니다")  # 원인 목록과 함께


def test_tool_stub_detection():
    from rag.agent import is_tool_stub

    assert is_tool_stub('{"description": "Translate the following text"}')
    assert is_tool_stub('{"subagent_type": "translator", "description": "x"}')
    assert is_tool_stub('{"description": "Translate the following English text into Korean, preserving')  # 잘린 JSON
    assert not is_tool_stub('{"name": "계약리스크", "role": "x"}')  # 도구 인자가 아닌 JSON 은 답일 수 있다
    assert not is_tool_stub("## 결과\n- 내용")


def test_unwrap_prose_fences_keeps_code():
    from rag.agent import demote_headings, unwrap_prose_fences

    assert unwrap_prose_fences("```\n안녕하세요 팀원 여러분\n```") == "안녕하세요 팀원 여러분"
    sql = "```sql\nSELECT 1;\n```"
    assert unwrap_prose_fences(sql) == sql
    assert unwrap_prose_fences("```\nfor i in range(3): print(i)\n```").startswith("```")
    assert demote_headings("# A\n```\n# 주석\n```\n## B") == "### A\n```\n# 주석\n```\n#### B"


# ─────────────────────────── 병렬 · 재시도 · 반복 한도 ───────────────────────────


def test_stages_group_independent_analysis_agents():
    assert stream.stages(["translator", "sentiment", "keywords", "web_search"]) == [["translator"], ["sentiment", "keywords", "web_search"]]
    assert stream.stages(["minutes", "action_items", "email"]) == [["minutes"], ["action_items"], ["email"]]  # 메일은 앞 결과를 모두 써야
    assert stream.stages(["keywords", "translator", "sentiment", "proofreader", "web_search"]) == [
        ["keywords"], ["translator"], ["sentiment"], ["proofreader"], ["web_search"]]


def test_parallel_stage_runs_concurrently_and_gets_previous_results(client, monkeypatch):
    import threading
    import time as _time

    from langchain_core.messages import AIMessage

    app = client.post("/api/apps", json={**APP, "agents": ["translator", "sentiment", "keywords", "web_search"]}).json()
    seen, active, peak = {}, [0], [0]
    lock = threading.Lock()

    class Slow:
        def __init__(self, name):
            self.name = name

        def invoke(self, inputs, config=None):
            assert config["recursion_limit"] == stream.STEP_LIMIT and config["configurable"] == {"ask_closed": False}  # 단계마다 반복 한도
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            seen[self.name] = inputs["messages"][-1].content
            _time.sleep(0.3)
            with lock:
                active[0] -= 1
            return {"messages": [*inputs["messages"], AIMessage(f"{self.name} 결과")]}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Slow(agent_id))
    t0 = _time.time()
    out = list(stream.ask_stream(None, "기사", None, agent_id=f"app:{app['id']}"))
    took = _time.time() - t0
    assert peak[0] == 3 and took < 1.0  # 감성·키워드·웹 검색은 동시에 (따로면 1.2초 이상)
    assert "[앞 단계 결과: 🌐 번역]" in seen["m:keywords"] and "sentiment 결과" not in seen["m:keywords"]  # 형제 결과는 안 받음
    assert "(동시 실행)" in " ".join(e.get("text", "") for e in out)
    ans = out[-1]["answer"]
    assert ans.index("## 🌐 번역") < ans.index("## 📈 감성") < ans.index("## 🏷️ 키워드") < ans.index("## 🌍 웹 검색")  # 순서 유지


def test_step_retries_up_to_three_times(client, monkeypatch):
    from langchain_core.messages import AIMessage

    app = client.post("/api/apps", json={**APP, "agents": ["minutes", "email"]}).json()
    calls = {"m:minutes": 0}

    class Flaky:
        def __init__(self, name):
            self.name = name

        def invoke(self, inputs, config=None):
            if self.name == "m:minutes":
                calls["m:minutes"] += 1
                if calls["m:minutes"] < 3:
                    raise ConnectionError("connection refused")
            return {"messages": [*inputs["messages"], AIMessage(f"{self.name} 결과")]}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Flaky(agent_id))
    out = list(stream.ask_stream(None, "메모", None, agent_id=f"app:{app['id']}"))
    assert calls["m:minutes"] == 3 and out[-1]["type"] == "final" and out[-1]["warnings"] == []  # 세 번째에 성공
    assert sum("다시 시도" in e.get("text", "") for e in out) == 2

    calls["m:minutes"] = -10  # 계속 실패 → 3번에서 멈춘다 (무한 재시도 없음)
    out = list(stream.ask_stream(None, "메모", None, agent_id=f"app:{app['id']}"))
    assert calls["m:minutes"] == -7 and "3번 시도" in out[-1]["warnings"][0] and "m:email 결과" in out[-1]["answer"]


def test_final_answer_retries_then_reports(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from rag import agent

    replies = iter(["", '{"description": "x"}', "## 답"])

    class Model:
        n = 0

        def invoke(self, msgs):
            Model.n += 1
            return AIMessage(next(replies))

    monkeypatch.setattr(agent, "_model", lambda env: Model())
    assert agent.final_answer([HumanMessage("q")]) == "## 답" and Model.n == 3  # 빈 답·흉내 뒤 세 번째에

    monkeypatch.setattr(agent, "_model", lambda env: type("M", (), {"invoke": lambda self, m: AIMessage("")})())
    with pytest.raises(agent.NoAnswerError, match="3번 시도"):
        agent.final_answer([HumanMessage("q")])


def test_orchestrator_loop_is_capped(client, monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()
    produced = []

    class Looping:
        def stream(self, inputs, config=None, **kw):
            assert config["recursion_limit"] == stream.AGENT_LIMIT
            for i in range(50):  # 같은 팀원에게 끝없이 맡기는 총괄
                produced.append(i)
                call = AIMessage("", tool_calls=[{"name": "task", "args": {"subagent_type": "minutes", "description": "정리"}, "id": f"t{i}"}])
                yield (), "updates", {"model": {"messages": [call]}}
                yield (), "updates", {"tools": {"messages": [ToolMessage(f"회의록 {i}", tool_call_id=f"t{i}", name="task")]}}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Looping())
    monkeypatch.setattr("rag.agent._model", lambda env: type("M", (), {"invoke": lambda self, m: AIMessage("## 지금까지 결과 정리")})())
    out = list(stream.ask_stream(None, "메모", None, agent_id=f"app:{app['id']}"))
    assert len(produced) == stream.TASK_CAP_PER_AGENT + 1  # 한도를 넘는 순간 멈춘다
    final = out[-1]
    assert final["type"] == "final" and final["answer"] == "## 지금까지 결과 정리"
    assert "반복 한도" in final["warnings"][0]


# ─────────────────────────── 처리 흐름 (답변 끝 'Agent 를 어떤 순서로 썼는지') ───────────────────────────


def _names(flow):
    return [[it["name"] for it in s["items"]] for s in flow]


def test_pipeline_final_has_flow_with_parallel_group(client, monkeypatch):
    app = client.post("/api/apps", json={**APP, "agents": ["translator", "sentiment", "keywords"]}).json()
    monkeypatch.setattr(stream, "get_agent", lambda agent_id: StepAgent(agent_id))
    flow = _run(app["id"], "기사")[-1]["flow"]
    from rag.market import BY_ID

    assert _names(flow) == [[BY_ID["translator"].name], [BY_ID["sentiment"].name, BY_ID["keywords"].name]]
    assert [s["parallel"] for s in flow] == [False, True]
    assert all(it["status"] == "ok" and it["sec"] is not None for s in flow for it in s["items"])


def test_auto_flow_records_delegations_in_order(client, monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()

    class Orchestrator:
        def stream(self, inputs, **kw):
            one = AIMessage("", tool_calls=[{"name": "task", "args": {"subagent_type": "minutes", "description": "정리"}, "id": "t1"}])
            yield (), "updates", {"model": {"messages": [one]}}
            yield ("tools:x",), "updates", {"model": {"messages": [AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "w"}])]}}
            yield (), "updates", {"tools": {"messages": [ToolMessage("회의록", tool_call_id="t1", name="task")]}}
            two = AIMessage("", tool_calls=[{"name": "task", "args": {"subagent_type": s, "description": "d"}, "id": f"t{i}"}
                                            for i, s in [(2, "action_items"), (3, "email")]])
            yield (), "updates", {"model": {"messages": [two]}}
            yield (), "updates", {"tools": {"messages": [ToolMessage("a", tool_call_id="t2", name="task"),
                                                          ToolMessage("Error: 실패", tool_call_id="t3", name="task")]}}
            yield (), "updates", {"model": {"messages": [AIMessage("최종 결과")]}}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Orchestrator())
    flow = _run(app["id"], "메모")[-1]["flow"]
    assert _names(flow)[1:] == [["회의록 정리"], ["할 일 추출", "이메일 작성"], ["최종 답 정리"]]
    assert flow[0]["items"][0]["name"].endswith("· 총괄")
    assert flow[1]["items"][0]["note"] == "웹 검색 1"  # 팀원이 쓴 도구
    assert flow[2]["parallel"] and flow[2]["items"][1]["status"] == "fail"


def test_fallback_flow_marks_orchestrator_and_continues_with_steps(client, monkeypatch):
    from langchain_core.messages import AIMessage

    app = client.post("/api/apps", json={**APP, "run": "auto"}).json()

    class Orchestrator:
        def stream(self, inputs, **kw):
            yield (), "updates", {"model": {"messages": [AIMessage("직접 답함")]}}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Orchestrator() if agent_id.startswith("app:") else StepAgent(agent_id))
    flow = _run(app["id"], "메모")[-1]["flow"]
    assert flow[0]["items"][0]["status"] == "fail" and "순서대로" in flow[0]["items"][0]["note"]
    assert _names(flow)[1:] == [["회의록 정리"], ["할 일 추출"], ["이메일 작성"]]
