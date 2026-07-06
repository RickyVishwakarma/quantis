from quantis.ai.copilot import _local_answer, ask, build_context


def test_build_context_shape():
    ctx = build_context(lake_root="nonexistent/lake", runs_root="nonexistent",
                        registry_root="nonexistent", paper_root="nonexistent")
    assert "lake" in ctx
    assert "strategies" in ctx
    assert "ai_signal" in ctx["strategies"]


def test_local_fallback_answers_model_questions():
    ctx = {"model_registry": [{"name": "ridge_fwd5d", "version": 1,
                               "stage": "SHADOW", "metrics": {"ic": 0.03}}]}
    answer = _local_answer("what models are in shadow?", ctx)
    assert "ridge_fwd5d" in answer and "SHADOW" in answer


def test_ask_without_llm_never_breaks():
    result = ask("summarize the platform", context={"lake": {"n_symbols": 5}},
                 use_llm=False)
    assert result["backend"] == "local"
    assert result["answer"]
