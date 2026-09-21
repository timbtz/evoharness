from evoharness.tasks.stellar_p2.workflows.discover import (CONTINUE_PROMPT, MODEL, _generate,
                                                      build_prompt, memory_markdown,
                                                      parse_memory,
                                                      response_requests_stop)


def test_pipeline_is_fixed_to_one_model_and_starts_without_inherited_candidate():
    state = {"attempts": [], "best_candidate_hash": None,
             "best_selection_score": None}
    prompt = build_prompt("task contract", state, None, 1)
    assert MODEL == "glm-5.2"
    assert "null point" in prompt
    assert "no candidate yet" in prompt.lower()
    assert "public seed bank is technically disabled" in prompt
    assert "web_search at least once" in prompt


def test_resumed_prompt_improves_only_its_own_incumbent():
    state = {"attempts": [], "best_candidate_hash": "abc",
             "best_selection_score": -0.2}
    prompt = build_prompt("task", state, "def solve(fm, rng):\n    return {}", 2)
    assert "created by this same campaign, not inherited" in prompt
    assert "def solve" in prompt


def test_stop_or_missing_code_triggers_confidence_continuation():
    assert response_requests_stop("I should stop now")
    assert response_requests_stop("Idea: only a plan")
    assert not response_requests_stop(
        "Idea: early stopping for bad inner trials\n```python\ndef solve(fm, rng):\n return {}\n```")
    assert not response_requests_stop(
        "Idea: x\nPrediction: y\nMemory: z\n```python\ndef solve(fm, rng):\n return {}\n```")
    assert "Believe" in CONTINUE_PROMPT
    assert "other research areas" in CONTINUE_PROMPT
    assert "web_search" in CONTINUE_PROMPT


def test_explicit_stop_with_code_gets_an_encouragement_retry():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def chat(self, model, messages, **kwargs):
            self.calls.append((model, messages, kwargs))
            if len(self.calls) == 1:
                return ("Idea: first\nPrediction: measurable\nMemory: persist\n"
                        "I should stop now.\n```python\n"
                        "def solve(fm, rng):\n    return fm.seed_ellipse(8, 2, .6, 2)\n```")
            return ("Idea: cross-domain retry\nPrediction: measurable\nMemory: continue\n"
                    "```python\ndef solve(fm, rng):\n"
                    "    return fm.seed_ellipse(9, 2, .6, 2)\n```")

    llm = FakeLLM()
    code, metadata = _generate(llm, "task", {"attempts": []}, None, 1, set())
    assert "seed_ellipse(9" in code
    assert metadata["generation_attempts"] == 2
    assert CONTINUE_PROMPT.splitlines()[0] in llm.calls[1][1][-1]["content"]


def test_memory_is_durable_and_contains_measured_outcomes():
    response = "Memory: Mirror-first repair was falsified.\n```python\nx=1\n```"
    assert parse_memory(response) == "Mirror-first repair was falsified."
    state = {"best_candidate_hash": "h1", "attempts": [{
        "round": 1, "status": "evaluated", "accepted": True,
        "candidate_hash": "h1", "selection_score": -0.1,
        "idea": "new basin", "memory": "keep this", "metrics": {
            "objective_L": 9.0, "feasibility": .1, "active_violation": "qi"}}]}
    text = memory_markdown(state)
    assert "Current incumbent" in text
    assert '"active_violation": "qi"' in text
    assert "keep this" in text
