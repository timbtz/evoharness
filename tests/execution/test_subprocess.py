from __future__ import annotations






from evoharness.execution.sandbox import run_python

from evoharness.tasks.common import run_harness


def test_sandbox_timeout_and_network():
    r = run_python("while True: pass", timeout=2)
    assert r.timed_out and r.seconds < 10
    res = run_harness("while True: pass", timeout=2)
    assert res.score == float("-inf") and "timeout" in res.error

    net = ("import json, urllib.request\n"
           "urllib.request.urlopen('http://example.com', timeout=5)\n"
           "print(json.dumps({'score': 1.0}))\n")
    res = run_harness(net, timeout=15)
    assert res.score == float("-inf")
