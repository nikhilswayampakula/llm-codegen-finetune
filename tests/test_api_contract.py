from app.api import GenReq

def test_req_defaults():
    r = GenReq(prompt="def f():\n    pass")
    assert r.temperature == 0.2 and r.top_p == 0.95
