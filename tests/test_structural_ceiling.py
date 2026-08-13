import json
from experiments.structural_ceiling import build

def test_ceiling_is_scoped_to_fresh_state(tmp_path):
    d=tmp_path/'x'; d.mkdir()
    (d/'report.json').write_text('{}')
    (d/'state.json').write_text(json.dumps({'jobs':{'a':{'status':'evaluated','operator':{'family':'ellipse'},'metrics':{'objective_L':9,'honest_score':-.7,'feasibility':.7}}}}))
    out=build([d/'report.json'])
    assert out['families']['ellipse']['best_L']==9.0
    assert out['official_candidates']==0
