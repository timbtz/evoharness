"""Is the qi gradient wrong, or is 3e-4 already outside its trust region?

The FD probe's descent predicted qi DOWN at every cap (3e-4, 1e-3, 3e-3) on two
boundaries and qi went UP every time, |actual| ~ |predicted|. Two candidate
explanations with opposite consequences for Plan 3: the gradient is wrong, or
the steps are outside the range where a linear model means anything.

Ruled out first: the estimator. On a step-size-converged column the probe's
contracted gradient matched a direct FD of the ORACLE's qi to 0.02%
(+2.23112 vs +2.23071). The gradient is right.

So this sweeps the step size down instead, reusing the cached gradient (no new
FD columns, 4 solves total). Result:

    cap     predicted    actual      ratio
    1e-6    -0.000473   -0.000394    +0.83
    1e-5    -0.004735   -0.004196    +0.89
    3e-5    -0.014204   -0.014071    +0.99
    1e-4    -0.047345   +0.002195    -0.05   <- breaks

d(qi)/d(boundary) is predictive up to ~3e-5 in max-coefficient and useless by
1e-4. The probe's original smallest cap was 10x outside its own trust region.

This lands on the SAME boundary that the objective study found independently
(min L-gradB is smooth at 1e-5 and branch-switching at 1e-4). Two unrelated
measurements, one trust region: ~3e-5.
"""
import json, sys, numpy as np
sys.path.insert(0,'/work')
from experiments.fd_qi_probe import load_cases
from constellaration import forward_model as fmod, problems
from constellaration.geometry import surface_rz_fourier as srf
from constellaration.mhd import vmec_settings as vs

def score(bd):
    b=srf.SurfaceRZFourier.model_validate(bd)
    m,_=fmod.forward_model(b, settings=fmod.ConstellarationSettings(
        vmec_preset_settings=vs.VmecPresetSettings(fidelity="very_low_fidelity"),
        turbulent_settings=None))
    p2=problems.SimpleToBuildQIStellarator()
    return float(m.qi), float(p2.compute_feasibility(m)), float(
        m.minimum_normalized_magnetic_gradient_scale_length)

g=json.load(open('/work/runs/fd_probe/champion/gradient.json'))
b0=load_cases(['champion'])[0]['boundary']
gq=np.concatenate([np.asarray(g['grad_r_cos']).ravel(), np.asarray(g['grad_z_sin']).ravel()])
ga=np.concatenate([np.asarray(g['aspect_grad_r_cos']).ravel(), np.asarray(g['aspect_grad_z_sin']).ravel()])
gq = gq - ga*float(gq@ga)/float(ga@ga)
d = -gq/np.abs(gq).max()
rc=np.asarray(b0['r_cos'],float); zs=np.asarray(b0['z_sin'],float); n=rc.size
base=g['base']['log10_qi_jax']
print("base log10(qi)=%.6f feas=%.5f"%(base, g['base']['feasibility']), flush=True)
out=[]
for cap in (1e-6, 1e-5, 3e-5, 1e-4):
    st=d*cap
    b=json.loads(json.dumps(b0))
    b['r_cos']=(rc+st[:n].reshape(rc.shape)).tolist()
    b['z_sin']=(zs+st[n:].reshape(zs.shape)).tolist()
    pred=float(gq@st)
    try:
        qi,feas,L=score(b)
    except Exception as e:
        print("  cap=%-8g SOLVER FAIL %s"%(cap,str(e)[:50]), flush=True); continue
    act=float(np.log10(qi))-base
    out.append({"cap":cap,"pred":pred,"actual":act,"feas":feas,"L":L})
    print("  cap=%-8g pred=%+.6f actual=%+.6f ratio=%+.2f feas=%.5f L=%.4f"%(
        cap,pred,act,act/pred,feas,L), flush=True)
json.dump(out, open('/work/runs/fd_probe/champion/small_caps.json','w'), indent=2)
