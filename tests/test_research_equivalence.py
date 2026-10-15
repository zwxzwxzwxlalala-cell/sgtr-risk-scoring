"""Execute only two inspected function definitions, never research entry points."""
import ast
from pathlib import Path
import unittest
import numpy as np
from sgtr_demo import derivatives, stable_gate

ROOT = Path(__file__).resolve().parents[1]


def isolated_function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = {'np': np}
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return namespace[name]


class ResearchEquivalence(unittest.TestCase):
    def test_hard_gate_matches_research(self):
        original = isolated_function(ROOT/'research_reference/adaptive_tr_s6c.py', 'gate_weight')
        g = np.array([0, .1, .25, .5, .9, 1, np.nan])
        np.testing.assert_array_equal(stable_gate(g), original(g, 'hard'))

    def test_objective_matches_research_on_synthetic_margins(self):
        original = isolated_function(ROOT/'research_reference/adaptive_tr_core.py', 'make_objective')
        rng = np.random.default_rng(42)
        n = 8
        f, y = rng.normal(size=n), np.repeat([0., 1.], 4)
        edges = np.array([[0,1],[1,2],[2,3],[4,5],[5,6],[6,7]])
        w = np.array([1.,0.,1.,0.,1.,0.])
        pp, pn = np.full(n, -1), np.full(n, -1)
        wp, wn = np.zeros(n), np.zeros(n)
        for (a,b), weight in zip(edges,w):
            pp[b], pn[a], wp[b], wn[a] = a, b, weight, weight
        class Dataset:
            def get_label(self):
                return y
        for lam in [0., .03]:
            expected = original(pp,pn,wp,wn,lam)(f.copy(), Dataset())
            actual = derivatives(f,y,edges,w,lam)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,rtol=0,atol=3e-16)
