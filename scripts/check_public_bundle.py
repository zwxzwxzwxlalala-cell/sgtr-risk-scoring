"""Fail-closed release allowlist and provenance checks; no network access."""
from pathlib import Path
import csv, hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'README.md', 'README.zh-CN.md', 'LICENSE', 'NOTICE.md', '.gitignore',
    'pyproject.toml', 'CONTRIBUTING.md', 'SECURITY.md',
    '.github/workflows/checks.yml',
    'src/sgtr_demo/__init__.py', 'src/sgtr_demo/core.py',
    'examples/synthetic_demo.py', 'tests/test_core.py', 'tests/test_research_equivalence.py',
    'scripts/check_public_bundle.py', 'scripts/render_results.py',
    'docs/method.md', 'docs/results.md', 'docs/reproducibility.md', 'docs/interview-guide.zh-CN.md',
    'research_reference/README.md',
    'research_reference/adaptive_tr_core.py', 'research_reference/adaptive_tr_s6c.py',
    'research_reference/adaptive_tr_experiment.py', 'research_reference/state_index_s2c.py',
    'research_reference/behavior_index.py', 'research_reference/sgtr_s6d.py',
    'research_reference/sgtr_s7.py', 'research_reference/dynamic_shap_s8.py',
    'results/final_period_metrics.csv', 'results/temporal_metrics.csv',
    'results/paired_bootstrap_summary.csv', 'results/development_to_holdout_comparison.csv',
    'results/state_layer_counts.csv', 'results/state_transportability.csv',
    'results/reliability_curve.csv', 'results/frozen_acceptance_result.json',
    'results/provenance.json', 'assets/effect_comparison.png', 'release_manifest.json',
}


def public_files():
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        # Build/test byproducts are never exported or included in the manifest.
        if any(x in {'.git','build','dist','__pycache__'} or x.endswith('.egg-info') for x in rel.parts):
            continue
        yield p, rel.as_posix()


def main():
    errors = []
    for path, name in public_files():
        if name not in ALLOWED:
            errors.append('Not allowlisted: '+name)
            continue
        raw = path.read_bytes()
        if len(raw) > 1_000_000:
            errors.append('Oversized: '+name)
        if path.suffix == '.png':
            continue
        text = raw.decode('utf-8-sig')
        patterns = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
                    r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----', r'(?i)\b[A-Z]:[\\/]']
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append('Sensitive token/path pattern: '+name)
        if path.suffix == '.csv':
            header = next(csv.reader(text.splitlines()))
            if set(header) & {'customer_ID','pair_id','row_index','row_prev','row_curr','target'}:
                errors.append('Individual-record schema: '+name)
    prov = json.loads((ROOT/'results/provenance.json').read_text(encoding='utf-8'))
    for item in prov['files']:
        p = ROOT/item['public_path']
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Provenance hash mismatch: '+item['public_path'])
    for name in ALLOWED - {'release_manifest.json'}:
        if not (ROOT/name).is_file():
            errors.append('Missing public file: '+name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {sum(1 for _ in public_files())} allowlisted public files; {len(prov["files"])} provenance hashes verified.')
    print('Scope: content guardrails only; rights review remains a maintainer responsibility.')


if __name__ == '__main__':
    main()
