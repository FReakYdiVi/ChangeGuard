# Final Audit (Hackathon Submission)

## Strongest Parts
- **Environment novelty is strong and clear**: tenant-aware schema migration safety (not generic incident response).
- **Objective verification is explicit**: deterministic verifier verdicts (`safe_finalized`, `approval_violation`, `compatibility_violation`, `timeout_failure`, etc.).
- **Anti-hacking hardening exists and is tested**: inspect-loop penalties, timeout anti-farming, hidden-state non-leakage, invalid-action penalties with reasons.
- **Curriculum path is practical**: `easy_stable -> medium_mixed -> hard_fragile` with fixed seed packs.
- **Training visibility is good**: per-episode metrics + reward components + fixed-seed baseline-vs-candidate eval script.
- **Local reproducibility is good**: deterministic seeds, test suite, Dockerfile, OpenEnv manifest, one-command pipeline script.

## Weakest Parts
- **True GRPO training path is dependency-sensitive**: `trl`/`datasets` must be installed; default path is dry-run.
- **Candidate policy in eval is scripted, not a learned checkpoint yet**: currently demonstrates environment quality and expected improvement direction.
- **OpenEnv validate command assumes CLI availability**: documented, but not enforced in CI.
- **HF Spaces is Docker-ready, but not yet proven with a public Space URL in repo docs.**

## What to Demo Live in 3 Minutes
1. **Problem + stakes (30s)**
- “We model tenant-safe schema upgrades where hidden legacy dependencies can break enterprise tenants.”

2. **Environment + verifier (60s)**
- Show one unsafe path (A->B->C finalize without approval/compat) producing objective violation.
- Show one safe path (inspect -> A -> B -> compat -> approval -> C finalize) producing safe completion.

3. **Metrics + fixed-seed comparison (60s)**
- Run baseline vs candidate on `final_demo` seed pack.
- Show headline deltas: safe full completion up, unsafe completion and approval violation down.

4. **Anti-hacking credibility (30s)**
- Mention tested protections: hidden state never exposed, inspect spam penalized, timeout cannot increase reward.

## What to Cut if Time Is Short
- Cut hard-difficulty live demo and keep only easy+medium.
- Cut non-essential reward-component discussion; keep 3 headline metrics.
- Cut full training run live; show dry-run + fixed-seed evaluation only.
- Cut older `tenantsafe/` legacy paths from verbal explanation to avoid narrative drift.

## Before/After Metrics Table to Show
Use output of:
`python -m training.evaluate_policy --seed-pack final_demo --candidate trained_like`

| Metric | Baseline (before) | Candidate/Trained (after) | Direction |
|---|---:|---:|---|
| safe_full_completion | 0.00 | 1.00 | higher is better |
| safe_partial_completion | 0.00 | 0.00 (or small) | context-dependent |
| unsafe_completion | 1.00 | 0.00 | lower is better |
| approval_violation | 1.00 | 0.00 | lower is better |
| data_integrity_violation | 1.00 | 0.00 | lower is better |
| blast_radius | 2.0-3.0 | 3.0 with safety gates satisfied | explain with context |
| invalid_action_count | low/medium | lower | lower is better |
| mean_steps | very low (rushed fail) | moderate (safe sequence) | contextual |
| reward_total | negative | positive | higher is better |
| progress_reward | lower | higher | higher |
| inspection_reward | low | moderate | contextual |
| safety_reward | negative/low | positive | higher |
| invalid_action_penalty | more negative | less negative | less negative |
| loop_penalty | similar or worse | better | less negative |
| terminal_bonus_or_penalty | strongly negative | strongly positive | higher |

Note: Values shown above are the expected direction profile from current policy definitions. Replace with exact numbers from your local run output for final slides.

## 3 Likely Judge Questions (and Answers)
1. **Q: How do you know the agent isn’t reward-hacking?**
- **A:** We explicitly test exploit patterns: hidden-state leakage, inspect-spam, timeout farming, unsafe finalize shortcuts. Verifier + penalties ensure these strategies are non-profitable.

2. **Q: Why is this different from generic SRE incident environments?**
- **A:** This is proactive change governance: staged tenant upgrades, approval gates, compatibility handling before irreversible migration. It optimizes safe evolution, not just post-failure recovery.

3. **Q: Is this reproducible and benchmarkable?**
- **A:** Yes. Deterministic seeds, fixed seed packs (`smoke`, `short_train`, `final_demo`), objective verifier verdicts, and baseline-vs-candidate scripts provide reproducible comparisons.

## Verdict
Submission quality is strong for a hackathon if you focus the demo on:
- objective verifier,
- fixed-seed before/after metrics,
- anti-hacking checks,
- clear tenant-safety business value.
