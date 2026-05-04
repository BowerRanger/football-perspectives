# Implementation decisions log

Auto-mode execution decisions for the broadcast-mono pipeline rewrite. Each entry: question that arose, decision taken, reasoning. User to review.

## 2026-05-05

### D1: Pre-implementation working tree handling

**Question:** Working tree on `main` had ~56 modified/untracked files (mid-flight pipeline work, partial GVHMR scaffolding, partial new schemas). How to fold those into the rewrite branch?

**Decision:** Created feature branch `feat/broadcast-mono-pipeline` from current dirty state, then committed everything (excluding `.pt` model files, `.venv311/`, `.DS_Store`, `test-media/` which are now gitignored) as a single baseline commit `c0596ab` titled "chore: snapshot pre-implementation working tree". Phase 0 then deletes the legacy files in a separate commit, so the deletions show cleanly in git log.

**Reasoning:** Preserves the user's in-flight work in git history (recoverable via `git show c0596ab` if any of it is needed later). Keeps legacy deletions semantically tagged in their own commits. Avoids stashing (which would make the user's work harder to recover post-merge).

### D2: Subagent batching granularity

**Question:** Plan has 44 tasks. Dispatching one subagent per task = 132 invocations (impl + spec review + quality review × 3). Excessive cost and time.

**Decision:** Batched the 44 plan tasks into 12 logical implementation chunks, each handled by one subagent dispatch with two-stage review:

1. Phase 0 — Cleanup
2. Phase 1.1–1.4 — Camera schemas + anchor solver
3. Phase 1.5–1.7 — Propagator + smoother + confidence
4. Phase 1.8–1.11 — Camera stage + synthetic fixture + integration
5. Phase 2.1–2.4 — HMR World schemas + transforms + smoothing
6. Phase 2.5–2.6 — GVHMR runner + HmrWorldStage
7. Phase 3 — Ball stage (all)
8. Phase 4.1–4.3 — Web server + anchor editor + viewer cleanup
9. Phase 4.4–4.6 — Export + quality report + web API tests
10. Phase 5 — Documentation
11. Phase 6 — Runner test + E2E + final QA
12. (reserved for any rework discovered in review)

**Reasoning:** Each chunk is self-contained, has cohesive state. Subagent cost per chunk stays bounded. Code-quality review still happens per chunk. If a chunk reveals issues, I can dispatch a focused fix subagent.

### D3: Phase 0 — bundle_adjust.py emptying

**Question:** Plan Task 0.9 said "drop functions only used by deleted stages; keep parabolic LM and seed helpers" in `src/utils/bundle_adjust.py`. The Phase 0 implementer found bundle_adjust.py was 100% cross-view player-correspondence code (no parabolic LM), so emptied it to a stub.

**Decision:** Accepted. Verified via `git show c0596ab:src/utils/ball_reconstruction.py` that the parabolic LM (`_fit_parabola_segment`, `_detect_flight_segments`, `_GRAVITY`, etc.) lived in `ball_reconstruction.py`, which Phase 0 deleted. Phase 3 (Task 3.3) provides full code for `fit_parabola_to_image_observations` to be written fresh into `bundle_adjust.py` — no information lost.

**Reasoning:** Plan was slightly inaccurate about where the parabolic code lived. Phase 3's plan code is self-contained and reconstructs the parabolic fit cleanly. Mentioning here so the user knows to verify Phase 3 brings the flight code back.

### D4: Phase 0 — tests with lazy imports of deleted stages

**Question:** `tests/test_manifest_inference.py` and `tests/test_tracking.py` import deleted stages inside test bodies (so collection passes but tests will runtime-fail). Skip-mark or leave?

**Decision:** Left collectable; affected tests will fail at runtime if executed. They cover behaviour that Phase 1c (camera stage rewire) and Phase 6 (test_runner rewrite) will rebuild. No automated test gate runs pytest at this stage.

**Reasoning:** Phase 0 is mechanical cleanup; broad test rewrites are scheduled. Adding speculative skip-marks would mask tests that may pass once Phase 1c lands. Phase 6 will sweep the test suite once stages are in place.

### D5: Subagent-driven review depth

**Question:** Spec compliance + code-quality review per task = 132 reviewer dispatches. Combine?

**Decision:** For Phase 0 (mechanical deletions) I'm running a single focused spec compliance review against the plan's Phase 0 task list. Code quality review is skipped for Phase 0 since there's almost no new code (just deletions + the already-templated runner.py / config.yaml / recon.py). For algorithmic phases (1–4) I'll run both stages of review per the skill.

**Reasoning:** Mechanical deletions don't benefit from code-quality review. Two-stage review pays off where there's algorithmic substance.

### D3: Constraints directory not in .gitignore

**Question:** Initial .gitignore update added `constraints/` but the README references `constraints/macos-py311-openmmlab.txt` for installs.

**Decision:** Removed `constraints/` from .gitignore. Kept `.venv311/`, `.DS_Store`, `*.pt`, `test-media/` as ignores.

**Reasoning:** Install instructions need that file. Not really a generated/transient directory.
