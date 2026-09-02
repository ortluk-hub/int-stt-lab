# Hand-off from Nemo to Morgan (test)

**Purpose:** Verify that the hand‑off mechanism and bot‑to‑bot communication work as intended.

**Summary:**
- Nemo has prepared a minimal hand‑off payload.
- The payload includes a short description, a list of changed files, and a placeholder verification step.

**Changed files (simulated):**
- `src/example.py`
- `docs/README.md`

**Validation steps:**
1. Verify that all listed files exist.
2. Confirm that each file contains the expected marker `/* HANDOFF */`.
3. Respond with an acknowledgment comment.

**Next action (Morgan):**
- Review the above items and reply with `ACK` if the hand‑off is acceptable, or raise any concerns.

---
*This is a test artifact; no real code changes are included.*