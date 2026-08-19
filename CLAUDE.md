# sgl-node — agent working rules

Rust CLI that turns a machine into a Singularity Layer grid node: runs local
inference (llama.cpp), reports capacity/health to the orchestrator, and **earns real
money per request**. Its own repo (`Singularity-Layer/sgl-network-node`, **public**) —
commit from inside this directory, never from the x402Studio root.

Because operators are paid from what this binary reports, **most bugs here are money
bugs**. Treat them that way.

---

## 1. Release blockers (non-negotiable)

A finding in any of these is a **NO-GO**, never a "follow-up":

| Area | The invariant |
|---|---|
| **Billing correctness** | Tokens billed == tokens actually produced. Over-report steals from users; under-report steals from the operator. |
| **Stream settlement** | A stream that fails, stalls, or is cancelled **must not settle as success**. Partial output bills only what was delivered. |
| **Empty completions** | An engine returning empty output must never bill as a completed job. It must trip self-heal, not silently earn. |
| **Setup / install writes** | Atomic and fail-closed. A partial extraction must never leave a half-installed engine that looks healthy. |
| **Key handling** | Keys and signing paths never log, never leak into telemetry, never weaken across versions. |

If you cannot prove the invariant holds, the answer is NO-GO. "Looks fine" is not proof.

## 2. Evidence standard

Every GO / NO-GO / finding cites **`file.rs:line`**. No claim without a location.
- Report what you *verified*, not what you assume. Say "did not verify" out loud.
- No style or taste findings unless they change production behavior.
- Distinguish **CONFIRMED** (traced the code path) from **PLAUSIBLE** (pattern-matched).

## 3. Every finding gets a recorded disposition

The failure mode this repo has had: a real finding gets discussed, then the session
ends with no record of what happened to it.

**Before ending any audit/review session**, write each finding into
`docs/audit/FINDINGS.md` with one of: `FIXED` (+ commit), `ACCEPTED-RISK` (+ why +
who), `REJECTED` (+ why), `DEFERRED` (+ tracking issue). A finding with no row is an
unfinished session. Same rule for a NO-GO: record the blocker, the fix owner, and the
command that will verify it.

## 4. Before any release

```bash
./scripts/release-check.sh        # fmt + clippy -D warnings + tests + audit
```

Never tag or publish a binary from a tree that is dirty or ahead of `origin`.
Order is always **commit → push → then release**. Binaries are hash-allowlisted by the
orchestrator, so a rebuilt binary means the allowlist must be updated in the same
change or nodes silently stop serving.

## 5. Tests are the safety net — not the reviewer

AI review is a second pair of eyes, **not** the gate. Any behavior we have manually
audited more than once must become a test. Required coverage, and the reason each one
exists (all are real historical failure shapes):

- **Billing:** token accounting on normal, truncated, and failed requests.
- **Streaming settlement:** Delta→Done accounting; failure mid-stream does not bill success.
- **Empty-completion quarantine:** restart budget is bounded; a zombie engine cannot loop forever.
- **Edge inputs:** `max_tokens=1`, immediate EOS, tool-call-only output, user-initiated stop, whitespace-only stream.
- **Setup:** rollback on failure, partial-extraction recovery.
- **Vision:** `mmproj_path` passthrough reaches the engine (in-process vs server mode differ — cover both).
- **Keybind/signing:** fixed test vectors, so signing compatibility can never drift silently.

New behavior in a money path ships **with** its test, in the same commit.

## 6. Keep the hot files from growing

`src/node.rs` (~2.4k lines) carries lifecycle, health, restart, capability and billing
concerns at once; `src/inference.rs` and `src/inprocess.rs` are the next densest. This
is the main reason bugs here are hard to see.

**Rule:** if you touch one of these files, leave it no larger than you found it. When a
change would add meaningful size, extract the seam you are working in first, as its own
commit, with tests. Preferred seams: empty-health state, streaming settlement, restart
policy, billing accounting, setup promotion/rollback.

Do not attempt a big-bang refactor. One seam, one commit, tests passing.

## 7. Commits

Current prefixes are good — keep them: `feat(scope):`, `fix(scope):`, `test:`,
`security:`, `refactor:`, `build:`, `release: vX.Y.Z — summary`.
State the *behavior* change, not the diff. Money-path commits say what is now billed
differently.

## 8. Docs that must stay true

`docs/` holds short design notes for: inference engine modes, restart/self-heal policy,
billing semantics, streaming settlement, setup/install safety, crypto/keybind model.
If you change one of those behaviors, update its note in the same commit — a stale
safety doc is worse than none.

## 9. Varying review angles

Repeating one broad prompt finds the same things. Rotate the lens instead:
tests-only · exploit-path-only · concurrency-only · billing-only ·
diff-against-the-prior-finding-list · what-regression-test-is-missing.
