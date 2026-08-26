# Audit findings ledger

Every review finding gets a row here with a **recorded disposition**. A finding that
was discussed but never dispositioned is an unfinished session.

**Disposition values:** `FIXED` (+ commit) · `ACCEPTED-RISK` (+ why, + who accepted) ·
`REJECTED` (+ why) · `DEFERRED` (+ where it's tracked) · `OPEN` (still undecided).

Newest first. Keep rows short; link the commit for detail.

| Date | Finding | Where | Severity | Disposition | Evidence / commit |
|---|---|---|---|---|---|
| 2026-08-19 | CI never runs the test suite — workflows only `cargo build`, so all 25 tests could break unnoticed | `.github/workflows/release.yml`, `windows.yml` | High | `FIXED` | `.github/workflows/ci.yml` — `cargo test` now blocks on every push/PR (verified 25/25 green) |
| 2026-08-19 | `cargo fmt` never run: 208 formatting diffs, so a fmt gate cannot be enabled | repo-wide | Low | `DEFERRED` | Needs a one-time `cargo fmt --all` in its own commit (no behaviour change, large diff — do it when no branch is mid-flight), then flip the advisory gate to blocking in `ci.yml` + `release-check.sh` |
| 2026-08-19 | 9 outstanding clippy lints block a `-D warnings` gate: 4× unneeded `return`, 2× dead code (`stop`, `is_embedding_model`), 1× loop-index, 2× too-many-arguments (**one function takes 15 args** — corroborates the overloaded-hot-file finding) | `src/` (bin `sgl`) | Low (cosmetic) | `DEFERRED` | Clear with `cargo clippy --all-targets --fix`, then hand-fix the arity ones; then make clippy blocking. Verify: `cargo clippy --all-targets -- -D warnings` |
| 2026-08-19 | `items_after_test_module` blocked clippy on the lib target | `src/inference.rs:1153` | Low | `FIXED` | Narrow `#[allow]` + comment; reordering a 1.2k-line file was not worth the history churn |
| 2026-08-19 | Release gates lived only in prompts, not in the repo | repo root | Medium | `FIXED` | `scripts/release-check.sh` + `make release-check` |
| 2026-08-19 | No repo-local agent rules, so money-path invariants were re-derived each session | repo root | Medium | `FIXED` | `CLAUDE.md` |
| 2026-08-19 | Findings had no durable accept/reject/fix record | process | Medium | `FIXED` | this file |
| 2026-08-19 | Hot files carry too many concerns (`node.rs` ~2.4k lines = 26% of the codebase; lifecycle + health + restart + capability + billing) | `src/node.rs`, `src/inference.rs`, `src/inprocess.rs` | Medium | `DEFERRED` | Seam-at-a-time extraction; rule recorded in `CLAUDE.md` §6. Not a big-bang refactor. |
| 2026-08-19 | Regression tests missing for repeatedly hand-audited behaviors (billing, stream settlement, empty-completion quarantine + restart budget, `max_tokens=1`, immediate EOS, tool-call-only, user stop, whitespace-only stream, setup rollback / partial extraction, `mmproj_path` vision passthrough, keybind vectors) | `src/inference.rs` (1 test), `src/inprocess.rs` (0), `src/setup.rs` (3) | **High** | `OPEN` | Required list recorded in `CLAUDE.md` §5. Highest-value next work. |
| 2026-08-19 | Keybind signing compatibility risk reviewed, no final disposition recorded | `src/crypto.rs` | Unknown | `OPEN` | Needs a decision + fixed test vectors |
| 2026-08-19 | VISION in-process vs server-mode risk raised, no recorded follow-up | `src/inprocess.rs`, `src/inference.rs` | Unknown | `OPEN` | Node forces server engine when mmproj is set — needs a test asserting that |
| 2026-08-19 | Setup partial-extraction risk found, closure not recorded | `src/setup.rs` | Unknown | `OPEN` | Needs rollback test |
| 2026-08-19 | Startup-hang diagnosis ended in recommendations, no decision recorded | `src/node.rs`, `src/service.rs` | Unknown | `OPEN` | Decide: fix, accept, or reject |
| 2026-08-26 | `post_chunk` accepted a `fmt` parameter but never wrote it into the request body, so every tool-enabled stream chunk reached the orchestrator untagged. The reader correctly read "untagged" as legacy raw text, terminalized the stream (`node ignored tools on stream`) and fell back to a buffered completion — tool-call deltas were generated then discarded on every request. Billing was never wrong: the failed stream settled $0.00 and the fallback billed exactly once. | `src/orchestrator.rs:684` | **High** (silently dead feature, not a money bug) | `FIXED` | Fixed in `90422cf`, shipped in `v1.9.7`. Live-verified on mainnet: 9 SSE frames / 8 incremental tool-call deltas / `finish_reason=tool_calls`; pre-v1.9.7 nodes still fall back to buffered tool calls unchanged. Root cause was a patch script whose assertion aborted mid-edit, so the signature landed without the body write — the fail-closed reader is what surfaced it. |

## The four `OPEN` audit-loop items

These are open because the *transcript shows analysis but no decision*. Each needs one
line added above saying FIXED / ACCEPTED-RISK / REJECTED, with `file.rs:line` evidence.
Leaving them as "we looked at it once" is exactly the gap this ledger exists to close.
