.PHONY: release-check check fmt lint test audit help

help:
	@echo "sgl-node"
	@echo "  make release-check  full gate before shipping a binary (fmt, clippy, test, audit, tree hygiene)"
	@echo "  make check          same gate, minus the network-dependent cargo audit"
	@echo "  make fmt            format in place"
	@echo "  make lint           clippy with warnings as errors"
	@echo "  make test           test suite"
	@echo "  make audit          known-vulnerable dependency scan"

# The one command to run before any release. See scripts/release-check.sh.
release-check:
	@./scripts/release-check.sh

check:
	@./scripts/release-check.sh --fast

fmt:
	cargo fmt --all

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-features

audit:
	cargo audit
