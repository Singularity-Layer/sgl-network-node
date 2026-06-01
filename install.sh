#!/bin/sh
# SGL Node — Install Script
# Usage: curl -sSf https://grid.x402compute.cc/install.sh | sh
set -e

BINARY_NAME="sgl"                       # installed command + built binary name
ASSET_NAME="sgl-darwin-arm64"           # release asset filename
INSTALL_DIR="/usr/local/bin"
REPO="Singularity-Layer/sgl-network-node"
RELEASES_URL="https://github.com/${REPO}/releases"

main() {
    echo ""
    echo "  ╔═══════════════════════════════════════╗"
    echo "  ║       SGL Node — Install Script       ║"
    echo "  ║   Decentralized Confidential Compute  ║"
    echo "  ╚═══════════════════════════════════════╝"
    echo ""

    check_platform
    check_dependencies
    detect_latest_version
    download_binary
    verify_install
    print_next_steps
}

check_platform() {
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS" in
        Darwin) ;;
        *)
            echo "Error: sgl-node currently only supports macOS."
            echo "Linux and Windows support coming in Phase 2."
            exit 1
            ;;
    esac

    case "$ARCH" in
        arm64|aarch64) ;;
        *)
            echo "Error: sgl-node requires Apple Silicon (arm64)."
            echo "Intel Macs are not supported (no Secure Enclave)."
            exit 1
            ;;
    esac

    echo "  Platform: macOS arm64 ✓"

    # Check minimum RAM (16GB)
    TOTAL_MEM=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    TOTAL_GB=$((TOTAL_MEM / 1073741824))
    if [ "$TOTAL_GB" -lt 16 ]; then
        echo "  Warning: ${TOTAL_GB}GB RAM detected. 16GB+ recommended for inference."
    else
        echo "  Memory: ${TOTAL_GB}GB ✓"
    fi
}

check_dependencies() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required. Install it with: brew install curl"
        exit 1
    fi
    if ! command -v shasum >/dev/null 2>&1 && ! command -v sha256sum >/dev/null 2>&1; then
        echo "Error: shasum or sha256sum is required to verify the download."
        exit 1
    fi
}

detect_latest_version() {
    echo "  Checking latest version..."

    LATEST_VERSION=$(curl -sSf "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
        | grep '"tag_name"' \
        | head -1 \
        | sed 's/.*"tag_name": *"//;s/".*//' || echo "")

    if [ -z "$LATEST_VERSION" ]; then
        LATEST_VERSION="v0.1.0"
        echo "  Using default version: ${LATEST_VERSION}"
    else
        echo "  Latest version: ${LATEST_VERSION}"
    fi
}

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

download_binary() {
    DOWNLOAD_URL="${RELEASES_URL}/download/${LATEST_VERSION}/${ASSET_NAME}"
    CHECKSUM_URL="${DOWNLOAD_URL}.sha256"
    TEMP_FILE=$(mktemp)
    TEMP_SUM=$(mktemp)

    echo "  Downloading ${ASSET_NAME}..."

    HTTP_CODE=$(curl -sSL -w "%{http_code}" -o "$TEMP_FILE" "$DOWNLOAD_URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" != "200" ]; then
        echo ""
        echo "  GitHub release not found (this is expected during development)."
        echo ""
        echo "  To install from source instead:"
        echo ""
        echo "    git clone https://github.com/${REPO}.git"
        echo "    cd sgl-network-node"
        echo "    cargo build --release"
        echo "    sudo cp target/release/${BINARY_NAME} ${INSTALL_DIR}/"
        echo ""
        rm -f "$TEMP_FILE" "$TEMP_SUM"
        exit 0
    fi

    # Verify the published checksum. Fail closed — never install an unverified
    # binary. The orchestrator additionally gates on a binary-hash allowlist, but
    # we refuse to install tampered bits in the first place.
    SUM_CODE=$(curl -sSL -w "%{http_code}" -o "$TEMP_SUM" "$CHECKSUM_URL" 2>/dev/null || echo "000")
    if [ "$SUM_CODE" != "200" ]; then
        echo "  Error: checksum file not found at ${CHECKSUM_URL}."
        echo "  Refusing to install an unverified binary."
        rm -f "$TEMP_FILE" "$TEMP_SUM"
        exit 1
    fi

    EXPECTED=$(awk '{print $1}' "$TEMP_SUM")
    ACTUAL=$(sha256_of "$TEMP_FILE")
    if [ -z "$EXPECTED" ] || [ "$EXPECTED" != "$ACTUAL" ]; then
        echo "  Error: checksum mismatch — refusing to install."
        echo "    expected: ${EXPECTED}"
        echo "    actual:   ${ACTUAL}"
        rm -f "$TEMP_FILE" "$TEMP_SUM"
        exit 1
    fi
    echo "  Checksum verified ✓"

    chmod +x "$TEMP_FILE"

    if [ -w "$INSTALL_DIR" ]; then
        mv "$TEMP_FILE" "${INSTALL_DIR}/${BINARY_NAME}"
    else
        echo "  Installing to ${INSTALL_DIR} (requires sudo)..."
        sudo mv "$TEMP_FILE" "${INSTALL_DIR}/${BINARY_NAME}"
    fi
    rm -f "$TEMP_SUM"

    echo "  Installed: ${INSTALL_DIR}/${BINARY_NAME} ✓"
}

verify_install() {
    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
        VERSION=$("$BINARY_NAME" --version 2>/dev/null || echo "installed")
        echo "  Verified: ${VERSION} ✓"
    else
        echo "  Warning: ${BINARY_NAME} not found in PATH."
        echo "  You may need to add ${INSTALL_DIR} to your PATH."
    fi
}

print_next_steps() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │                 Next Steps                   │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""
    echo "  1. Detect your hardware:"
    echo "     $ sgl detect"
    echo ""
    echo "  2. Log in and register this node (browser):"
    echo "     $ sgl login"
    echo ""
    echo "  3. Download a model (e.g., Llama 3.2 3B):"
    echo "     $ curl -L -o model.gguf <model-download-url>"
    echo ""
    echo "  4. Start earning:"
    echo "     $ sgl start --model-path ./model.gguf --model-name llama-3.2-3b"
    echo ""
    echo "  Docs: https://sgl.network/docs"
    echo "  Discord: https://discord.gg/singularity"
    echo ""
}

main
