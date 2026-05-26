#!/bin/sh
# SGL Node — Install Script
# Usage: curl -sSf https://grid.x402compute.cc/install.sh | sh
set -e

BINARY_NAME="sgl-node"
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
}

detect_latest_version() {
    echo "  Checking latest version..."

    # Try GitHub API first
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

download_binary() {
    DOWNLOAD_URL="${RELEASES_URL}/download/${LATEST_VERSION}/${BINARY_NAME}-darwin-arm64"
    TEMP_FILE=$(mktemp)

    echo "  Downloading ${BINARY_NAME}..."

    HTTP_CODE=$(curl -sSL -w "%{http_code}" -o "$TEMP_FILE" "$DOWNLOAD_URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" != "200" ]; then
        echo ""
        echo "  GitHub release not found (this is expected during development)."
        echo ""
        echo "  To install from source instead:"
        echo ""
        echo "    git clone https://github.com/${REPO}.git"
        echo "    cd sgl-node"
        echo "    cargo build --release"
        echo "    sudo cp target/release/sgl-node /usr/local/bin/"
        echo ""
        rm -f "$TEMP_FILE"
        exit 0
    fi

    chmod +x "$TEMP_FILE"

    if [ -w "$INSTALL_DIR" ]; then
        mv "$TEMP_FILE" "${INSTALL_DIR}/${BINARY_NAME}"
    else
        echo "  Installing to ${INSTALL_DIR} (requires sudo)..."
        sudo mv "$TEMP_FILE" "${INSTALL_DIR}/${BINARY_NAME}"
    fi

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
    echo "     $ sgl-node detect"
    echo ""
    echo "  2. Initialize your node:"
    echo "     $ sgl-node init --wallet <YOUR_SOLANA_ADDRESS>"
    echo ""
    echo "  3. Verify wallet ownership:"
    echo "     $ sgl-node attest"
    echo ""
    echo "  4. Download a model (e.g., Llama 3.2 3B):"
    echo "     $ curl -L -o model.gguf <model-download-url>"
    echo ""
    echo "  5. Start earning:"
    echo "     $ sgl-node start --model-path ./model.gguf --model-name llama-3.2-3b"
    echo ""
    echo "  Docs: https://sgl.network/docs"
    echo "  Discord: https://discord.gg/singularity"
    echo ""
}

main
