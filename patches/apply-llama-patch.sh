#!/bin/bash
# Apply a patch to llama.cpp's llama-model-loader.cpp to accept 3-element
# rope.dimension_sections arrays (required for Qwen3.5/Qwen3.6 GGUF models).
#
# Run this once after `cargo build` to apply the patch, then rebuild:
#   ./patches/apply-llama-patch.sh && cargo clean -p llama-cpp-sys-2 && cargo build

set -e

LOADER=$(find ~/.cargo/git/checkouts/llama-cpp-rs-* -name "llama-model-loader.cpp" -path "*/llama.cpp/src/*" 2>/dev/null | head -1)

if [ -z "$LOADER" ]; then
    echo "ERROR: Could not find llama-model-loader.cpp in cargo git checkouts"
    echo "Run 'cargo build' first to download the dependency, then re-run this script."
    exit 1
fi

if grep -q "PATCHED: arr_info.length > n" "$LOADER" 2>/dev/null; then
    echo "Patch already applied to $LOADER"
    exit 0
fi

if ! grep -q "if (n != arr_info.length) {" "$LOADER" 2>/dev/null; then
    echo "Patch target not found — upstream may have already fixed this."
    echo "Checking if upstream fix is present..."
    if grep -q "if (arr_info.length > n)" "$LOADER" 2>/dev/null; then
        echo "Upstream fix confirmed. No patch needed."
        exit 0
    fi
    echo "ERROR: Could not find patch target. The source may have changed."
    exit 1
fi

# Apply the patch
sed -i.bak 's/if (n != arr_info.length) {/if (arr_info.length > n) { \/* PATCHED: arr_info.length > n *\//g' "$LOADER"
rm -f "${LOADER}.bak"

echo "Patch applied to $LOADER"
echo ""
echo "Now rebuild with:"
echo "  cargo clean -p llama-cpp-sys-2 && cargo build"
