#!/bin/bash
# =============================================================================
# Test All Model Configs
# =============================================================================
# Tests each config preset by:
# 1. Starting the server with --config <path>
# 2. Waiting for gRPC to be ready
# 3. Sending a test request
# 4. Recording success/failure
#
# Usage:
#   ./scripts/test-all-configs.sh [--quick] [--foundation-only] [--nlp-only]
#                                 [--audio-only] [--vision-only] [--multimodal-only]
#                                 [--quiet]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Config
GRPC_PORT=50051
STARTUP_TIMEOUT=320

# Skip these configs (known blockers)
SKIP_CONFIGS=""

# Results
PASSED=0
FAILED=0
SKIPPED=0
PASSED_LIST=""
FAILED_LIST=""
SKIPPED_LIST=""

# Parse arguments
FOUNDATION_ONLY=false
NLP_ONLY=false
AUDIO_ONLY=false
VISION_ONLY=false
MULTIMODAL_ONLY=false
VERBOSE=true
for arg in "$@"; do
    case $arg in
        --foundation-only) FOUNDATION_ONLY=true ;;
        --nlp-only) NLP_ONLY=true ;;
        --audio-only) AUDIO_ONLY=true ;;
        --vision-only) VISION_ONLY=true ;;
        --multimodal-only) MULTIMODAL_ONLY=true ;;
        --quick) STARTUP_TIMEOUT=30 ;;
        --quiet) VERBOSE=false ;;
    esac
done

# Force kill all inference servers
kill_server() {
    pkill -9 -f "inference-service" 2>/dev/null || true
    sleep 2
    while lsof -i :${GRPC_PORT} >/dev/null 2>&1; do
        echo "  Waiting for port ${GRPC_PORT} to be free..."
        pkill -9 -f "inference-service" 2>/dev/null || true
        sleep 1
    done
}

# Extract values from config (macOS compatible)
get_task_type() {
    grep -E '^type[[:space:]]*=' "$1" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]*)".*/\1/'
}

get_task_name() {
    grep -E '^name[[:space:]]*=' "$1" | head -1 | sed -E 's/.*=[[:space:]]*"([^"]*)".*/\1/'
}

# Get test payload based on task type
get_test_payload() {
    case "$1" in
        echo)
            echo '{"message":"hello world"}'
            ;;
        feature-extraction|text-classification|token-classification|fill-mask|text-generation|summarization|translation)
            echo '{"text":"Hello world, this is a test."}'
            ;;
        question-answering)
            echo '{"question":"What is Paris?","context":"Paris is the capital of France."}'
            ;;
        sentence-similarity)
            echo '{"text_a":"Hello world","text_b":"Hi there"}'
            ;;
        zero-shot-classification)
            echo '{"text":"I love this product!","candidate_labels":["positive","negative"]}'
            ;;
        automatic-speech-recognition|audio-classification|audio-text-to-text)
            echo '{"audio":"data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="}'
            ;;
        text-to-speech)
            echo '{"text":"Hello world"}'
            ;;
        image-classification|object-detection|image-segmentation|depth-estimation|image-to-text|image-feature-extraction|ocr)
            echo '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="}'
            ;;
        zero-shot-image-classification)
            echo '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==","candidate_labels":["cat","dog"]}'
            ;;
        visual-question-answering|document-question-answering|image-text-to-text)
            echo '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==","text":"What is this?"}'
            ;;
        *)
            echo '{"text":"Hello world"}'
            ;;
    esac
}

# Wait for server to be ready (also checks if process died)
wait_for_ready() {
    local timeout=$1
    local pid=$2
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 2
        fi
        if grpcurl -plaintext 127.0.0.1:${GRPC_PORT} list 2>/dev/null | grep -q "maiia.worker"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

# Test a single config
test_one_config() {
    local config="$1"
    local config_name=$(basename "$config" .toml)
    local config_dir=$(dirname "$config" | xargs basename)
    local display_name="${config_dir}/${config_name}"

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing: ${display_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if echo " $SKIP_CONFIGS " | grep -q " $config_name "; then
        echo -e "  ${YELLOW}SKIPPED${NC} - not yet implemented"
        SKIPPED=$((SKIPPED + 1))
        SKIPPED_LIST="${SKIPPED_LIST}\n  - ${display_name} - skipped"
        return 0
    fi

    local task_type=$(get_task_type "$config")
    local task_name=$(get_task_name "$config")

    echo "  Task Type: ${task_type}"
    echo "  Task Name: ${task_name}"

    # Step 1: Kill any existing server
    echo "  [1/4] Killing existing servers..."
    kill_server

    # Step 2: Start server using the CLI --config flag
    echo "  [2/4] Starting server..."
    local log_file="/tmp/server_${config_name}.log"

    if [ "$VERBOSE" = true ]; then
        RUST_LOG=info ./target/release/inference-service --config "$config" 2>&1 | tee "$log_file" &
        sleep 0.5
    else
        RUST_LOG=info ./target/release/inference-service --config "$config" > "$log_file" 2>&1 &
    fi
    local server_pid=$!
    echo "  Server PID: $server_pid"

    # Step 3: Wait for ready
    echo "  [3/4] Waiting for server (timeout: ${STARTUP_TIMEOUT}s)..."
    wait_for_ready $STARTUP_TIMEOUT $server_pid
    local wait_result=$?

    if [ $wait_result -eq 2 ]; then
        echo -e "  ${RED}FAILED: Server process crashed during startup${NC}"
        echo "  Last 15 lines of log:"
        tail -15 "$log_file" 2>/dev/null | sed 's/^/    /'
        kill_server
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${display_name} - server crashed"
        return 1
    elif [ $wait_result -eq 1 ]; then
        echo -e "  ${RED}FAILED: Server did not become ready - timeout${NC}"
        echo "  Last 10 lines of log:"
        tail -10 "$log_file" 2>/dev/null | sed 's/^/    /'
        kill_server
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${display_name} - startup timeout"
        return 1
    fi
    echo "  Server ready!"

    # Step 4: Send test request
    echo "  [4/4] Sending test request..."
    local payload=$(get_test_payload "$task_type")
    local escaped_payload=$(echo "$payload" | jq -c @json)

    local response=$(grpcurl -plaintext -max-msg-sz 104857600 \
        -d "{\"task_name\":\"${task_name}\",\"payload\":${escaped_payload},\"request_id\":\"test-$$\"}" \
        127.0.0.1:${GRPC_PORT} maiia.worker.v1.WorkerService/ExecuteTask 2>&1)

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        local duration=$(echo "$response" | jq -r '.durationMs // "?"')
        grep -E "Task completed|execute_task" "$log_file" 2>/dev/null | tail -1 | sed 's/^/  /'
        echo -e "  ${GREEN}PASSED${NC} - ${duration}ms"
        PASSED=$((PASSED + 1))
        PASSED_LIST="${PASSED_LIST}\n  - ${display_name} - ${duration}ms"
    else
        local error=$(echo "$response" | jq -r '.error // empty' 2>/dev/null)
        if [ -z "$error" ]; then
            error="$response"
        fi
        echo -e "  ${RED}FAILED: ${error}${NC}"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${display_name}: ${error}"
    fi

    kill_server
    return 0
}

# =============================================================================
# MAIN
# =============================================================================

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              Model Config Test Suite                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Initial cleanup...${NC}"
kill_server

# Detect platform and set GPU build features
BUILD_FEATURES=""
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${YELLOW}Detected macOS - enabling Metal GPU acceleration${NC}"
    BUILD_FEATURES="--features metal"
elif command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Detected NVIDIA GPU - enabling CUDA acceleration${NC}"
    BUILD_FEATURES="--features all-cuda"
else
    echo -e "${YELLOW}No GPU detected - CPU-only (default features)${NC}"
fi

echo -e "${YELLOW}Building release binary...${NC}"
cargo build --release $BUILD_FEATURES

# Collect configs
if [ "$FOUNDATION_ONLY" = true ]; then
    configs=(configs/foundation/*.toml)
elif [ "$NLP_ONLY" = true ]; then
    configs=(configs/echo.toml configs/nlp/*.toml)
elif [ "$AUDIO_ONLY" = true ]; then
    configs=(configs/audio/*.toml)
elif [ "$VISION_ONLY" = true ]; then
    configs=(configs/vision/*.toml)
elif [ "$MULTIMODAL_ONLY" = true ]; then
    configs=(configs/multimodal/*.toml)
else
    configs=(
        configs/echo.toml
        configs/nlp/*.toml
        configs/foundation/*.toml
        configs/audio/*.toml
        configs/vision/*.toml
        configs/multimodal/*.toml
    )
fi

echo -e "${YELLOW}Will test ${#configs[@]} configs${NC}"

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        test_one_config "$config"
    fi
done

kill_server

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         SUMMARY                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}PASSED [${PASSED}]:${NC}"
echo -e "${PASSED_LIST}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}FAILED [${FAILED}]:${NC}"
    echo -e "${FAILED_LIST}"
fi

if [ $SKIPPED -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}SKIPPED [${SKIPPED}]:${NC}"
    echo -e "${SKIPPED_LIST}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Total: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC}, ${YELLOW}${SKIPPED} skipped${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

[ $FAILED -eq 0 ]
