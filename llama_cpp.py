from pathlib import Path
import subprocess


# ============================================================
# Configuration — edit these values
# ============================================================

LLAMA_SERVER = Path(
    r""
)

CHAT_TEMPLATE = Path(
    r""
)

MODEL_PATH = Path(
    r""
)

# Set this to a valid mmproj file to enable vision.
#
# The server continues without vision when:
#   - MMPROJ_PATH is None
#   - The path is blank
#   - The file does not exist
#
# The mmproj is always kept in system RAM.
#
# Example:
# MMPROJ_PATH = Path(
#     r"C:\Users\Neal\Documents\llms\Models\mmproj\Qwen3.6-35B-A3B-mmproj-BF16.gguf"
# )

MMPROJ_PATH = None


# ============================================================
# Server options — may be useful to edit
# ============================================================

MODEL_NAME = "Qwen3.6 35B A3B"

HOST = "127.0.0.1"
PORT = 8081

CONTEXT_SIZE = 128 * 1024

# Valid values:
#   "bf16"
#   "q8_0"
KV_CACHE_TYPE = "bf16"

REASONING = False
PRESERVE_REASONING = False


# ============================================================
# Qwen coding sampling settings
# ============================================================

# These settings are used regardless of whether reasoning is
# enabled or disabled.

SAMPLING_SETTINGS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}


# ============================================================
# Shared llama.cpp options — technical tuning
# ============================================================

PARALLEL_SLOTS = 1
GPU_LAYERS = -1

BATCH_SIZE = 2048
UBATCH_SIZE = 512

FLASH_ATTENTION = True

FIT_ENABLED = True
FIT_TARGET_MIB = 1024

# Prompt cache size in MiB.
# Set to 0 to disable the prompt cache.
PROMPT_CACHE_MIB = 4096

LOG_TIMESTAMPS = True
LOG_VERBOSITY = 4


# ============================================================
# Speculative decoding options
# ============================================================

# N-gram is always enabled.
NGRAM_MATCH = 24
NGRAM_MIN = 48
NGRAM_MAX = 64

# MTP uses the MTP heads embedded in MODEL_PATH.
ENABLE_MTP = True

MTP_DRAFT_TOKENS = 3
MTP_MIN_PROBABILITY = 0.75


# ============================================================
# Validation
# ============================================================

def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(
            f"\nERROR: {description} was not found:\n"
            f"{path}"
        )


def validate_configuration() -> None:
    if KV_CACHE_TYPE not in {"bf16", "q8_0"}:
        raise SystemExit(
            f"\nERROR: Invalid KV_CACHE_TYPE: {KV_CACHE_TYPE!r}\n"
            'Use either "bf16" or "q8_0".'
        )

    if not isinstance(HOST, str) or not HOST.strip():
        raise SystemExit(
            "\nERROR: HOST cannot be empty."
        )

    if not isinstance(PORT, int) or not 1 <= PORT <= 65535:
        raise SystemExit(
            f"\nERROR: Invalid PORT: {PORT!r}\n"
            "PORT must be an integer from 1 through 65535."
        )

    if CONTEXT_SIZE <= 0:
        raise SystemExit(
            "\nERROR: CONTEXT_SIZE must be greater than zero."
        )

    if PARALLEL_SLOTS <= 0:
        raise SystemExit(
            "\nERROR: PARALLEL_SLOTS must be greater than zero."
        )

    if BATCH_SIZE <= 0:
        raise SystemExit(
            "\nERROR: BATCH_SIZE must be greater than zero."
        )

    if UBATCH_SIZE <= 0:
        raise SystemExit(
            "\nERROR: UBATCH_SIZE must be greater than zero."
        )

    if UBATCH_SIZE > BATCH_SIZE:
        raise SystemExit(
            "\nERROR: UBATCH_SIZE cannot exceed BATCH_SIZE."
        )

    if FIT_TARGET_MIB < 0:
        raise SystemExit(
            "\nERROR: FIT_TARGET_MIB cannot be negative."
        )

    if not isinstance(PROMPT_CACHE_MIB, int):
        raise SystemExit(
            "\nERROR: PROMPT_CACHE_MIB must be an integer."
        )

    if PROMPT_CACHE_MIB < 0:
        raise SystemExit(
            "\nERROR: PROMPT_CACHE_MIB cannot be negative."
        )

    if NGRAM_MATCH <= 0:
        raise SystemExit(
            "\nERROR: NGRAM_MATCH must be greater than zero."
        )

    if NGRAM_MIN <= 0:
        raise SystemExit(
            "\nERROR: NGRAM_MIN must be greater than zero."
        )

    if NGRAM_MAX < NGRAM_MIN:
        raise SystemExit(
            "\nERROR: NGRAM_MAX must be greater than or equal to NGRAM_MIN."
        )

    if MTP_DRAFT_TOKENS <= 0:
        raise SystemExit(
            "\nERROR: MTP_DRAFT_TOKENS must be greater than zero."
        )

    if not 0.0 <= MTP_MIN_PROBABILITY <= 1.0:
        raise SystemExit(
            "\nERROR: MTP_MIN_PROBABILITY must be between 0 and 1."
        )

    if SAMPLING_SETTINGS["temperature"] < 0:
        raise SystemExit(
            "\nERROR: Sampling temperature cannot be negative."
        )

    if not 0.0 <= SAMPLING_SETTINGS["top_p"] <= 1.0:
        raise SystemExit(
            "\nERROR: Sampling top_p must be between 0 and 1."
        )

    if SAMPLING_SETTINGS["top_k"] < 0:
        raise SystemExit(
            "\nERROR: Sampling top_k cannot be negative."
        )

    if not 0.0 <= SAMPLING_SETTINGS["min_p"] <= 1.0:
        raise SystemExit(
            "\nERROR: Sampling min_p must be between 0 and 1."
        )

    if SAMPLING_SETTINGS["repeat_penalty"] <= 0:
        raise SystemExit(
            "\nERROR: Sampling repeat_penalty must be greater than zero."
        )

    require_file(
        LLAMA_SERVER,
        "llama-server",
    )

    require_file(
        CHAT_TEMPLATE,
        "fixed Qwen chat template",
    )

    require_file(
        MODEL_PATH,
        MODEL_NAME,
    )


# ============================================================
# Optional mmproj handling
# ============================================================

def get_mmproj_path() -> Path | None:
    if MMPROJ_PATH is None:
        return None

    if not str(MMPROJ_PATH).strip():
        return None

    mmproj_path = Path(MMPROJ_PATH)

    if not mmproj_path.is_file():
        print(
            "\nWARNING: mmproj was not found. "
            "Continuing without vision:"
        )
        print(f"{mmproj_path}\n")
        return None

    return mmproj_path


# ============================================================
# Speculative decoding
# ============================================================

def build_speculative_args() -> tuple[list[str], str]:
    spec_types = ["ngram-mod"]

    args = [
        "--spec-ngram-mod-n-match",
        str(NGRAM_MATCH),

        "--spec-ngram-mod-n-min",
        str(NGRAM_MIN),

        "--spec-ngram-mod-n-max",
        str(NGRAM_MAX),
    ]

    if ENABLE_MTP:
        spec_types.insert(0, "draft-mtp")

        args.extend([
            "--spec-draft-n-max",
            str(MTP_DRAFT_TOKENS),

            "--spec-draft-p-min",
            str(MTP_MIN_PROBABILITY),
        ])

    description = " + ".join(spec_types)

    return [
        "--spec-type",
        ",".join(spec_types),
        *args,
    ], description


# ============================================================
# Build command
# ============================================================

def build_command(
    mmproj_path: Path | None,
) -> tuple[list[str], str]:

    cmd = [
        str(LLAMA_SERVER),

        "-m",
        str(MODEL_PATH),

        "--alias",
        MODEL_NAME,

        "--host",
        HOST,

        "--port",
        str(PORT),

        "-np",
        str(PARALLEL_SLOTS),

        "--ctx-size",
        str(CONTEXT_SIZE),

        "-ngl",
        str(GPU_LAYERS),

        "--cache-type-k",
        KV_CACHE_TYPE,

        "--cache-type-v",
        KV_CACHE_TYPE,

        "-b",
        str(BATCH_SIZE),

        "-ub",
        str(UBATCH_SIZE),

        "--jinja",

        "--chat-template-file",
        str(CHAT_TEMPLATE),

        "--reasoning",
        "on" if REASONING else "off",

        (
            "--reasoning-preserve"
            if PRESERVE_REASONING
            else "--no-reasoning-preserve"
        ),

        # Qwen coding sampling settings
        "--temp",
        str(SAMPLING_SETTINGS["temperature"]),

        "--top-p",
        str(SAMPLING_SETTINGS["top_p"]),

        "--top-k",
        str(SAMPLING_SETTINGS["top_k"]),

        "--min-p",
        str(SAMPLING_SETTINGS["min_p"]),

        "--presence-penalty",
        str(SAMPLING_SETTINGS["presence_penalty"]),

        "--repeat-penalty",
        str(SAMPLING_SETTINGS["repeat_penalty"]),

        # Prompt cache size in MiB.
        "--cache-ram",
        str(PROMPT_CACHE_MIB),
    ]

    if mmproj_path is not None:
        cmd.extend([
            "--mmproj",
            str(mmproj_path),

            # Keep the multimodal projector in system RAM.
            "--no-mmproj-offload",
        ])

    if FLASH_ATTENTION:
        cmd.extend([
            "--flash-attn",
            "on",
        ])
    else:
        cmd.extend([
            "--flash-attn",
            "off",
        ])

    if FIT_ENABLED:
        cmd.extend([
            "-fit",
            "on",

            "-fitt",
            str(FIT_TARGET_MIB),
        ])
    else:
        cmd.extend([
            "-fit",
            "off",
        ])

    speculative_args, speculative_description = (
        build_speculative_args()
    )

    cmd.extend(speculative_args)

    if LOG_TIMESTAMPS:
        cmd.append("--log-timestamps")

    cmd.extend([
        "--log-verbosity",
        str(LOG_VERBOSITY),
    ])

    return cmd, speculative_description


# ============================================================
# Run llama-server
# ============================================================

def main() -> int:
    validate_configuration()

    mmproj_path = get_mmproj_path()

    cmd, speculative_description = build_command(
        mmproj_path
    )

    print(f"\nStarting: {MODEL_NAME}")
    print(f"Model: {MODEL_PATH}")

    if mmproj_path is not None:
        print(f"mmproj: {mmproj_path}")
        print("mmproj location: System RAM")
    else:
        print("mmproj: Disabled")

    print(f"Context: {CONTEXT_SIZE:,} tokens")
    print(f"KV cache type: {KV_CACHE_TYPE.upper()}")
    print("KV cache location: GPU offload enabled")

    print(f"MTP: {'Enabled' if ENABLE_MTP else 'Disabled'}")
    print(f"Speculative decoding: {speculative_description}")

    print(f"Fit: {'On' if FIT_ENABLED else 'Off'}")

    if FIT_ENABLED:
        print(f"Fit target: {FIT_TARGET_MIB:,} MiB free")

    if PROMPT_CACHE_MIB > 0:
        print(f"Prompt cache: {PROMPT_CACHE_MIB:,} MiB")
    else:
        print("Prompt cache: Disabled")

    print(f"Reasoning: {'On' if REASONING else 'Off'}")
    print(
        "Preserve reasoning history: "
        f"{'On' if PRESERVE_REASONING else 'Off'}"
    )

    print("Sampling preset: Coding")
    print(
        f"  Temperature: "
        f"{SAMPLING_SETTINGS['temperature']}"
    )
    print(
        f"  Top P: "
        f"{SAMPLING_SETTINGS['top_p']}"
    )
    print(
        f"  Top K: "
        f"{SAMPLING_SETTINGS['top_k']}"
    )
    print(
        f"  Min P: "
        f"{SAMPLING_SETTINGS['min_p']}"
    )
    print(
        f"  Presence penalty: "
        f"{SAMPLING_SETTINGS['presence_penalty']}"
    )
    print(
        f"  Repeat penalty: "
        f"{SAMPLING_SETTINGS['repeat_penalty']}"
    )

    print(f"Endpoint: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.\n")

    process = subprocess.Popen(
        cmd,
        shell=False,
    )

    try:
        return process.wait()

    except KeyboardInterrupt:
        print("\nStopping llama-server...")
        process.terminate()

        try:
            return process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            print("Forcing llama-server to close...")
            process.kill()
            return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())