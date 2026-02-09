# vLLM for Production: Research & Migration Plan

Research document covering what it would take to use vLLM instead of (or alongside) Ollama for LLM inference in the Starship Horizons Learning AI system.

---

## 1. Current LLM Architecture (As-Is)

### Four Independent Ollama Clients

The codebase contains **four separate HTTP clients** that each independently call Ollama's proprietary `/api/generate` endpoint. None share a common interface or base class.

| Client | File | HTTP Lib | Sync/Async | Used By |
|--------|------|----------|------------|---------|
| `OllamaClient` | `src/llm/ollama_client.py` | `requests` | Sync | `src/metrics/mission_summarizer.py`, `scripts/generate_mission_report.py`, tests |
| `NarrativeSummaryGenerator` | `src/web/narrative_summary.py` | `httpx` | Async | `src/web/audio_processor.py` (web pipeline) |
| `TranscriptLLMCleaner` | `src/audio/domain_postcorrector.py` | `requests` | Sync | `src/web/audio_processor.py` (transcript cleanup) |
| `TitleGenerator` | `src/web/title_generator.py` | `httpx` | Async | `src/web/audio_processor.py` (title generation) |

### Fragmentation Problems

1. **No shared interface.** Each client constructs its own HTTP payload, reads its own env vars, and handles errors differently. Changing a parameter name or adding a feature (like request retries) means touching four files.

2. **Duplicate env var reads.** All four independently read `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, and `OLLAMA_TIMEOUT` from the environment, with inconsistent defaults:
   - `OllamaClient` defaults model to `llama3.2`, timeout to `120`
   - `NarrativeSummaryGenerator` defaults model to `qwen2.5:14b-instruct`, timeout to `600`
   - `TranscriptLLMCleaner` defaults model to `llama3.2`, timeout to `120`
   - `TitleGenerator` defaults model to `llama3.2`, timeout to `30.0`

3. **Mixed HTTP libraries.** Two clients use `requests` (sync), two use `httpx` (async). Neither `httpx` nor `openai` are in `requirements.txt` (`httpx` is pulled in as a transitive dependency of `fastapi`).

4. **Dead dependency.** The `ollama>=0.4.0` pip package is listed in `requirements.txt` (line 58) but **never imported anywhere in the codebase**. It can be removed.

5. **`OllamaClient` is not used by the web pipeline.** The main web analysis flow in `src/web/audio_processor.py` uses `NarrativeSummaryGenerator`, `TitleGenerator`, and `TranscriptLLMCleaner` directly. `OllamaClient` is only used by the older `mission_summarizer.py` code path and standalone scripts.

### How Each Client Calls Ollama

All four use Ollama's native `/api/generate` endpoint with this payload shape:

```json
{
  "model": "qwen2.5:14b-instruct",
  "prompt": "...",
  "system": "...",           // OllamaClient only (prompt-level)
  "stream": false,
  "options": {
    "temperature": 0.3,
    "num_predict": 2500,     // max output tokens
    "num_ctx": 32768,        // context window
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
  }
}
```

Response shape (non-streaming):

```json
{
  "response": "generated text...",
  "prompt_eval_count": 1234,
  "eval_count": 567,
  "eval_duration": 12345678900,
  "total_duration": 23456789000
}
```

---

## 2. vLLM Compatibility Analysis

### OpenAI API Protocol

vLLM's primary interface is the OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`). Ollama **also** exposes this same OpenAI-compatible API. This means a single client that speaks the OpenAI protocol works with both backends.

### Current Pattern Maps to OpenAI Messages

The current `prompt` + `system` pattern maps directly:

```python
# Current (Ollama native)
payload = {
    "model": "qwen2.5:14b-instruct",
    "prompt": user_text,
    "system": system_text,
    "options": {"temperature": 0.3, "num_predict": 2500}
}
requests.post(f"{host}/api/generate", json=payload)

# Equivalent (OpenAI protocol - works with both Ollama and vLLM)
payload = {
    "model": "qwen2.5:14b-instruct",
    "messages": [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ],
    "temperature": 0.3,
    "max_tokens": 2500
}
requests.post(f"{host}/v1/chat/completions", json=payload)
```

### Parameter Mapping

| Current (Ollama native) | OpenAI Protocol | Notes |
|--------------------------|----------------|-------|
| `options.temperature` | `temperature` | Same semantics |
| `options.num_predict` | `max_tokens` | Same semantics |
| `options.top_p` | `top_p` | Same semantics |
| `options.top_k` | `extra_body.top_k` | Not in OpenAI spec; both Ollama and vLLM accept via `extra_body` |
| `options.repeat_penalty` | `extra_body.repetition_penalty` | vLLM uses this name; Ollama's OpenAI endpoint also accepts it |
| `options.num_ctx` | Server-side config | Ollama: `OLLAMA_NUM_CTX` env var or per-model. vLLM: `--max-model-len` flag |
| `options.stop` | `stop` | Same semantics |
| `stream: true` | `stream: true` | Both return SSE chunks with `data: {...}` |

### Response Mapping

| Ollama native response | OpenAI protocol response | Notes |
|------------------------|--------------------------|-------|
| `response` | `choices[0].message.content` | Text extraction path |
| `prompt_eval_count` | `usage.prompt_tokens` | Token counts |
| `eval_count` | `usage.completion_tokens` | Token counts |
| `eval_duration` | Not directly available | vLLM doesn't expose this; compute from streaming timestamps |

---

## 3. Key Differences: Ollama vs vLLM

| Aspect | Ollama | vLLM |
|--------|--------|------|
| **GPU required** | No (CPU fallback via llama.cpp) | Yes (CUDA required) |
| **Model management** | Built-in: `ollama pull model` | Load from HuggingFace path or local dir at startup |
| **Models per process** | Multiple (dynamic load/unload) | One model per process (set at startup with `--model`) |
| **Throughput** | ~30-50 tok/s (single GPU, 14B Q4) | ~500-800+ tok/s (single GPU, 14B, continuous batching) |
| **Concurrent requests** | Sequential (one request at a time per model) | Continuous batching (many concurrent requests) |
| **OpenAI API** | Yes (also has native API) | Yes (primary and only interface) |
| **Quantization** | GGUF via llama.cpp | AWQ, GPTQ, FP8, bitsandbytes |
| **Guided decoding** | No | Yes (`response_format`, `guided_json`, `guided_regex`) |
| **Setup complexity** | `ollama pull model` (one command) | `pip install vllm && vllm serve HF/model` |
| **Memory management** | Conservative (loads/unloads) | PagedAttention (efficient KV cache sharing) |
| **Multi-GPU** | Limited (model splitting) | Tensor parallelism (`--tensor-parallel-size N`) |
| **Docker image** | `ollama/ollama` (~1GB) | `vllm/vllm-openai` (~8GB with CUDA) |
| **CPU-only dev** | Works out of the box | Not supported (needs CUDA) |

### When to Use Which

- **Development / CPU-only / small models:** Ollama. It runs everywhere, manages models simply, and is good enough for development iteration.
- **Production / GPU / throughput matters:** vLLM. Continuous batching means multiple concurrent analysis requests don't queue. Guided decoding can enforce JSON output schemas.
- **Both at once:** Perfectly valid. The unified client (Phase 1) makes the backend a config change. Dev uses Ollama, prod uses vLLM, same code.

---

## 4. Recommended Migration Path

### Phase 1: Unify LLM Clients Behind a Single Abstraction

**Goal:** Create one client class that all four consumers use. The client speaks the OpenAI `/v1/chat/completions` protocol, which both Ollama and vLLM support.

#### New File: `src/llm/llm_client.py`

```python
"""
Unified LLM client using the OpenAI-compatible API protocol.

Works with any backend that implements the OpenAI chat completions API:
Ollama, vLLM, OpenAI, Azure OpenAI, etc.
"""

class LLMClient:
    """
    Unified LLM client.

    Provides both sync generate() and async agenerate() methods.
    Uses the OpenAI /v1/chat/completions protocol.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,    # LLM_BASE_URL or OLLAMA_HOST + "/v1"
        model: Optional[str] = None,        # LLM_MODEL or OLLAMA_MODEL
        timeout: Optional[float] = None,    # LLM_TIMEOUT or OLLAMA_TIMEOUT
        api_key: Optional[str] = None,      # LLM_API_KEY (default "ollama" for Ollama)
    ):
        ...

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict] = None,  # top_k, repetition_penalty, etc.
    ) -> Optional[str]:
        """Synchronous generation. Used by OllamaClient, TranscriptLLMCleaner."""
        ...

    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict] = None,
    ) -> Optional[str]:
        """Async generation. Used by NarrativeSummaryGenerator, TitleGenerator."""
        ...
```

#### Environment Variable Migration

| Current | New (with fallback) | Notes |
|---------|---------------------|-------|
| `OLLAMA_HOST` | `LLM_BASE_URL` (fallback: `OLLAMA_HOST` + `"/v1"`) | Append `/v1` automatically if not present |
| `OLLAMA_MODEL` | `LLM_MODEL` (fallback: `OLLAMA_MODEL`) | Same value |
| `OLLAMA_TIMEOUT` | `LLM_TIMEOUT` (fallback: `OLLAMA_TIMEOUT`) | Same value |
| `OLLAMA_NUM_CTX` | Removed from client | Server-side config for both Ollama and vLLM |
| (new) | `LLM_API_KEY` | Default `"ollama"` for Ollama (which ignores it); required for OpenAI/Azure |

Backward compatibility: if `LLM_BASE_URL` is not set, the client reads `OLLAMA_HOST` and appends `/v1`. Existing `.env` files work unchanged.

#### HTTP Library Choice

Use the `openai` Python package (`openai>=1.0`). Benefits:
- Type-safe request/response objects
- Built-in streaming support (sync and async)
- Handles SSE parsing, retries, and error types
- `openai.OpenAI(base_url=..., api_key=...)` works with any compatible backend
- Already supports `extra_body` for vendor-specific params

Add to `requirements.txt`: `openai>=1.0.0`. Remove: `ollama>=0.4.0` (unused).

#### Refactoring Each Consumer

**1. `OllamaClient` (`src/llm/ollama_client.py`)**

The `generate()` method becomes a thin wrapper around `LLMClient.generate()`. The `generate_streaming()` and `generate_with_progress()` methods use `LLMClient.generate(..., stream=True)`. Higher-level methods (`generate_mission_summary`, `generate_hybrid_report`, etc.) stay as-is but call the unified client internally.

Files changed: `src/llm/ollama_client.py`
Risk: Low. `OllamaClient` is used by `mission_summarizer.py` and scripts — can be tested independently.

**2. `NarrativeSummaryGenerator` (`src/web/narrative_summary.py`)**

Replace the inline `httpx.post(f"{host}/api/generate", ...)` calls in `generate_summary()` and `generate_story()` with `LLMClient.agenerate()`. Remove the `_get_client()` / `close()` httpx lifecycle methods.

Files changed: `src/web/narrative_summary.py`
Risk: Medium. This is the main web pipeline path. The streaming variant (`generate_summary_streaming`) needs `LLMClient.agenerate_streaming()` or can be implemented with the `openai` package's streaming iterator.

**3. `TranscriptLLMCleaner` (`src/audio/domain_postcorrector.py`)**

Replace `_call_ollama()` (which uses `requests.post`) with `LLMClient.generate()`. The rest of the class (batching, parsing, thread pool) stays unchanged.

Files changed: `src/audio/domain_postcorrector.py`
Risk: Low. Self-contained; the class already has good error handling.

**4. `TitleGenerator` (`src/web/title_generator.py`)**

Replace `generate_title_with_llm()` (which uses `httpx.post`) with `LLMClient.agenerate()`. Remove httpx client lifecycle.

Files changed: `src/web/title_generator.py`
Risk: Low. Small, isolated class with fallback behavior.

#### Phase 1 Summary

| Item | Effort |
|------|--------|
| Create `src/llm/llm_client.py` | ~200 lines (sync + async + streaming + config) |
| Create `tests/test_llm_client.py` | ~150 lines (mock-based unit tests) |
| Refactor `OllamaClient` | Small — replace `requests.post` calls with `LLMClient.generate()` |
| Refactor `NarrativeSummaryGenerator` | Medium — replace httpx calls, remove client lifecycle |
| Refactor `TranscriptLLMCleaner` | Small — replace `_call_ollama()` internals |
| Refactor `TitleGenerator` | Small — replace httpx calls, remove client lifecycle |
| Update `requirements.txt` | Add `openai>=1.0.0`, remove `ollama>=0.4.0` |
| Update `.env.example` | Add `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` with docs |
| Update `src/llm/__init__.py` | Export `LLMClient` |

**Total estimate:** ~400 lines of new code, ~200 lines of deletions across 4 consumer files, plus tests.

### Phase 2: Add vLLM as a Deployment Option

With the unified OpenAI-protocol client from Phase 1, vLLM support becomes a **configuration change only** — no code changes required.

#### Configuration for vLLM

```bash
# .env for vLLM deployment
LLM_BASE_URL=http://vllm-host:8000/v1
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
LLM_API_KEY=token-placeholder       # vLLM accepts any non-empty string by default
LLM_TIMEOUT=300
```

#### Docker Compose Addition

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-14B-Instruct
      --max-model-len 32768
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.90
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - huggingface-cache:/root/.cache/huggingface

  app:
    # ... existing app config ...
    environment:
      LLM_BASE_URL: http://vllm:8000/v1
      LLM_MODEL: Qwen/Qwen2.5-14B-Instruct
    depends_on:
      - vllm
```

#### vLLM-Specific Features Available via `extra_body`

Once using the OpenAI protocol, vLLM-specific features become available through the `extra_body` parameter without any client changes:

```python
# Guided JSON output (enforces valid JSON schema)
result = client.generate(
    prompt="Analyze this transcript...",
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "grade": {"type": "string", "enum": ["A", "B", "C", "D", "F"]},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "improvements": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["grade", "strengths", "improvements"]
        }
    }
)

# Guided regex (enforces output matches a pattern)
result = client.generate(
    prompt="Generate a title...",
    extra_body={
        "guided_regex": r"[A-Z][a-z ]{10,60}"
    }
)
```

This is particularly valuable for:
- **Title generation** — enforce length/format constraints instead of post-hoc validation
- **Transcript cleanup** — enforce the `LINE_NUMBER|corrected text` format
- **Structured analysis** — get guaranteed-valid JSON instead of hoping the LLM follows format instructions

---

## 5. Effort Estimate and Risk Assessment

### Phase 1 (Unified Client)

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Ollama OpenAI endpoint behavior** | Ollama's `/v1/chat/completions` may have subtle differences from its native `/api/generate` (e.g., system prompt handling, stop sequences) | Test with the specific Ollama version in use; the OpenAI endpoint has been stable since Ollama 0.1.29+ |
| **Streaming response format** | The `openai` package expects SSE `data: {...}` format; verify Ollama's streaming matches | Ollama's OpenAI endpoint uses standard SSE; well-tested with the `openai` package |
| **Token metrics** | Moving from Ollama's native response (which includes `eval_duration`, `prompt_eval_count`) to OpenAI's `usage` object loses some metrics | OpenAI `usage` provides `prompt_tokens` and `completion_tokens`; `eval_duration` (tokens/sec) can be computed client-side from wall-clock time |
| **`num_ctx` handling** | Currently passed per-request; OpenAI protocol doesn't have this field | Set server-side via `OLLAMA_NUM_CTX` env var (Ollama) or `--max-model-len` (vLLM). Remove from per-request payloads. |
| **Backward compatibility** | Existing `.env` files use `OLLAMA_*` variables | New client reads `LLM_*` with fallback to `OLLAMA_*`. No `.env` changes required. |

### Phase 2 (vLLM Deployment)

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Single model per process** | vLLM serves one model; the system currently assumes one Ollama serves all requests | All four consumers use the same `OLLAMA_MODEL` — this is fine. If different models were needed, run multiple vLLM instances. |
| **GPU requirement** | vLLM requires CUDA; no CPU fallback | Keep Ollama as the dev/CPU option. vLLM is production-only. |
| **Model name format** | Ollama uses `qwen2.5:14b-instruct`; vLLM uses HuggingFace names like `Qwen/Qwen2.5-14B-Instruct` | The `LLM_MODEL` env var handles this — set the right name for your backend. |
| **Docker image size** | `vllm/vllm-openai` is ~8GB (includes CUDA runtime) | Acceptable for production GPU servers; not suitable for dev containers. |
| **Cold start time** | vLLM takes 30-120s to load a 14B model | Use health checks and readiness probes in Docker/Kubernetes. |

---

## 6. Hardware and Deployment Considerations

### GPU Requirements for vLLM

| Model | VRAM Required (FP16) | VRAM Required (AWQ/GPTQ) | Recommended GPU |
|-------|-----------------------|--------------------------|-----------------|
| Qwen2.5-7B-Instruct | ~14 GB | ~4 GB | RTX 3090 / A10 |
| Qwen2.5-14B-Instruct | ~28 GB | ~8 GB | A100-40GB / RTX 4090 |
| Qwen2.5-32B-Instruct | ~64 GB | ~18 GB | A100-80GB / 2x A10 |
| Qwen2.5-72B-Instruct | ~144 GB | ~40 GB | 2x A100-80GB |

For the currently configured `qwen2.5:14b-instruct`, an AWQ-quantized variant on a single A10 (24GB) or RTX 4090 (24GB) is the sweet spot.

### Deployment Architectures

#### Option A: Ollama Only (Current)

```
[App Container] --HTTP--> [Ollama Container]
                           - CPU or GPU
                           - Dynamic model loading
                           - ~30-50 tok/s
```

Best for: development, single-user, CPU-only environments.

#### Option B: vLLM Only (Production)

```
[App Container] --HTTP--> [vLLM Container]
                           - GPU required
                           - Single model, always loaded
                           - ~500-800 tok/s
                           - Continuous batching
```

Best for: production with dedicated GPU, multiple concurrent users.

#### Option C: Both (Recommended for Flexibility)

```
[App Container] --HTTP--> [Ollama Container]    (dev/CPU)
       |
       +-- env switch --> [vLLM Container]      (prod/GPU)
```

The unified client makes switching between backends a single env var change (`LLM_BASE_URL`). The Dockerfile can optionally include both, or they can be separate services in Docker Compose.

### Cloud Deployment Options

| Cloud | GPU Instance | Monthly Cost (approx) | Notes |
|-------|-------------|----------------------|-------|
| **Azure** | NC24ads A100 v4 (A100 80GB) | ~$3,700 | Best for 14B-72B models |
| **Azure** | NC6s v3 (V100 16GB) | ~$900 | 7B quantized only |
| **AWS** | g5.xlarge (A10G 24GB) | ~$1,000 | Good for 14B AWQ |
| **AWS** | p4d.24xlarge (8x A100 80GB) | ~$32,000 | Overkill; multi-tenant |
| **GCP** | g2-standard-4 (L4 24GB) | ~$700 | Budget option for 14B AWQ |
| **RunPod/Lambda** | A10G 24GB (spot) | ~$200-400 | Cheapest for experimentation |

### Current Dockerfile Impact

The existing `Dockerfile` bundles Ollama inside the app container (installs via `curl -fsSL https://ollama.com/install.sh | sh`). For a vLLM deployment:

1. **Remove Ollama from Dockerfile** (or make it optional via build arg)
2. **Add vLLM as a separate Docker Compose service** (cleaner separation)
3. **Keep the app container lightweight** — it only needs the `openai` Python package, not Ollama or vLLM installed locally

---

## 7. Summary and Recommendations

### Immediate Action (Phase 1)

Unify the four LLM clients behind `src/llm/llm_client.py` using the OpenAI protocol. This:
- Eliminates code duplication across four files
- Fixes inconsistent defaults (model names, timeouts)
- Removes a dead dependency (`ollama` pip package)
- Makes the system backend-agnostic (Ollama, vLLM, OpenAI, Azure OpenAI)
- Is a prerequisite for any production LLM improvements

### When Ready (Phase 2)

Add vLLM as a deployment option. This:
- Is a config-only change once Phase 1 is complete
- Provides 10-20x throughput improvement
- Enables concurrent request handling (continuous batching)
- Unlocks guided decoding for structured output
- Requires a GPU in production (keep Ollama for dev)

### What NOT to Do

- **Don't add vLLM without Phase 1.** Attempting to support vLLM with four independent Ollama-specific clients would double the fragmentation.
- **Don't remove Ollama support.** It's essential for CPU-only development and simple deployments.
- **Don't use the `ollama` pip package.** It's already unused and the OpenAI protocol is the better interface for backend portability.
