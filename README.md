# audio-to-rag

> An async, queue-driven worker that turns raw meeting recordings into structured, speaker-aware, theme-labelled chunks with embeddings and a FAISS index, on AWS.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![AWS](https://img.shields.io/badge/AWS-SQS%20%7C%20S3%20%7C%20DynamoDB%20%7C%20Bedrock-orange)
![Whisper](https://img.shields.io/badge/Whisper-turbo-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Most teams record meetings and never revisit them. Finding a specific decision means listening to the whole thing again.

audio-to-rag solves this as a backend worker: given a recording in S3, it produces **speaker-attributed, theme-labelled chunks** with **512-dimensional vector embeddings** and a FAISS index, stored in S3, with the processing status in DynamoDB — ready for semantic search from any frontend or API.

### Scope

This repository contains the **processing worker only**. The trigger that launches it and the search / API layer that reads its output run in separate AWS Lambda functions that are **not part of this repository**.

---

## System architecture

```
Your app
   │
   │  1. Upload audio file
   ▼
S3 bucket ─────────────────────────────────────────────────┐
   │                                                       │
   │  2. Push SQS message { userId, reunionId, s3Audio }   │
   ▼                                                       │
SQS queue                                                  │
   │                                                       │
   │  3. Trigger (not in this repo) starts one task        │
   ▼                                                       │
ECS task — one-shot container (this repo)                  │
   │  ┌──────────────────────────────────────────────┐    │
   │  │  download audio from S3 ◄─────────────────────────┘
   │  │  ↓
   │  │  [processing pipeline — see below]
   │  │  ↓
   │  │  upload chunks + FAISS index to S3
   │  │  update DynamoDB status → TRAITÉ_INTELLIGENT
   │  └──────────────────────────────────────────────┘
   │
   ▼
S3 (chunks JSON + FAISS index) + DynamoDB (status)
   │
   └── consumed by your search / frontend layer (not in this repo)
```

With `ONE_SHOT_MODE=true` (the default) the container reads at most one message, processes it and exits: exit code `0` on success or when the queue is empty, non-zero when processing failed. Set `ONE_SHOT_MODE=false` to keep polling in a long-running worker.

---

## Processing pipeline

Inside the container, the audio goes through these steps, in this order:

```
S3 audio file
    │
    ├─► [1] ffmpeg              → convert to 16 kHz mono WAV
    │
    ├─► [2] Whisper turbo       → timestamped transcript segments
    │
    ├─► [3] Nova — Step 1       → clean transcription errors, remove filler words
    │                             (cleaned text is redistributed over the original
    │                              segments proportionally to their word counts)
    │
    ├─► [4] Nova — Step 2       → extract global themes, keywords, hot zones
    │
    ├─► [5] Pyannote           → speaker diarization (who spoke when)
    │
    ├─► [6] Merge               → each segment gets the speaker with the longest overlap
    │
    ├─► [7] Chunk + Nova Step 3 → rule-based chunking (word count, pauses, speaker
    │                             changes), then one Nova call per chunk to label it
    │                             (theme, keywords, summary) using the step 4 themes;
    │                             oversized chunks are split, short ones merged
    │
    ├─► [8] Titan Embed v2      → 512-dim vectors, enriched with Nova metadata
    │
    ├─► [9] FAISS IndexFlatIP   → cosine similarity index (L2-normalized inner product)
    │
    └─► S3 + DynamoDB           → persist everything, update processing status
```

---

## Technical decisions

### Why three separate Nova calls instead of one prompt?

A single mega-prompt asking Nova to simultaneously clean text, extract global themes, and produce semantic chunks would perform worse than three focused calls. The reason is both practical and architectural:

- **Step 1** outputs cleaned text. Steps 2 and 3 receive denoised input — no filler words, no transcription artifacts biasing the semantic analysis.
- **Step 2** produces a global thematic map of the full document. Step 3 receives this map as context, so each chunk is labelled with awareness of *all* themes across the recording, not just the local vicinity.
- Each call has its own `maxTokens` budget and `temperature=0.1` for determinism. A combined prompt would require guessing a single budget for three very different tasks.

The outputs are chained: `clean text → global themes → chunks labelled against those themes`. This is a deliberate pipeline, not three independent LLM calls. Note that step 3 is one Nova call *per chunk*, so the total number of Bedrock calls grows with the length of the recording.

### Why enriched embeddings?

Standard RAG encodes raw text into vectors. The problem: two semantically equivalent sentences with different vocabulary produce distant embeddings. A recording saying *"on reporte au T3"* won't match a query for *"deadline change"* on vocabulary alone.

Before calling Titan, each chunk's text is augmented with its Nova-derived metadata (`NOVA_ENHANCED_EMBEDDINGS=true`, the default):

```
"[Speaker_A1B2]: On reporte la release au T3 vu le planning."
  + [THEME: planning-release]
  + [TYPE: decision]
  + [SUBJECT: planning-release]
```

A `[SPEAKERS: ...]` tag is added only for chunks that Nova could not label. The stored chunk text is not modified — only the text sent to the embedding model. The embedding now carries **semantic intent** alongside lexical content: metadata-aware (contextually enriched) embeddings, applied here to the meeting domain. With `NOVA_ENHANCED_EMBEDDINGS=false`, the embeddings are computed on the plain chunk text.

### Why rule-based chunks labelled by an LLM instead of fixed windows?

Fixed-window chunking (e.g. every 30 seconds) is simple but lossy: a window may cut a sentence in half or glue two speakers' unrelated turns together. Both hurt retrieval quality.

| Fixed-window | This pipeline (default) |
|---|---|
| Arbitrary time boundaries | Variable length, 50–150 words targeted |
| May split mid-sentence | Closes on a sentence end followed by a pause (> 0.8 s) or a speaker change; hard cut at 150 words |
| No metadata | Theme, content type, keywords and one-line summary attached by Nova |
| Retrieval on raw text only | Retrieval on text + theme metadata |

**Default behaviour.** Chunk boundaries are decided by deterministic rules in `chunking.py` (word count, pauses, speaker changes), not by the LLM. Nova then analyses each chunk and attaches its metadata without changing the content. Finally, chunks above 150 words are split on segment boundaries and chunks under 50 words are merged into the most similar neighbour (cosine similarity above 0.5). Boundaries are therefore sentence- and turn-aligned, but they are not guaranteed to coincide with topic changes.

**Optional: `NOVA_DIRECT_CHUNKING=true`.** Nova receives all segments in one prompt and groups them by topic; those groups become the base chunks, which then go through the same labelling and size adjustment. Nova's answer is only accepted if it assigns every segment exactly once, in contiguous runs. On any failure (call error, unparsable JSON, incomplete grouping) the worker falls back to the rule-based chunker. This mode is off by default: it puts the whole transcript in a single prompt and has been exercised by unit tests only.

### Why one-shot ECS instead of a persistent worker?

A persistent worker container running 24/7 costs money even when idle and serializes all jobs. One-shot ECS solves both problems:

- **Cost**: the container exits after one message. Zero compute cost between jobs.
- **Isolation**: no shared state between messages. Each job is a clean process.
- **Parallelism**: each task handles one recording, so the trigger can launch as many tasks as there are messages.

The trade-off is cold start latency: the Whisper and Pyannote weights are not baked into the image, so they are downloaded and loaded on every start. For async audio processing this is acceptable; for real-time use cases it would not be.

### Resilience and failure handling

Some steps degrade gracefully, others fail the job on purpose:

| Situation | Behaviour |
|---|---|
| Whisper turbo fails | One retry with Whisper tiny (lower accuracy); if that fails too, the job fails |
| ffmpeg conversion fails | The job fails |
| Nova step 1 (cleaning) fails | The raw transcript is used |
| Nova step 2 (global themes) fails | The job fails |
| Diarization fails (or `HF_TOKEN` is missing) | All segments keep a single placeholder speaker |
| Nova labelling fails for one chunk | That chunk keeps its placeholder theme (most frequent words) |
| `NOVA_DIRECT_CHUNKING` grouping fails | Rule-based chunking is used |
| Nova JSON response looks truncated | Up to 5 continuation requests are appended. The detection is a heuristic (a response that fails to parse with an "Expecting ..." error); a response cut in the middle of a string is not detected |
| An embedding is missing when the index is built | The job fails, so that row *i* of the FAISS index always matches chunk *i* of the chunks JSON |
| S3 download/upload, the main DynamoDB update, Bedrock calls (Nova and Titan) | Retried on `ClientError` with exponential backoff (1 s, 2 s, ...) up to `MAX_RETRIES` attempts, on top of boto3's own retries |
| SQS calls and the secondary-table update | Single attempt (boto3's own retries only) |

**When a job fails**, the worker logs the error, writes `etat = "FAILED"` to both DynamoDB tables on a best-effort basis (a failure of that write is only logged), leaves the message in the queue, and — in one-shot mode — exits with a non-zero code. Whether the message is retried depends on your trigger and on the queue's redrive policy; a dead-letter queue is recommended.

### Security surface

- **SQS message validation**: every field (`userId`, `reunionId`, `s3Audio`) is validated against a strict regex before any processing. Malformed messages are discarded, not retried.
- **SQS visibility extension**: the visibility timeout is set to `VISIBILITY_TIMEOUT` when the message is received and again after the download. There is no heartbeat: a job that runs longer than `VISIBILITY_TIMEOUT` can be delivered a second time.
- **Path traversal**: audio files are restricted to `/tmp` via `Path.relative_to()` before any filesystem operation.
- **Speaker labels**: Pyannote speaker labels (`SPEAKER_00`, etc.) are replaced with random IDs (`Speaker_A1B2`) at diarization time. The pipeline never tries to identify speakers; names that are spoken aloud remain in the transcript text.
- **IAM least-privilege**: the included policy is scoped to specific resources — `sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility/GetQueueAttributes`, `s3:GetObject/PutObject`, `dynamodb:UpdateItem`, `bedrock:InvokeModel`.

---

## Tech stack

| Component | Technology | Role |
|-----------|------------|------|
| Transcription | OpenAI Whisper turbo | Speech-to-text with timestamps |
| Speaker diarization | pyannote.audio 4 (`speaker-diarization-3.1` pipeline) | Who spoke and when |
| Text cleaning | Amazon Nova Lite | Fix errors, remove filler words |
| Semantic analysis | Amazon Nova Lite | Extract themes, keywords, hot zones |
| Chunking | Rule-based (`chunking.py`) | Word-count, pause and speaker-change boundaries |
| Chunk labelling | Amazon Nova Lite | Theme, keywords, summary per chunk (optional topic grouping) |
| Embeddings | Amazon Titan Embed v2 (512d) | Dense vector representations |
| Vector search | FAISS IndexFlatIP | Cosine similarity via L2-normalized inner product |
| Queue | AWS SQS | Async job dispatch |
| Storage | AWS S3 | Audio input, chunks JSON, FAISS index |
| Status | AWS DynamoDB | Processing state and metadata |
| Inference | AWS Bedrock | Managed LLM and embedding APIs |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12+ | The pinned dependency set (`requirements.lock`) targets 3.12 |
| FFmpeg | `brew install ffmpeg` / `apt install ffmpeg` |
| AWS account | Bedrock must be enabled; activate **Nova Lite** and **Titan Embed v2** in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) |
| Hugging Face account | Free; accept [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) terms of use |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/SaturneAtom/audio-to-rag.git
cd audio-to-rag

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the pinned dependencies (see INSTALLATION.md for GPU notes)
pip install -r requirements.lock

# 4. Configure
cp .env.example .env
# Edit .env — see Configuration section below

# 5. Run (polls the SQS queue; needs AWS credentials)
python main.py
```

See [INSTALLATION.md](INSTALLATION.md) for AWS credentials, the Hugging Face token, GPU builds and troubleshooting.

### Tests

The unit tests cover the pure logic (message validation, path guard, speaker merge, chunking rules, split/merge, Nova JSON handling, retries). They stub the heavy dependencies and AWS clients, so they need neither torch nor AWS access:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The Docker image has its own smoke test, run in CI on every change to the Dockerfile or the lock file: it imports the real dependencies and the application modules, converts and decodes audio with ffmpeg and torchcodec, transcribes a short clip with Whisper `tiny` and queries a FAISS index. It makes no AWS call and needs no GPU:

```bash
docker build -t audio-to-rag .
docker run --rm audio-to-rag python docker/smoke_test.py
```

There is no end-to-end test: the full pipeline is only exercised against real AWS services.

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face token (Pyannote diarization) |
| `SQS_QUEUE_URL` | Yes | — | Full SQS queue URL |
| `S3_BUCKET_NAME` | Yes | — | S3 bucket for input audio and output artifacts |
| `DYNAMODB_TABLE_NAME` | Yes | — | Main DynamoDB table |
| `UNIQUE_MEETINGS_TABLE_NAME` | No | `UniqueMeetings` | Secondary table for frontend |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `MAX_RETRIES` | No | `3` | Attempts for boto3 and for the backoff wrapper |
| `VISIBILITY_TIMEOUT` | No | `3600` | SQS visibility timeout in seconds |
| `ONE_SHOT_MODE` | No | `true` | Exit after one message — `false` for a persistent worker |
| `NOVA_ENHANCED_EMBEDDINGS` | No | `true` | Enrich embeddings with Nova metadata before vectorizing |
| `NOVA_DIRECT_CHUNKING` | No | `false` | Let Nova group segments by topic; falls back to rule-based chunking on any failure |
| `WHISPER_LANGUAGE` | No | `fr` | Whisper transcription language |

The Nova prompts are written in French because the pipeline was built for French meetings; the generated themes, keywords and summaries, and several output keys, are therefore in French.

---

## IAM setup

`iam-policy.json` contains a minimal least-privilege policy. Replace the six placeholders before applying:

| Placeholder | Replace with |
|-------------|-------------|
| `REGION` | e.g. `us-east-1` |
| `ACCOUNT_ID` | Your 12-digit AWS account ID |
| `QUEUE_NAME` | Your SQS queue name |
| `BUCKET_NAME` | Your S3 bucket name |
| `TABLE_NAME` | Main DynamoDB table name |
| `UNIQUE_TABLE` | Secondary DynamoDB table name |

---

## SQS message format

```json
{
  "userId":    "user-abc123",
  "reunionId": "meeting-xyz456",
  "s3Audio":   "user-abc123/meeting-xyz456/audio.mp3"
}
```

| Field | Validation | Description |
|-------|-----------|-------------|
| `userId` | `[a-zA-Z0-9_-]{1,128}` | Owner of the recording |
| `reunionId` | `[a-zA-Z0-9_-]{1,128}` | Unique meeting identifier |
| `s3Audio` | S3 key, max 512 chars | Path to the audio file in `S3_BUCKET_NAME` |

Messages failing validation (or that are not valid JSON) are deleted from the queue immediately — they would fail on every retry. No status is written for them.

---

## S3 output layout

```
{userId}/{reunionId}/
├── chunks/
│   └── {reunionId}.json                      # topic chunks + speaker metadata
├── index/
│   └── {reunionId}.faiss                     # FAISS vector index
└── metadata/
    └── {reunionId}_global_analysis.json      # themes, keywords, hot zones
```

### Chunk structure

Abridged example (each chunk also carries `reasoning`, `is_topic_transition`, `transition_score`, `cluster_id`, and more `nova_chunking` fields):

```json
{
  "metadata": {
    "user_id": "user-abc123",
    "reunion_id": "meeting-xyz456",
    "total_chunks": 12,
    "creation_date": "2024-01-15T10:30:00",
    "chunking_strategy": "semantic_intelligent"
  },
  "chunks": [
    {
      "id": 1,
      "start_time": 0.0,
      "end_time": 47.3,
      "text": "[Speaker_A1B2]: On est d'accord pour décaler la release au T3 vu le planning actuel.",
      "theme": "planning-release",
      "speakers_info": [
        { "name": "Speaker_A1B2", "speaking_time": 32.1, "dominance_score": 0.82, "turns": 3 }
      ],
      "nova_chunking": {
        "theme_principal": "planning-release",
        "type_contenu": "decision",
        "mots_cles_chunk": ["release", "T3", "planning"],
        "resume_chunk": "L'équipe décide de reporter la release au T3.",
        "coherence_score": 0.91
      }
    }
  ]
}
```

> Speaker IDs (e.g. `Speaker_A1B2`) are randomly generated per recording — consistent within a meeting, not across meetings.

---

## DynamoDB status

The worker updates the item with key `{ userId, reunionId }` in `DYNAMODB_TABLE_NAME`, and the `etat` attribute of the same key in `UNIQUE_MEETINGS_TABLE_NAME`.

| Attribute | Written when | Value |
|-----------|--------------|-------|
| `etat` | Success | `TRAITÉ_INTELLIGENT` |
| `etat` | Failure (best effort) | `FAILED` |
| `s3Chunks` | Success | S3 key of the chunks JSON |
| `s3Index` | Success | S3 key of the FAISS index |
| `processedAt` | Success | ISO 8601 timestamp |

---

## Deploy on ECS

```bash
docker build -t audio-to-rag .
docker tag audio-to-rag:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/audio-to-rag:latest
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/audio-to-rag:latest
```

The image is built from `python:3.12-slim`, installs ffmpeg and the fully pinned `requirements.lock`, and runs as a non-root user. The PyTorch wheel bundles its own CUDA runtime, so the same image runs on CPU and on GPU.

Run it as an **ECS task** started once per SQS message by your own trigger (for example a Lambda function or an EventBridge Pipe calling `RunTask`); the trigger is not part of this repository. With `ONE_SHOT_MODE=true` the task exits when its message is done, so nothing runs between jobs.

- **CPU**: Fargate is an option for CPU-only runs. Fargate has no GPU, so Whisper and Pyannote run on CPU there (slower; size the task memory for the models).
- **GPU**: Whisper and Pyannote are significantly faster on GPU. Run the task on ECS with **EC2 GPU instances** (`g4dn.xlarge` or larger). The pinned PyTorch (2.11) ships a CUDA 13 runtime, which needs an NVIDIA driver from the 580 series or later on the host; use a recent ECS GPU-optimized AMI.

Helper scripts for the EC2 GPU option:
- `simple-ecs.sh` — configure the ECS agent on a GPU EC2 instance (ECS-optimized AMI)
- `user-data` — EC2 user-data bootstrap that installs and starts the ECS agent

---

## Module reference

| File | Responsibility |
|------|----------------|
| `main.py` | Entry point — logs the configuration, starts the worker, sets the exit code |
| `worker.py` | SQS polling, input validation, pipeline orchestration, DynamoDB status |
| `audio.py` | WAV conversion, Whisper transcription, Pyannote diarization, speaker anonymization |
| `nova.py` | 3-step Nova pipeline, JSON pagination handling, chunk labelling, optional Nova grouping |
| `embeddings.py` | Titan embedding generation, metadata enrichment, FAISS index creation |
| `chunking.py` | Rule-based chunking, chunk splitting/merging, cosine similarity |
| `storage.py` | S3 uploads of the chunks JSON and FAISS index |
| `models.py` | `Chunk` dataclass, custom JSON encoder |
| `aws_utils.py` | Exponential backoff retry helper, S3 / DynamoDB wrappers, SQS visibility extension |
| `config.py` | Environment loading and validation, constants, AWS client initialization, device detection |
| `tests/` | Unit tests for the pure logic (heavy dependencies stubbed) |

---

## License

MIT
