# audio-to-rag

> An async, event-driven backend pipeline that transforms raw meeting recordings into a structured, speaker-aware, semantically searchable knowledge base — fully automated on AWS.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AWS](https://img.shields.io/badge/AWS-SQS%20%7C%20S3%20%7C%20DynamoDB%20%7C%20Bedrock-orange)
![Whisper](https://img.shields.io/badge/Whisper-turbo-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Most teams record meetings and never revisit them. Finding a specific decision means listening to the whole thing again.

AudioPipeline solves this as a backend service: when a recording is uploaded, it automatically produces **speaker-attributed topic chunks** with **512-dimensional vector embeddings**, stored in S3 and DynamoDB, ready for semantic search from any frontend or API.

---

## System architecture

```
Your app
   │
   │  1. Upload audio file
   ▼
S3 bucket ─────────────────────────────────────────────────┐
   │                                                       │
   │  2. Push SQS message { userId, reunionId, s3Key }     │
   ▼                                                       │
SQS queue                                                  │
   │                                                       │
   │  3. Trigger (EventBridge Pipes or Lambda)             │
   ▼                                                       │
ECS Fargate task — one-shot container (this repo)          │
   │  ┌──────────────────────────────────────────────┐    │
   │  │  download audio from S3 ◄─────────────────────────┘
   │  │  ↓
   │  │  [processing pipeline — see below]
   │  │  ↓
   │  │  upload chunks + FAISS index to S3
   │  │  update DynamoDB status → PROCESSED
   │  └──────────────────────────────────────────────┘
   │
   ▼
S3 (chunks JSON + FAISS index) + DynamoDB (status)
   │
   └── consumed by your search / frontend layer
```

The container exits after processing one message (`ONE_SHOT_MODE=true`). No idle cost. A new task is launched per message.

---

## Processing pipeline

Inside the container, the audio goes through 9 sequential steps:

```
S3 audio file
    │
    ├─► [1] ffmpeg              → convert to 16kHz mono WAV
    │
    ├─► [2] Whisper turbo       → timestamped transcript segments
    │
    ├─► [3] Pyannote 3.1        → speaker diarization (who spoke when)
    │
    ├─► [4] Merge               → segments enriched with speaker attribution
    │                             (majority-vote per segment on overlap)
    │
    ├─► [5] Nova — Step 1       → clean transcription errors, remove filler words
    │
    ├─► [6] Nova — Step 2       → extract global themes, keywords, hot zones
    │
    ├─► [7] Nova — Step 3       → semantic rechunking guided by step 6 output
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
- **Step 2** produces a global thematic map of the full document. Step 3 receives this map as context, so it can group segments by topic with awareness of *all* themes across the recording, not just the local vicinity.
- Each call has its own `maxTokens` budget and `temperature=0.1` for determinism. A combined prompt would require guessing a single budget for three very different tasks.

The outputs are chained: `clean text → global themes → rechunked segments guided by those themes`. This is a deliberate pipeline, not three independent LLM calls.

### Why enriched embeddings?

Standard RAG encodes raw text into vectors. The problem: two semantically equivalent sentences with different vocabulary produce distant embeddings. A recording saying *"on reporte au T3"* won't match a query for *"deadline change"* on vocabulary alone.

Before calling Titan, each chunk's text is augmented with its Nova-derived metadata:

```
"On reporte la release au T3 vu le planning."
  + [THEME: planning-release]
  + [TYPE: decision]
  + [SPEAKERS: 2]
```

The embedding now carries **semantic intent** alongside lexical content. This is a standard technique in production RAG systems — metadata-aware or contextually enriched embeddings — applied here to the meeting domain.

### Why semantic chunking instead of fixed windows?

Fixed-window chunking (e.g. every 30 seconds) is simple but lossy: a window may span multiple unrelated topics, or split a single decision across two chunks. Both hurt retrieval quality.

| Fixed-window | Semantic chunking (this pipeline) |
|---|---|
| Arbitrary time boundaries | Variable-length, topic-aligned |
| May split a decision mid-sentence | Each chunk ends at a natural topic break |
| No theme metadata | Theme, content type, keywords attached |
| Retrieval matches by proximity | Retrieval matches by idea |

Nova (step 7) reads all segments and groups them by topic. If Nova returns nothing, a fallback algorithm uses temporal gaps and speaker changes as chunk boundaries.

### Why one-shot ECS instead of a persistent worker?

A persistent worker container running 24/7 costs money even when idle and serializes all jobs. One-shot ECS solves both problems:

- **Cost**: the container exits after one message. Zero compute cost between jobs.
- **Isolation**: no shared state between messages. Each job is a clean process.
- **Parallelism**: 10 messages in the queue = 10 independent containers launched simultaneously, each processing one recording.

The trade-off is cold start latency (~30–60s for model loading). For async audio processing this is acceptable; for real-time use cases it would not be.

### Resilience: three levels of fallback

The pipeline is designed to degrade gracefully rather than fail hard:

1. **Whisper turbo fails** → automatic retry with Whisper tiny (lower accuracy but always available)
2. **Nova rechunking returns nothing** → fallback to algorithmic chunking (temporal gaps + speaker changes + word count thresholds)
3. **Nova JSON response truncated** → automatic pagination: the pipeline detects incomplete JSON and requests continuations, up to 5 iterations
4. **All AWS operations** → exponential backoff retry wrapper with configurable max attempts

### Security surface

- **SQS message validation**: every field (`userId`, `reunionId`, `s3Audio`) is validated against a strict regex before any processing. Malformed messages are discarded, not retried.
- **SQS visibility extension**: the visibility timeout is extended during processing to prevent a second consumer from picking up the same message if the job is slow.
- **Path traversal**: audio files are restricted to `/tmp` via `Path.relative_to()` before any filesystem operation.
- **Speaker anonymization**: Pyannote speaker labels (`SPEAKER_00`, etc.) are replaced with random IDs (`Speaker_A1B2`) at diarization time. No real names are ever stored.
- **IAM least-privilege**: the included policy grants exactly the permissions used — `sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility`, `s3:GetObject/PutObject`, `dynamodb:UpdateItem`, `bedrock:InvokeModel` on specific resources only.

---

## Tech stack

| Component | Technology | Role |
|-----------|------------|------|
| Transcription | OpenAI Whisper turbo | Speech-to-text with timestamps |
| Speaker diarization | Pyannote Audio 3.1 | Who spoke and when |
| Text cleaning | Amazon Nova Lite | Fix errors, remove filler words |
| Semantic analysis | Amazon Nova Lite | Extract themes, keywords, hot zones |
| Semantic chunking | Amazon Nova Lite | Group segments by topic |
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
| Python 3.10+ | |
| FFmpeg | `brew install ffmpeg` / `apt install ffmpeg` |
| AWS account | Bedrock must be enabled; activate **Nova Lite** and **Titan Embed v2** in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) |
| Hugging Face account | Free; accept [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) terms of use |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/SaturneAtom/AudioPipeline.git
cd AudioPipeline

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install git+https://github.com/openai/whisper.git
pip install -r requirements.txt
python -m spacy download fr_core_news_md

# 4. Configure
cp .env.example .env
# Edit .env — see Configuration section below

# 5. Run
python main.py
```

> **Verify install without AWS calls:**
> ```bash
> python -m py_compile *.py
> ```

---

## Configuration

Copy `.env.example` to `.env`. **Never commit `.env`** — it is in `.gitignore`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face token (Pyannote diarization) |
| `SQS_QUEUE_URL` | Yes | — | Full SQS queue URL |
| `S3_BUCKET_NAME` | Yes | — | S3 bucket for input audio and output artifacts |
| `DYNAMODB_TABLE_NAME` | Yes | — | Main DynamoDB table |
| `UNIQUE_MEETINGS_TABLE_NAME` | No | `UniqueMeetings` | Secondary table for frontend |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `MAX_RETRIES` | No | `3` | Boto3 retry attempts |
| `VISIBILITY_TIMEOUT` | No | `3600` | SQS visibility timeout in seconds |
| `ONE_SHOT_MODE` | No | `true` | Exit after one message — `false` for a persistent worker |
| `NOVA_ENHANCED_EMBEDDINGS` | No | `true` | Enrich embeddings with Nova metadata before vectorizing |
| `WHISPER_LANGUAGE` | No | `fr` | Whisper transcription language — any [supported code](https://github.com/openai/whisper#available-models-and-languages) (`en`, `es`, `de`, …) |

### AWS credentials

**Do not hardcode `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.**

Boto3 resolves credentials in order:

1. **IAM role** — recommended for ECS (attach `iam-policy.json` to the task role; no credentials in the container)
2. Environment variables — local dev only, never committed
3. `~/.aws/credentials` — local dev via `aws configure`

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

```bash
aws iam create-policy \
  --policy-name AudioPipelinePolicy \
  --policy-document file://iam-policy.json

aws iam attach-role-policy \
  --role-name <your-ecs-task-role> \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/AudioPipelinePolicy
```

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

Messages failing validation are deleted from the queue immediately — they would fail on every retry.

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

> Speaker IDs (e.g. `Speaker_A1B2`) are randomly generated per recording — consistent within a meeting, not across meetings. No real names are stored.

---

## Deploy on ECS

```bash
docker build -t audio-pipeline .
docker tag audio-pipeline:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/audio-pipeline:latest
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/audio-pipeline:latest
```

Configure an **EventBridge Pipe** (or Lambda trigger) from the SQS queue to launch a new ECS task per message. With `ONE_SHOT_MODE=true` the cluster scales to zero between jobs.

For GPU inference (Whisper + Pyannote are significantly faster on GPU), the base image is `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`. Use `g4dn.xlarge` or larger.

Helper scripts:
- `simple-ecs.sh` — configure the ECS agent on a GPU EC2 instance
- `user-data` — EC2 user-data bootstrap for ECS-optimized AMI

---

## Module reference

| File | Responsibility |
|------|----------------|
| `main.py` | Entry point — validates config, starts worker |
| `worker.py` | SQS polling, input validation, pipeline orchestration |
| `audio.py` | WAV conversion, Whisper transcription, Pyannote diarization, speaker anonymization |
| `nova.py` | 3-step Nova pipeline, JSON pagination handling, chunk enrichment |
| `embeddings.py` | Titan embedding generation, metadata enrichment, FAISS index creation |
| `chunking.py` | Fallback chunking, chunk splitting/merging, cosine similarity |
| `storage.py` | S3 uploads (chunks JSON, FAISS index, metadata), DynamoDB writes |
| `models.py` | `Chunk` dataclass, custom JSON encoder |
| `aws_utils.py` | Exponential backoff retry wrappers for S3, DynamoDB, SQS |
| `config.py` | Environment loading, AWS client initialization, device detection |

---

## Contributing

1. Fork and create a branch from `main`
2. Make your changes
3. `python -m py_compile *.py`
4. Open a pull request — describe what changed and why

---

## License

MIT
