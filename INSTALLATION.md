# Installation guide

Detailed setup instructions for audio-to-rag. For a quick start and the full list of environment variables, see the [README](README.md).

---

## System requirements

- Python 3.10+
- FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

## Python environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# PyTorch is not listed in requirements.txt (the Docker base image provides it)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies
pip install -r requirements.txt

# Whisper must be installed from source
pip install git+https://github.com/openai/whisper.git
```

The Docker image performs the same steps (plus `apt-get install ffmpeg`), so none of this is needed when you only build and deploy the container.

Verify the install (no AWS calls made):

```bash
python -c "import whisper, librosa, faiss, boto3, pyannote.audio; print('OK')"
python -m py_compile *.py
```

### Tests and linting

The unit tests stub the heavy dependencies and the AWS clients, so they run without torch, Whisper or AWS access:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

---

## AWS setup

### Credentials

**Do not store `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` or in code.**

| Context | Recommended approach |
|---------|----------------------|
| ECS (Fargate or EC2) | Attach `iam-policy.json` to the ECS task role — no credentials needed in the container |
| Local development | Run `aws configure` once; boto3 reads `~/.aws/credentials` automatically |
| CI / CD | Use OIDC federation or short-lived assumed-role tokens |

### Enable Bedrock models

In the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess), request access to:

- `amazon.nova-lite-v1:0`
- `amazon.titan-embed-text-v2:0`

Both are available in `us-east-1` and several other regions.

---

## Hugging Face token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and generate a read token
3. Accept the terms of use for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. Add the token to your `.env`:

```
HF_TOKEN=hf_...
```

---

## Environment file

```bash
cp .env.example .env
```

Edit `.env` with your values. Required variables:

```
HF_TOKEN=hf_...
SQS_QUEUE_URL=https://sqs.REGION.amazonaws.com/ACCOUNT_ID/QUEUE_NAME
S3_BUCKET_NAME=your-bucket-name
DYNAMODB_TABLE_NAME=your-table-name
```

Optional variables (`NOVA_DIRECT_CHUNKING`, `NOVA_ENHANCED_EMBEDDINGS`, `ONE_SHOT_MODE`, ...) are described in the [README](README.md#configuration).

`.env` is listed in `.gitignore` — it will not be committed.

---

## GPU support (optional)

Whisper turbo and Pyannote run significantly faster on GPU. The pipeline detects CUDA automatically.

```bash
# Replace the CPU torch build with a CUDA-enabled one
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

On AWS, GPU inference requires ECS on EC2 GPU instances (for example `g4dn.xlarge`); Fargate does not provide GPUs.

Verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Troubleshooting

**`Diarization failed: HF_TOKEN is required for diarization`** (logged; every segment then keeps a single placeholder speaker)
→ Your `.env` is missing or the token is blank. Check with:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('set' if os.getenv('HF_TOKEN') else 'NOT SET')"
```

**`ValueError: Environment variable '...' is required and is not set`**
→ `S3_BUCKET_NAME`, `DYNAMODB_TABLE_NAME` and `SQS_QUEUE_URL` are mandatory. Copy `.env.example` to `.env` and fill them in.

**`RuntimeError: ffmpeg conversion failed`**
→ ffmpeg could not decode the input file; the message contains the end of ffmpeg's output. The job is marked `FAILED`.

**The container exits with a non-zero code**
→ In one-shot mode this means the job failed. The item's `etat` attribute is set to `FAILED` and the SQS message is left in the queue; check the container logs for the traceback.

**`ffmpeg: command not found`**
→ FFmpeg is not installed or not on your PATH. See system requirements above.

**Bedrock `AccessDeniedException`**
→ The model is not enabled in your region, or your IAM role/user lacks `bedrock:InvokeModel` permission. Check the Bedrock console and `iam-policy.json`.
