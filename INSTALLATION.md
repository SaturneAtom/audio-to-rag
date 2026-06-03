# Installation guide

Detailed setup instructions for AudioPipeline. For a quick start, see the [README](README.md).

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

# Whisper must be installed from source
pip install git+https://github.com/openai/whisper.git

# All other dependencies
pip install -r requirements.txt

# French NLP model (used by the fallback chunker)
python -m spacy download fr_core_news_md
```

Verify the install (no AWS calls made):

```bash
python -c "import whisper, librosa, faiss, boto3, pyannote.audio; print('OK')"
python -m py_compile *.py
```

---

## AWS setup

### Credentials

**Do not store `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` or in code.**

| Context | Recommended approach |
|---------|----------------------|
| ECS / Fargate | Attach `iam-policy.json` to the ECS task role — no credentials needed in the container |
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

`.env` is listed in `.gitignore` — it will not be committed.

---

## GPU support (optional)

Whisper turbo and Pyannote run significantly faster on GPU. The pipeline detects CUDA automatically.

```bash
# Replace the CPU torch build with a CUDA-enabled one
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Troubleshooting

**`ValueError: 'HF_TOKEN' is required`**
→ Your `.env` is missing or the token is blank. Check with:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('HF_TOKEN', 'NOT SET'))"
```

**`ffmpeg: command not found`**
→ FFmpeg is not installed or not on your PATH. See system requirements above.

**Bedrock `AccessDeniedException`**
→ The model is not enabled in your region, or your IAM role/user lacks `bedrock:InvokeModel` permission. Check the Bedrock console and `iam-policy.json`.

**`botocore.exceptions.NoRegionError`**
→ Set `AWS_REGION` in `.env` or run `aws configure`.
