import os
import logging
import subprocess
import random
import string
import torch
import librosa
from pathlib import Path
from typing import List, Dict, Optional
from whisper import load_model
from pyannote.audio import Pipeline
from config import DEVICE, WHISPER_LANGUAGE

logger = logging.getLogger(__name__)


WHISPER_MODEL = "turbo"
WHISPER_FALLBACK_MODEL = "tiny"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def get_audio_duration(audio_file: str) -> float:
    """Return the duration of an audio file in seconds (0.0 if it cannot be read)."""
    try:
        y, sr = librosa.load(audio_file)
        return len(y) / sr
    except Exception as e:
        logger.error(f"Could not read audio duration: {e}")
        return 0.0


def convert_to_wav(input_file: str) -> str:
    """Convert an audio file under /tmp to 16 kHz mono WAV and return the WAV path.

    Raises ValueError if the file is outside /tmp and RuntimeError if ffmpeg fails.
    """
    tmp_dir = Path('/tmp').resolve()
    resolved = Path(input_file).resolve()
    try:
        resolved.relative_to(tmp_dir)
    except ValueError:
        raise ValueError(f"Audio file must be inside /tmp: {input_file}")

    output_file = str(resolved.with_suffix('.wav'))

    if Path(output_file).exists():
        logger.info(f"Using existing WAV file: {output_file}")
        return output_file

    logger.info("Converting to 16 kHz mono WAV...")
    try:
        subprocess.run([
            'ffmpeg', '-i', str(resolved),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y', output_file
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg conversion failed (exit code {e.returncode}): {stderr[-500:]}"
        ) from e

    logger.info(f"Converted to WAV: {output_file}")
    return output_file


def generate_random_speaker_id() -> str:
    """Generate a random speaker ID (e.g. Speaker_A1B2)."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=2))
    return f"Speaker_{letters[0]}{numbers[0]}{letters[1]}{numbers[1]}"


def create_speaker_mapping(original_speakers: List[str]) -> Dict[str, str]:
    """Map each original speaker label to a unique random ID."""
    speaker_mapping = {}
    used_ids = set()

    for original_speaker in original_speakers:
        new_id = generate_random_speaker_id()
        while new_id in used_ids:
            new_id = generate_random_speaker_id()
        speaker_mapping[original_speaker] = new_id
        used_ids.add(new_id)

    return speaker_mapping


def speaker_annotation(output):
    """Return the speaker Annotation from a pyannote pipeline output.

    pyannote.audio 4.x returns a DiarizeOutput whose `speaker_diarization`
    attribute holds the Annotation; legacy pipelines return the Annotation itself.
    """
    return getattr(output, "speaker_diarization", output)


def perform_diarization(audio_file: str) -> Optional[List[Dict]]:
    """Run pyannote speaker diarization. Returns None if diarization fails."""
    try:
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise ValueError(
                "HF_TOKEN is required for diarization. "
                "Create a token at https://huggingface.co/settings/tokens "
                f"and accept the terms of use of {DIARIZATION_MODEL}."
            )

        logger.info("Loading diarization model...")
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

        if DEVICE == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("Diarization pipeline moved to GPU")

        logger.info("Running speaker diarization...")
        diarization = speaker_annotation(pipeline(audio_file))

        segments = []
        original_speakers = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            original_speakers.add(speaker)

        speaker_mapping = create_speaker_mapping(list(original_speakers))
        for segment in segments:
            segment["speaker"] = speaker_mapping[segment["speaker"]]

        new_speakers = set(seg["speaker"] for seg in segments)
        logger.info(f"{len(new_speakers)} speakers detected, random IDs: {', '.join(sorted(new_speakers))}")

        return segments

    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        return None


def _transcribe_with_model(audio_file: str, model_name: str) -> List[Dict]:
    """Transcribe with one Whisper model; every segment gets the same placeholder speaker."""
    logger.info(f"Loading Whisper model '{model_name}'...")
    model = load_model(model_name, device=DEVICE)

    logger.info("Transcribing...")
    result = model.transcribe(audio_file, language=WHISPER_LANGUAGE, task="transcribe")

    default_speaker_id = generate_random_speaker_id()
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "speaker": default_speaker_id
        }
        for segment in result["segments"]
    ]
    logger.info(f"{len(segments)} segments transcribed ({model_name})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments


def transcribe_audio(audio_file: str) -> List[Dict]:
    """Transcribe with Whisper turbo, retrying once with Whisper tiny.

    Returns an empty list if both models fail; the caller treats that as a failed job.
    """
    logger.info(f"Audio duration: {get_audio_duration(audio_file):.1f}s")

    for model_name in (WHISPER_MODEL, WHISPER_FALLBACK_MODEL):
        try:
            return _transcribe_with_model(audio_file, model_name)
        except Exception as e:
            logger.error(f"Transcription with Whisper '{model_name}' failed: {e}", exc_info=True)

    return []


def merge_transcription_diarization(transcript_segments: List[Dict],
                                    diarization_segments: List[Dict]) -> List[Dict]:
    """Assign to each transcript segment the speaker with the longest overlap.

    Segments without any overlapping diarization turn keep their current speaker.
    """
    if not diarization_segments:
        return transcript_segments

    merged_segments = []

    for t_seg in transcript_segments:
        t_start, t_end = t_seg["start"], t_seg["end"]
        speaker_times = {}

        for d_seg in diarization_segments:
            d_start, d_end = d_seg["start"], d_seg["end"]
            intersection_start = max(t_start, d_start)
            intersection_end = min(t_end, d_end)

            if intersection_start < intersection_end:
                intersection_duration = intersection_end - intersection_start
                speaker = d_seg["speaker"]
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += intersection_duration

        if speaker_times:
            majority_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            t_seg["speaker"] = majority_speaker

        merged_segments.append(t_seg)

    return merged_segments
