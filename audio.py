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


def get_audio_duration(audio_file: str) -> float:
    """Récupère la durée d'un fichier audio"""
    try:
        y, sr = librosa.load(audio_file)
        return len(y) / sr
    except Exception as e:
        logger.error(f"Erreur durée audio: {e}")
        return 0.0


def convert_to_wav(input_file: str) -> str:
    """Convertit automatiquement tout fichier audio en WAV optimisé"""
    tmp_dir = Path('/tmp').resolve()
    resolved = Path(input_file).resolve()
    try:
        resolved.relative_to(tmp_dir)
    except ValueError:
        raise ValueError(f"Audio file must be inside /tmp: {input_file}")

    output_file = str(resolved.with_suffix('.wav'))

    if Path(output_file).exists():
        logger.info(f"✅ Fichier WAV existant utilisé: {output_file}")
        return output_file

    try:
        logger.info(f"🔄 Conversion en WAV optimisé...")
        subprocess.run([
            'ffmpeg', '-i', str(resolved),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y', output_file
        ], check=True, capture_output=True)
        logger.info(f"✅ Converti en WAV: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur conversion: {e}")
        return str(resolved)


def generate_random_speaker_id() -> str:
    """Génère un ID de speaker aléatoire (ex: Speaker_A1B2)"""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=2))
    return f"Speaker_{letters[0]}{numbers[0]}{letters[1]}{numbers[1]}"


def create_speaker_mapping(original_speakers: List[str]) -> Dict[str, str]:
    """Mappe les speakers originaux vers des IDs aléatoires anonymes"""
    speaker_mapping = {}
    used_ids = set()

    for original_speaker in original_speakers:
        new_id = generate_random_speaker_id()
        while new_id in used_ids:
            new_id = generate_random_speaker_id()
        speaker_mapping[original_speaker] = new_id
        used_ids.add(new_id)

    return speaker_mapping


def perform_diarization(audio_file: str) -> Optional[List[Dict]]:
    """Effectue la diarisation des locuteurs avec pyannote"""
    try:
        HF_TOKEN = os.environ.get('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN est obligatoire pour la diarisation. "
                "Obtenez un token sur https://huggingface.co/settings/tokens "
                "et acceptez les conditions d'utilisation de pyannote/speaker-diarization-3.1."
            )

        logger.info("🔄 Initialisation du modèle de diarisation...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

        if DEVICE == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("✅ Pipeline diarisation configuré pour GPU")

        duration = get_audio_duration(audio_file)
        logger.info(f"⏱️ Durée audio: {duration:.1f}s")

        logger.info("🔄 Début de l'analyse des locuteurs...")
        diarization = pipeline(audio_file)

        segments = []
        original_speakers = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            original_speakers.add(speaker)

        speaker_mapping = create_speaker_mapping(list(original_speakers))
        for segment in segments:
            segment["speaker"] = speaker_mapping[segment["speaker"]]

        new_speakers = set([seg["speaker"] for seg in segments])
        logger.info(f"👥 {len(new_speakers)} locuteurs détectés avec IDs aléatoires: {', '.join(sorted(new_speakers))}")

        return segments

    except Exception as e:
        logger.error(f"❌ Erreur diarisation: {e}")
        return None


def transcribe_audio(audio_file: str) -> List[Dict]:
    """Transcrit l'audio avec Whisper turbo"""
    try:
        logger.info("🔄 Chargement du modèle Whisper...")
        model = load_model("turbo", device=DEVICE)
        logger.info("✅ Modèle Whisper chargé")

        duration = get_audio_duration(audio_file)
        logger.info(f"⏱️ Durée audio: {duration:.1f}s")

        logger.info("🔄 Transcription en cours...")
        result = model.transcribe(audio_file, language=WHISPER_LANGUAGE, task="transcribe")

        default_speaker_id = generate_random_speaker_id()
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "speaker": default_speaker_id
            })

        logger.info(f"✅ {len(segments)} segments transcrits")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segments

    except Exception as e:
        logger.error(f"❌ Erreur transcription: {e}")
        try:
            logger.info("🔄 Tentative avec modèle tiny...")
            model = load_model("tiny", device=DEVICE)
            result = model.transcribe(audio_file, language=WHISPER_LANGUAGE, task="transcribe")

            default_speaker_id = generate_random_speaker_id()
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": default_speaker_id
                })

            logger.info(f"✅ {len(segments)} segments transcrits (tiny)")
            del model
            return segments

        except Exception as e2:
            logger.error(f"❌ Erreur fallback: {e2}")
            return []


def merge_transcription_diarization(transcript_segments: List[Dict],
                                    diarization_segments: List[Dict]) -> List[Dict]:
    """Fusionne transcription Whisper et diarisation pyannote"""
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
