from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import threading
from faster_whisper import WhisperModel, BatchedInferencePipeline
import json
from datetime import datetime
import os
import subprocess
from backend.mel_from_transcript import add_mel_paths_to_transcript
from backend.similarity_analyzer import add_similarity_analysis_to_transcript, add_similarity_analysis_with_reference

app = FastAPI()

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

TEMP_TRANSCRIPTIONS_DIR = Path("temp_transcriptions")
TEMP_TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

# Try to use CUDA if available, else fallback to CPU
try:
    WHISPER_MODEL = WhisperModel("turbo", device="cuda", compute_type="float16")
except Exception:
    WHISPER_MODEL = WhisperModel("turbo", device="cpu")

BATCHED_MODEL = BatchedInferencePipeline(model=WHISPER_MODEL)

# In-memory job store
jobs = {}

@app.post("/api/upload")
async def upload_audio(
    reference_audio: UploadFile = File(...),
    long_audio: UploadFile = File(...)
):
    try:
        print(f"[INFO] Starting upload for reference: {reference_audio.filename}, long: {long_audio.filename}")
        job_id = str(uuid.uuid4())
        
        # Save reference audio (500ms)
        ref_ext = Path(reference_audio.filename).suffix
        ref_save_path = UPLOADS_DIR / f"ref_{job_id}{ref_ext}"
        print(f"[INFO] Saving reference file to: {ref_save_path}")
        with ref_save_path.open("wb") as f:
            shutil.copyfileobj(reference_audio.file, f)

        # Save long audio
        long_ext = Path(long_audio.filename).suffix
        long_save_path = UPLOADS_DIR / f"long_{job_id}{long_ext}"
        print(f"[INFO] Saving long file to: {long_save_path}")
        with long_save_path.open("wb") as f:
            shutil.copyfileobj(long_audio.file, f)

        print(f"[INFO] Files saved successfully, creating job: {job_id}")
        jobs[job_id] = {"status": "processing", "result": None}

        print(f"[INFO] Starting processing thread for job: {job_id}")
        thread = threading.Thread(target=process_job, args=(job_id, ref_save_path, long_save_path))
        thread.start()

        print(f"[INFO] Job {job_id} queued successfully")
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job

def process_job(job_id: str, ref_audio_path: Path, long_audio_path: Path):
    try:
        print(f"[INFO] Starting processing for job {job_id}")
        print(f"[INFO] Reference audio: {ref_audio_path}")
        print(f"[INFO] Long audio: {long_audio_path}")
        
        # Add these two lines:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        print(f"[INFO] Project root: {project_root}")
        
        # Step 1: Create reference spectrogram (500ms)
        print(f"[INFO] Step 1: Creating reference spectrogram...")
        ref_spec_path = create_reference_spectrogram(ref_audio_path, job_id)
        print(f"[INFO] Reference spectrogram created: {ref_spec_path}")
        
        # Step 2: Whisper transcription of long audio
        print(f"[INFO] Step 2: Starting Whisper transcription of long audio...")
        whisper_result, transcription_path = run_whisper(long_audio_path)
        print(f"[INFO] Whisper transcription completed: {transcription_path}")
        
        # Step 3: Mel cutting for long audio
        print(f"[INFO] Step 3: Starting mel cutting for long audio...")
        mel_json_path = None
        try:
            mel_json_path = add_mel_paths_to_transcript(str(transcription_path), str(long_audio_path))
            print(f"[INFO] Mel cutting completed: {mel_json_path}")
        except Exception as e:
            print(f"[ERROR] Mel cutting failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Similarity analysis with reference
        print(f"[INFO] Step 4: Starting similarity analysis with reference...")
        try:
            model_path = str(project_root / "experiments/exp_01_baseline/checkpoints/model_best_epoch_23_20250703_153314.pt")
            config_path = str(project_root / "experiments/exp_01_baseline/config.yaml")
            print(f"[INFO] Model path: {model_path}")
            print(f"[INFO] Config path: {config_path}")
            
            similarity_analysis = add_similarity_analysis_with_reference(
                str(transcription_path), 
                str(long_audio_path), 
                str(ref_spec_path),
                model_path, 
                config_path
            )
            print(f"[INFO] Similarity analysis completed")
        except Exception as e:
            print(f"[ERROR] Similarity analysis failed: {e}")
            import traceback
            traceback.print_exc()
            similarity_analysis = {"error": str(e)}
        
        # Step 5: Return complete results
        print(f"[INFO] Step 5: Finalizing results...")
        jobs[job_id] = {
            "status": "done",
            "result": {
                "whisper": whisper_result,
                "whisper_json_path": str(transcription_path),
                "mel_json_path": mel_json_path,
                "similarity_analysis": similarity_analysis,
                "reference_spec_path": str(ref_spec_path)
            }
        }
        print(f"[INFO] Job {job_id} completed successfully")
    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "error": str(e)}

def run_whisper(audio_path: Path):
    print(f"[DEBUG] Running Batched Whisper on {audio_path}")
    segments, info = BATCHED_MODEL.transcribe(
        str(audio_path), batch_size=16, word_timestamps=True
    )
    words_data = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words_data.append({
                    "word": word.word,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "confidence": round(word.probability, 3) if hasattr(word, 'probability') else None
                })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TEMP_TRANSCRIPTIONS_DIR / f"transcription_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "language": info.language,
                "language_probability": info.language_probability,
                "timestamp": timestamp
            },
            "words": words_data
        }, f, ensure_ascii=False, indent=2)
    print(f"Transcription saved to {output_file}")
    return {"words": words_data, "metadata": {"language": info.language, "language_probability": info.language_probability, "timestamp": timestamp}}, output_file

def create_reference_spectrogram(audio_path: Path, job_id: str) -> Path:
    """Create a mel spectrogram from the first 500ms of the reference audio."""
    import torch
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
    
    SAMPLE_RATE = 16000
    REFERENCE_DURATION_MS = 500
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Take first 500ms
        num_frames = int(SAMPLE_RATE * REFERENCE_DURATION_MS / 1000.0)
        waveform = waveform[:num_frames]
        
        # Create mel spectrogram
        mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)(waveform)
        mel_db = AmplitudeToDB()(mel)
        
        # Save reference spectrogram
        ref_spec_path = Path("mel_spectrograms") / f"reference_{job_id}.pt"
        ref_spec_path.parent.mkdir(exist_ok=True)
        torch.save(mel_db.clone(), ref_spec_path)
        
        print(f"[INFO] Reference spectrogram saved: {ref_spec_path}")
        return ref_spec_path
        
    except Exception as e:
        print(f"[ERROR] Failed to create reference spectrogram: {e}")
        raise

