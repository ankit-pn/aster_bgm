#!/usr/bin/env python3
"""
Extract cleaned background music (instrumental) from a list of YouTube videos
specified in a JSON file (mapping channel to video URL list).
Each output .wav will be named after its YouTube video ID.
"""

import os
import sys
import subprocess
import tempfile
import warnings
import json
import logging
from urllib.parse import urlparse, parse_qs

from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import torch
import soundfile as sf

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

import concurrent.futures

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 44_100   # Hz
CHANNELS         = 2        # stereo (Demucs needs 2)
CHUNK_MS         = 30_000   # 30 s
MIN_SILENCE_LEN  = 1500     # ms
SILENCE_THRESH   = -45      # dB
KEEP_SILENCE     = 500      # ms
NON_MUSIC_THRESH = 0.02     # fraction of max amplitude
# Optional cookies files for authenticated YouTube downloads
COOKIES_JSON = os.path.join(os.path.dirname(__file__), "cookies.json")
COOKIES_TXT  = os.path.join(os.path.dirname(__file__), "cookies.txt")
# ─────────────────────────────────────────────────────────────────────────

# ─── LOGGING SETUP ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbose output
    format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    handlers=[
        logging.StreamHandler(),      # Log to console
        # logging.FileHandler("bgm_extractor.log")  # Uncomment to log to file
    ]
)
logger = logging.getLogger(__name__)

def run(cmd: list[str], **kw):
    logger.debug(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)

def convert_json_cookies(json_path: str, txt_path: str) -> str:
    """Convert a Chrome/Firefox JSON cookies export to Netscape format."""
    logger.info("Converting JSON cookies → Netscape format")
    with open(json_path, "r") as f:
        data = json.load(f)
    cookies = data.get("cookies", data)
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c.get("domain", "")
        flag = "FALSE" if c.get("hostOnly") else "TRUE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = str(int(c.get("expirationDate", 0)))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append("\t".join([domain, flag, path, secure, expiry, name, value]))
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path

def download_wav(url: str, tmp: str) -> str:
    webm = os.path.join(tmp, "audio.webm")
    wav  = os.path.join(tmp, "audio.wav")
    logger.info(f"Downloading audio for URL: {url}")
    cmd = ["yt-dlp", "-f", "bestaudio"]
    cookies_file = None
    if os.path.exists(COOKIES_TXT):
        cookies_file = COOKIES_TXT
    elif os.path.exists(COOKIES_JSON):
        cookies_file = convert_json_cookies(COOKIES_JSON, COOKIES_TXT)
    if cookies_file:
        logger.debug(f"Using cookies from {cookies_file}")
        cmd += ["--cookies", cookies_file]
    cmd += ["-o", webm, url]
    run(cmd)
    logger.debug(f"Downloaded to {webm}, converting to {wav}")
    run([
        "ffmpeg", "-y", "-i", webm,
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-c:a", "pcm_s16le",
        wav
    ])
    logger.info(f"Audio converted to WAV: {wav}")
    return wav

def make_chunks(wav: str, tmp: str) -> list[str]:
    logger.debug(f"Splitting {wav} into chunks.")
    audio = AudioSegment.from_wav(wav)
    paths = []
    for i in range(0, len(audio), CHUNK_MS):
        seg = audio[i:i+CHUNK_MS]
        p   = os.path.join(tmp, f"chunk_{i//CHUNK_MS:04d}.wav")
        seg.export(p, format="wav")
        paths.append(p)
        logger.debug(f"Chunk created: {p}")
    logger.info(f"Created {len(paths)} chunks from {wav}")
    return paths

def demucs_accompaniment(chunk: str, model) -> str:
    logger.debug(f"Separating accompaniment for chunk: {chunk}")
    wav = AudioFile(chunk).read(streams=0,
                                samplerate=SAMPLE_RATE,
                                channels=CHANNELS)
    ref = wav.mean(0)
    norm = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        srcs = apply_model(model, norm[None], device="cpu")[0]  # (4,2,T)
    other = model.sources.index("other")
    drums = model.sources.index("drums")
    bass  = model.sources.index("bass")
    acc = srcs[other] + srcs[drums] + srcs[bass]             # (2,T)
    out = os.path.splitext(chunk)[0] + "_bg.wav"
    acc_np = acc.cpu().numpy().T                              # (T,2)
    sf.write(out, acc_np, SAMPLE_RATE, subtype="PCM_16")
    logger.debug(f"Accompaniment extracted and saved: {out}")
    return out

def prune_silence(wav: str) -> AudioSegment | None:
    logger.debug(f"Pruning silence from {wav}")
    audio = effects.normalize(AudioSegment.from_wav(wav))
    parts = split_on_silence(audio,
                             min_silence_len=MIN_SILENCE_LEN,
                             silence_thresh=SILENCE_THRESH,
                             keep_silence=KEEP_SILENCE)
    keep = [p for p in parts
            if p.rms > NON_MUSIC_THRESH * audio.max_possible_amplitude]
    logger.info(f"Kept {len(keep)} segments with audio above silence threshold.")
    return sum(keep) if keep else None

def get_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com", "music.youtube.com"):
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        if parsed.path.startswith("/embed/") or parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]
    raise ValueError(f"Cannot extract video ID from URL: {url}")

def process_video(url: str, model, outdir: str):
    video_id = get_video_id(url)
    logger.info(f"▶️  Processing {video_id}: {url}")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            wav = download_wav(url, tmp)
            chunks = make_chunks(wav, tmp)
            final = AudioSegment.empty()
            for c in chunks:
                try:
                    acc = demucs_accompaniment(c, model)
                    cleaned = prune_silence(acc)
                    if cleaned:
                        final += cleaned
                    os.remove(acc)
                except Exception as e:
                    logger.warning(f"Skipped {os.path.basename(c)}: {e}", exc_info=True)
            if len(final) > 0:
                out = os.path.join(outdir, f"{video_id}.wav")
                logger.info(f"💾  Saving → {out}")
                final.export(out, format="wav")
            else:
                logger.error(f"No music extracted for {video_id}")
        except Exception as e:
            logger.error(f"Error processing {url}: {e}", exc_info=True)

def process_video_star(args):
    channel, url, outdir = args
    logger.info(f"Loading Demucs model in process for {url}")
    model = get_model("htdemucs_ft")
    model.cpu().eval()
    process_video(url, model, outdir)

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python bgm_extractor.py <json_file>")
        sys.exit(1)
    json_path = sys.argv[1]
    if not os.path.isfile(json_path):
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    with open(json_path, "r") as f:
        video_map = json.load(f)

    outdir = "bgm_outputs"
    os.makedirs(outdir, exist_ok=True)

    video_tuples = []
    for channel, urls in video_map.items():
        for url in urls:
            video_tuples.append((channel, url, outdir))

    logger.info(f"Using up to {os.cpu_count()} processes. Starting parallel processing …")

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        list(executor.map(process_video_star, video_tuples))

    logger.info("✅  Done! All outputs in: %s", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
