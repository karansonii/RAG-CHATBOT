# src/voice_modules.py
import speech_recognition as sr
import tempfile
import os
from gtts import gTTS
import threading
import numpy as np
import pygame
from rapidfuzz import process, fuzz
import pyaudio
import wave
import logging
from src.utils import ConfigManager
from src.system_prompt import SystemPrompts
# Setup logging
logger = logging.getLogger(__name__)
# Load voice configuration
voice_config = ConfigManager.get_voice_config()
SAMPLE_RATE = voice_config['sample_rate']
CHUNK_SIZE = voice_config['chunk_size']
FUZZY_THRESHOLD = voice_config['fuzzy_threshold']
TTS_LANG = voice_config['tts_lang']
# Dynamic vocabulary
DOMAIN_KEYWORDS = ["EGP", "e-GP", "egp"]
def load_domain_keywords(metas: list, ngram_range=(1, 3)):
    """
    Extract keywords and multi-word phrases from RAG index metadata.
   
    Args:
        metas: List of metadata from RAG index
        ngram_range: Tuple of (min, max) n-gram sizes
    """
    global DOMAIN_KEYWORDS
    vocab = set()
   
    for m in metas:
        text = m.get("text") or m.get("content") or ""
        words = [w.strip().lower() for w in text.split() if len(w.strip()) > 3]
        # Single words
        vocab.update(words)
        # Multi-word ngrams
        min_n, max_n = ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                vocab.add(phrase)
    vocab.update(["egp", "EGP", "e-GP"])
    DOMAIN_KEYWORDS = list(vocab)
    logger.info(f"Loaded {len(DOMAIN_KEYWORDS)} domain keywords")
def fuzzy_correct(text: str, choices=None, threshold: int = None) -> str:
    """
    Correct transcription errors using fuzzy matching.
   
    Args:
        text: Input text to correct
        choices: List of valid terms (uses DOMAIN_KEYWORDS if None)
        threshold: Minimum match score (uses config if None)
    """
    if not text:
        return text
   
    if choices is None:
        choices = DOMAIN_KEYWORDS or []
   
    if not choices:
        return text
   
    if threshold is None:
        threshold = FUZZY_THRESHOLD
   
    # Force match special words with looser threshold
    special_words = ConfigManager.get('FUZZY_SPECIAL_WORDS', 'egp,e-gp').split(',')
    best_special, score_special, _ = process.extractOne(
        text.lower(), special_words, scorer=fuzz.ratio
    )
    special_threshold = ConfigManager.get('FUZZY_SPECIAL_THRESHOLD', 60, config_type=int)
    if score_special >= special_threshold:
        return best_special
   
    # General fuzzy matching
    best_match, score, _ = process.extractOne(
        text, choices, scorer=fuzz.token_sort_ratio
    )
   
    if score >= threshold:
        return best_match
   
    return text
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file (.wav) to text using SpeechRecognition.
   
    Args:
        file_path: Path to audio file
    """
    recognizer = sr.Recognizer()
   
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
       
        text = recognizer.recognize_google(audio)
        corrected_text = fuzzy_correct(text)
        logger.info(f"Transcribed: '{text}' → '{corrected_text}'")
        return corrected_text
       
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        logger.error(f"Speech recognition error: {e}")
        return f"Speech recognition error: {e}"
class LiveMicRecorder:
    """Handles live microphone recording with waveform and manual controls."""
    def __init__(self):
        self.transcription = ""
        self.listening = False
        self.frames = []
        self.chunk = CHUNK_SIZE
        self.rate = SAMPLE_RATE
        self.stream = None
        self.p = None
        self.audio_data = []
        logger.info(f"LiveMicRecorder initialized: rate={self.rate}, chunk={self.chunk}")
    def start_listening(self):
        """Start recording audio from microphone."""
        self.listening = True
        self.audio_data.clear()
       
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            threading.Thread(target=self._record_loop, daemon=True).start()
            logger.info("Started listening to microphone")
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self.listening = False
    def _record_loop(self):
        """Internal recording loop."""
        while self.listening:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.audio_data.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                logger.error(f"Recording error: {e}")
                break
    def stop_listening(self):
        """Stop recording audio."""
        self.listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        logger.info("Stopped listening to microphone")
    def save_temp_wav(self):
        """
        Save recorded PCM data as proper .wav file.
       
        Returns:
            str: Path to temporary .wav file or None
        """
        if not self.audio_data:
            logger.warning("No audio data to save")
            return None
       
        audio_array = np.concatenate(self.audio_data).astype(np.int16)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
       
        try:
            with wave.open(tmpfile.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.rate)
                wf.writeframes(audio_array.tobytes())
            logger.info(f"Saved audio to {tmpfile.name}")
            return tmpfile.name
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None
    def transcribe(self):
        """
        Transcribe recorded audio to text.
       
        Returns:
            str: Transcribed text
        """
        tmp_path = self.save_temp_wav()
        if tmp_path:
            text = transcribe_audio(tmp_path)
            try:
                os.remove(tmp_path)
            except:
                pass
            self.transcription = text
            return text
        return ""
    def get_waveform_snapshot(self, bars: int = 30) -> str:
        """
        Return ASCII bar waveform of latest audio chunk.
       
        Args:
            bars: Number of bars to display
        """
        if not self.audio_data:
            return "▁" * bars
       
        latest = self.audio_data[-1]
        norm = np.abs(latest) / 32768.0
        chunked = np.array_split(norm, bars)
        levels = [np.mean(c) for c in chunked]
        return "".join("█" if lvl > 0.3 else "▁" for lvl in levels)
# TTS functionality
_current_audio = {"is_playing": False, "thread": None}
pygame.mixer.init()
def _play_audio(file_path):
    """Internal function to play audio file."""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        logger.error(f"Audio playback error: {e}")
    finally:
        _current_audio["is_playing"] = False
def play_tts(text: str, lang: str = None):
    """
    Play TTS audio in toggle mode.
   
    Args:
        text: Text to convert to speech
        lang: Language code (uses TTS_DEFAULT_LANG from config if None)
    """
    # Toggle: stop if already playing
    if _current_audio["is_playing"]:
        pygame.mixer.music.stop()
        _current_audio["is_playing"] = False
        logger.info("Stopped TTS playback")
        return
   
    # Use configured language if not provided
    if lang is None:
        lang = TTS_LANG
   
    try:
        tts = gTTS(text=text, lang=lang)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmpfile.name)
        tmpfile.close()
       
        _current_audio["is_playing"] = True
        t = threading.Thread(target=_play_audio, args=(tmpfile.name,), daemon=True)
        t.start()
        _current_audio["thread"] = t
        logger.info(f"Started TTS playback in {lang}")
       
    except Exception as e:
        logger.error(f"TTS error: {e}")
        _current_audio["is_playing"] = False
def stop_tts():
    """Stop TTS playback if playing."""
    if _current_audio["is_playing"]:
        pygame.mixer.music.stop()
        _current_audio["is_playing"] = False
        logger.info("TTS playback stopped")
# Export
__all__ = [
    'LiveMicRecorder',
    'play_tts',
    'stop_tts',
    'transcribe_audio',
    'DOMAIN_KEYWORDS',
    'load_domain_keywords',
    'fuzzy_correct'
]
