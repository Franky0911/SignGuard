import io
from typing import Optional

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False


def synthesize_tts(text: str, lang: str = 'en') -> Optional[bytes]:
    # Prefer gTTS for easy file bytes
    if HAS_GTTS:
        try:
            tts = gTTS(text=text, lang=lang)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()
        except Exception:
            pass
    # Fallback: pyttsx3 requires local playback; return None
    return None


