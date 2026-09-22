"""Matnni ovozga aylantirish (TTS) va ijro etish.

Barcha provayderlar xom PCM qaytaradi (mp3 emas) — shuning uchun `ffmpeg` kerak emas
va ijro paytida ovoz darajasini o'lchab, orb animatsiyasini ovozga moslashtirish mumkin.

O'zbek tili uchun variantlar:
  * edge        — Microsoft Edge TTS, `uz-UZ-SardorNeural` / `uz-UZ-MadinaNeural`.
                  Kalit va hisob talab qilmaydi (faqat internet) — eng tekin
                  haqiqiy o'zbekcha neyron ovoz;
  * azure      — `uz-UZ-SardorNeural` / `uz-UZ-MadinaNeural`, kalit kerak;
  * mohir      — Mohir.ai, mahalliy provayder;
  * windows    — tizim SAPI ovozlari; o'zbek ovozi yo'q, matnni harflab o'qiydi;
  * macos      — tizim `say` buyrug'i; o'zbek ovozi yo'q, faqat sinov uchun.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

import httpx
import numpy as np

from ..config import env, require_env

log = logging.getLogger("jarvis.voice.tts")

TTS_SAMPLE_RATE = 24000
LevelCallback = Callable[[float], None]


class TtsProvider(ABC):
    """Umumiy interfeys: matn -> PCM bo'laklari oqimi."""

    sample_rate: int = TTS_SAMPLE_RATE

    @abstractmethod
    def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """`int16` PCM bo'laklarini oqim sifatida beradi."""

    async def aclose(self) -> None:
        """Resurslarni bo'shatadi."""


# ElevenLabs javobida `labels.gender` bo'ladi, lekin har doim emas. Shuning
# uchun ikkinchi yo'l: kutubxonadagi tanish ovozlarning nomlari.
KNOWN_VOICES: dict[str, frozenset[str]] = {
    "female": frozenset({
        "aria", "sarah", "laura", "charlotte", "alice", "matilda", "jessica",
        "lily", "rachel", "domi", "bella", "elli", "freya", "gigi", "glinda",
        "nicole", "dorothy", "serena", "grace", "emily",
    }),
    "male": frozenset({
        "roger", "charlie", "george", "callum", "liam", "will", "eric", "chris",
        "brian", "daniel", "adam", "antoni", "arnold", "josh", "sam", "bill",
    }),
}


class ElevenLabsTts(TtsProvider):
    """ElevenLabs streaming TTS."""

    BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, voice: str = "Aria", model: str = "eleven_multilingual_v2",
                 speed: float = 1.0, gender: str = "female") -> None:
        self.sample_rate = 24000
        self._voice = voice
        self._model = model
        self._speed = speed
        self._gender = gender.lower().strip()
        self._voice_id: str | None = None
        self._client = httpx.AsyncClient(timeout=60.0)

    def _pick_fallback(self, voices: list[dict]) -> dict:
        """So'ralgan nom yo'q — jinsi mos keladigan ovozni topadi.

        Avval ElevenLabs bergan yorliqqa ishonamiz, u bo'lmasa nomlar
        ro'yxatiga qaraymiz, ikkisi ham yordam bermasa birinchisini olamiz.
        """
        if self._gender:
            for voice in voices:
                labels = voice.get("labels") or {}
                if str(labels.get("gender", "")).lower() == self._gender:
                    return voice

            known = KNOWN_VOICES.get(self._gender, frozenset())
            for voice in voices:
                if str(voice.get("name", "")).lower() in known:
                    return voice

        return voices[0]

    def _gender_word(self) -> str:
        return {"female": "ayol", "male": "erkak"}.get(self._gender, self._gender)

    async def _resolve_voice_id(self, api_key: str) -> str:
        """Nom berilgan bo'lsa, uni ovoz ID'siga aylantiradi (bir marta)."""
        if self._voice_id:
            return self._voice_id

        # ElevenLabs ID'lari 20 belgili alfanumerik — nomdan shu bilan ajratamiz.
        if len(self._voice) == 20 and self._voice.isalnum():
            self._voice_id = self._voice
            return self._voice_id

        response = await self._client.get(f"{self.BASE}/voices", headers={"xi-api-key": api_key})
        response.raise_for_status()
        voices = response.json().get("voices", [])

        for voice in voices:
            if str(voice.get("name", "")).lower() == self._voice.lower():
                self._voice_id = str(voice["voice_id"])
                log.info("ElevenLabs ovozi topildi: %s -> %s", self._voice, self._voice_id)
                return self._voice_id

        # So'ralgan ovoz yo'q. Har bir hisobdagi ovoz to'plami boshqacha, shuning
        # uchun bu odatiy hol — Jarvisni butunlay ovozsiz qoldirgandan ko'ra
        # mavjud birinchi ovozga o'tamiz va nimani ishlatganimizni aytamiz.
        if voices:
            fallback = self._pick_fallback(voices)
            self._voice_id = str(fallback["voice_id"])
            log.warning(
                "ElevenLabs'da `%s` ovozi yo'q — %sovoz `%s` ishlatiladi. "
                "Mavjud ovozlar: %s. Kerakligini config/jarvis.yaml da "
                "`voice.tts.voice` ga yozing.",
                self._voice,
                f"{self._gender_word()} " if self._gender else "",
                fallback.get("name", self._voice_id),
                ", ".join(str(v.get("name", "?")) for v in voices[:10]) or "—",
            )
            return self._voice_id

        raise RuntimeError(
            "ElevenLabs hisobingizda birorta ovoz yo'q. elevenlabs.io > Voices da "
            "kutubxonadan ovoz qo'shing (masalan multilingual ovozlardan bittasi)."
        )

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        api_key = require_env("ELEVENLABS_API_KEY", "ElevenLabs TTS uchun")
        voice_id = await self._resolve_voice_id(api_key)

        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                               "speed": self._speed},
        }
        url = f"{self.BASE}/text-to-speech/{voice_id}/stream?output_format=pcm_24000"

        async with self._client.stream(
            "POST", url, headers={"xi-api-key": api_key}, json=payload
        ) as response:
            response.raise_for_status()
            # Baytlar bo'lakma-bo'lak keladi; toq baytni keyingi bo'lakka o'tkazamiz.
            leftover = b""
            async for chunk in response.aiter_bytes():
                data = leftover + chunk
                usable = len(data) - (len(data) % 2)
                leftover = data[usable:]
                if usable:
                    yield np.frombuffer(data[:usable], dtype=np.int16)

    async def aclose(self) -> None:
        await self._client.aclose()


class AzureTts(TtsProvider):
    """Azure Speech — haqiqiy o'zbekcha neyron ovozlar (uz-UZ)."""

    def __init__(self, voice: str = "uz-UZ-SardorNeural", speed: float = 1.0) -> None:
        self.sample_rate = 24000
        self._voice = voice
        self._speed = speed
        self._client = httpx.AsyncClient(timeout=60.0)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        key = require_env("AZURE_SPEECH_KEY", "Azure TTS uchun")
        region = require_env("AZURE_SPEECH_REGION", "Azure TTS uchun (masalan, westeurope)")

        # Tezlikni SSML foiziga aylantiramiz: 1.0 -> 0%, 1.2 -> +20%
        rate = f"{(self._speed - 1.0) * 100:+.0f}%"
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="uz-UZ">'
            f'<voice name="{self._voice}"><prosody rate="{rate}">'
            f"{_escape_xml(text)}</prosody></voice></speak>"
        )

        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
            "User-Agent": "jarvis",
        }

        async with self._client.stream("POST", url, headers=headers, content=ssml.encode()) as resp:
            resp.raise_for_status()
            leftover = b""
            async for chunk in resp.aiter_bytes():
                data = leftover + chunk
                usable = len(data) - (len(data) % 2)
                leftover = data[usable:]
                if usable:
                    yield np.frombuffer(data[:usable], dtype=np.int16)

    async def aclose(self) -> None:
        await self._client.aclose()


class MohirTts(TtsProvider):
    """Mohir.ai TTS — mahalliy o'zbek provayderi."""

    DEFAULT_URL = "https://uzbekvoice.ai/api/v1/tts"

    def __init__(self, voice: str = "gulnoza", speed: float = 1.0) -> None:
        self.sample_rate = 16000
        self._voice = voice
        self._speed = speed
        self._url = env("MOHIR_TTS_URL", self.DEFAULT_URL)
        self._client = httpx.AsyncClient(timeout=60.0)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        api_key = require_env("MOHIR_API_KEY", "Mohir.ai TTS uchun")
        response = await self._client.post(
            self._url,
            headers={"Authorization": api_key},
            json={"text": text, "model": self._voice, "format": "wav"},
        )
        response.raise_for_status()
        yield _pcm_from_wav(response.content)

    async def aclose(self) -> None:
        await self._client.aclose()


class EdgeTts(TtsProvider):
    """Microsoft Edge TTS — bepul neyron ovozlar, kalit va hisob talab qilmaydi.

    Edge brauzerining matn o'qish xizmatidan foydalanadi (internet kerak, lekin
    API kalit emas). O'zbekcha haqiqiy neyron ovozlari bor — SAPI'dagi kabi
    "harflab o'qish" muammosi yo'q.
    """

    def __init__(self, voice: str = "uz-UZ-SardorNeural", speed: float = 1.0) -> None:
        self.sample_rate = 24000
        self._voice = voice
        self._speed = speed

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        import io

        import edge_tts
        import soundfile as sf

        # Tezlikni Edge'ning foiz formatiga aylantiramiz: 1.0 -> +0%, 1.2 -> +20%
        rate = f"{(self._speed - 1.0) * 100:+.0f}%"
        communicate = edge_tts.Communicate(text, self._voice, rate=rate)

        mp3 = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3.write(chunk["data"])
        data, _ = sf.read(io.BytesIO(mp3.getvalue()), dtype="int16")
        yield data


class WindowsSapiTts(TtsProvider):
    """Windows SAPI — PowerShell System.Speech orqali. Kalitlarsiz lokal ovoz.

    O'zbek ovozi yo'q (standart ovoz o'zbekcha matnni talaffuz qiladi), lekin
    javob eshitilib turadi va xuddi macOS `say` kabi sinov uchun qulay.
    """

    def __init__(self, voice: str = "", speed: float = 1.0) -> None:
        self.sample_rate = 22050
        self._voice = voice
        self._rate = max(-10, min(10, round((speed - 1.0) * 10)))

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        import os
        import tempfile

        def run() -> bytes:
            # Matn va natija vaqtinchalik fayllarda — PowerShell satriga
            # joylashtirganda qo'shtirnoq/apostrof muammolari chiqmasligi uchun.
            wav_path = os.path.join(tempfile.gettempdir(), f"jarvis_tts_{os.getpid()}.wav")
            txt_path = os.path.join(tempfile.gettempdir(), f"jarvis_tts_{os.getpid()}.txt")
            with open(txt_path, "w", encoding="utf-8-sig") as handle:
                handle.write(text)

            select = f"$s.SelectVoice('{self._voice}'); " if self._voice else ""
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {self._rate}; {select}"
                f"$s.SetOutputToWaveFile('{wav_path}'); "
                f"$s.Speak([System.IO.File]::ReadAllText('{txt_path}')); "
                "$s.Dispose()"
            )
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True, text=True, timeout=120,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"SAPI TTS xato: {(proc.stderr or '').strip()[:200]}")
                with open(wav_path, "rb") as handle:
                    return handle.read()
            finally:
                for path in (wav_path, txt_path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        yield _pcm_from_wav(await asyncio.to_thread(run))


class MacosSayTts(TtsProvider):
    """macOS `say` — o'zbek ovozi yo'q, lekin kalitlarsiz darhol sinab ko'rish uchun qulay."""

    def __init__(self, voice: str = "Samantha", speed: float = 1.0) -> None:
        self.sample_rate = 22050
        self._voice = voice
        self._rate = int(175 * speed)

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        if not shutil.which("say"):
            raise RuntimeError("`say` buyrug'i topilmadi — bu backend faqat macOS'da ishlaydi")

        def run() -> bytes:
            result = subprocess.run(
                ["say", "-v", self._voice, "-r", str(self._rate),
                 "--data-format=LEI16@22050", "-o", "-", text],
                capture_output=True, check=True,
            )
            return result.stdout

        yield _pcm_from_wav(await asyncio.to_thread(run))


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pcm_from_wav(data: bytes) -> np.ndarray:
    """WAV baytlaridan PCM ajratib oladi; WAV bo'lmasa, xom PCM deb qaraydi."""
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    except (wave.Error, EOFError):
        usable = len(data) - (len(data) % 2)
        return np.frombuffer(data[:usable], dtype=np.int16)


class Speaker:
    """PCM oqimini dinamikka chiqaradi va ovoz darajasini xabar qiladi.

    `stop()` chaqirilsa, gapirish darhol to'xtaydi — foydalanuvchi Jarvis'ning
    gapini bo'lishi mumkin bo'lishi kerak.
    """

    def __init__(self, device: object = None) -> None:
        self._device = device
        self._stop = asyncio.Event()
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def stop(self) -> None:
        """Joriy gapirishni bekor qiladi."""
        self._stop.set()

    async def play(
        self,
        chunks: AsyncIterator[np.ndarray],
        sample_rate: int,
        on_level: LevelCallback | None = None,
    ) -> bool:
        """Oqimni ijro etadi. To'liq tugasa `True`, bo'lingan bo'lsa `False`."""
        import sounddevice as sd

        self._stop.clear()
        self._speaking = True
        stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", device=self._device
        )
        stream.start()
        completed = True

        try:
            async for chunk in chunks:
                if self._stop.is_set():
                    completed = False
                    break
                if chunk.size == 0:
                    continue

                if on_level is not None:
                    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
                    on_level(min(1.0, rms / 8000.0))

                # Yozish bloklaydi — alohida oqimda bajaramiz, aks holda
                # butun yadro (uyg'otuvchi so'z, UI) muzlab qoladi.
                await asyncio.to_thread(stream.write, chunk)
        finally:
            self._speaking = False
            await asyncio.to_thread(stream.stop)
            stream.close()
            if on_level is not None:
                on_level(0.0)

        return completed


# Har bir provayderning standart ovozi jinsga qarab. `voice.tts.gender`
# sozlamada nom ko'rsatilmagan yoki topilmagan hollarda tanlovni boshqaradi.
DEFAULT_VOICES: dict[str, dict[str, str]] = {
    "edge": {"female": "uz-UZ-MadinaNeural", "male": "uz-UZ-SardorNeural"},
    "azure": {"female": "uz-UZ-MadinaNeural", "male": "uz-UZ-SardorNeural"},
    # Mohir.ai'da erkak ovozining nomini tasdiqlaganim yo'q — sozlamada
    # `voice.tts.voice` ga yozib qo'yish kerak.
    "mohir": {"female": "gulnoza"},
    "macos": {"female": "Samantha", "male": "Daniel"},
}


def _default_voice(provider: str, gender: str) -> str:
    choices = DEFAULT_VOICES[provider]
    return choices.get(gender, choices["female"])


def build_tts(cfg: dict) -> TtsProvider:
    provider = str(cfg.get("provider", "elevenlabs")).lower()
    gender = str(cfg.get("gender", "female")).lower().strip()
    voice = str(cfg.get("voice", "") or "")
    speed = float(cfg.get("speed", 1.0))

    if provider == "elevenlabs":
        return ElevenLabsTts(voice=voice or "Aria", gender=gender, speed=speed,
                             model=str(cfg.get("model", "eleven_multilingual_v2")))
    if provider == "edge":
        # Edge ovozlari `uz-UZ-...` ko'rinishida; boshqa nom kelsa e'tibor bermaymiz.
        return EdgeTts(voice=voice if voice.startswith("uz-") else _default_voice("edge", gender),
                       speed=speed)
    if provider == "azure":
        # Azure ovozlari `uz-UZ-...` ko'rinishida; boshqa nom kelsa e'tibor bermaymiz.
        return AzureTts(voice=voice if voice.startswith("uz-") else _default_voice("azure", gender),
                        speed=speed)
    if provider == "mohir":
        return MohirTts(voice=voice or _default_voice("mohir", gender), speed=speed)
    if provider == "macos":
        return MacosSayTts(
            voice=voice if voice and not voice.startswith("uz-") else _default_voice("macos", gender),
            speed=speed,
        )
    if provider == "windows":
        return WindowsSapiTts(voice=voice, speed=speed)
    raise ValueError(f"Noma'lum TTS provayderi: {provider}")
