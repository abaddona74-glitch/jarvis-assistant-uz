"""Jarvis yadrosi — barcha qismlarni birlashtiruvchi asosiy sikl.

Oqim:
    uyg'otish (so'z / qarsak / bosish)
        -> tinglash (VAD gapirish tugaganini aniqlaguncha)
        -> matnga aylantirish (STT)
        -> o'ylash va ish qilish (Claude Agent SDK + xavfsizlik darvozasi)
        -> gapirish (TTS, gap-gap; foydalanuvchi gapirsa — to'xtash)
        -> yana tinglash (uyg'otuvchi so'zsiz) yoki kutishga qaytish
"""

from __future__ import annotations

import asyncio
import logging
import random
import signal
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any

import numpy as np

from .audio.bargein import build_barge_in
from .audio.clap import ClapDetector
from .audio.mic import MicStream, frame_level
from .audio.phrases import build_matcher
from .audio.vad import Endpointer, build_speech_detector
from .audio.wake import WakeHit, build_wake_detector
from .brain.agenda import Agenda
from .brain.agent import Brain
from .brain.memory import Memory
from .brain.prompts import (
    CLAP_GREETINGS,
    ERROR_REPLY,
    FAREWELLS,
    FIRST_GREETING,
    GREETINGS,
    NOT_UNDERSTOOD,
)
from .bus import EventBus, State
from .config import Config, load_config
from .doctor import hint_for
from .health import Health, Status, System
from .safety.gate import SafetyGate
from .scheduler import Announcement, Scheduler
from .ui.server import UiServer, base64_to_pcm
from .voice.consent import consent_prompt, parse_consent
from .voice.intents import is_end_of_conversation, is_stop_speaking
from .voice.stt import build_stt, transcribe_guarded
from .voice.tts import Speaker, build_tts

log = logging.getLogger("jarvis")


def chime(sample_rate: int = 24000, freq: float = 880.0, duration: float = 0.12) -> np.ndarray:
    """Uyg'onganini bildiruvchi qisqa signal.

    Fayl saqlamaymiz — generatsiya qilish arzon va o'rnatishni soddalashtiradi.
    Boshi va oxiri silliqlanadi, aks holda "chirt" eshitiladi.
    """
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    fade = int(sample_rate * 0.015)
    if fade * 2 < wave.size:
        wave[:fade] *= np.linspace(0.0, 1.0, fade)
        wave[-fade:] *= np.linspace(1.0, 0.0, fade)
    return (wave * 0.25 * 32767).astype(np.int16)


class Jarvis:
    """Yadro. Bitta event loop'da ishlaydi."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bus = EventBus()
        # Bo'g'inlar tirikligi — HUD siferblatlari shu reyestrdan oziqlanadi.
        self.health = Health(self.bus.emit)
        self.memory = Memory(config.memory_path)
        self.agenda = Agenda(config.memory_path)
        self.gate = SafetyGate(config=config, bus=self.bus)
        # Miya provayderi: "claude" (Agent SDK, kalit/obuna) yoki "gemini"
        # (Google AI Studio kaliti — bepul).
        if str(config.get("brain.provider", "claude")).lower() == "gemini":
            from .brain.gemini_agent import GeminiBrain

            self.brain: Any = GeminiBrain(
                config=config,
                bus=self.bus,
                memory=self.memory,
                agenda=self.agenda,
                gate=self.gate,
                announce=self._speak,
            )
        else:
            self.brain = Brain(
                config=config,
                bus=self.bus,
                memory=self.memory,
                agenda=self.agenda,
                gate=self.gate,
                announce=self._speak,
            )

        # Rejalashtiruvchi vaqti kelgan eslatmalarni shu navbatga qo'yadi;
        # asosiy sikl uni bo'sh bo'lganda bo'shatadi.
        self.proactive: asyncio.Queue[Announcement] = asyncio.Queue()
        sched_cfg = config.section("scheduler")
        self.scheduler = Scheduler(
            agenda=self.agenda,
            queue=self.proactive,
            check_interval_sec=float(sched_cfg.get("check_interval_sec", 30)),
            quiet_start=int(sched_cfg.get("quiet_start_hour", 22)),
            quiet_end=int(sched_cfg.get("quiet_end_hour", 7)),
            brief_time=str(sched_cfg.get("brief_time", "08:30")),
            enabled=bool(sched_cfg.get("enabled", True)),
        )

        self.stt = build_stt(config.section("voice.stt"))
        self.tts = build_tts(config.section("voice.tts"))
        self.speaker = Speaker(device=config.get("audio.output_device"))

        self.ui = UiServer(
            self.bus,
            host=str(config.get("ui.host", "127.0.0.1")),
            port=int(config.get("ui.port", 8765)),
            token=str(config.get("ui.token") or ""),
        )

        self.mic = MicStream(
            sample_rate=config.sample_rate,
            frame_samples=config.frame_samples,
            device=config.get("audio.input_device"),
            preroll_ms=int(config.get("audio.endpointing.preroll_ms", 300)),
            gain=float(config.get("audio.input_gain", 1.0)),
            recent_ms=int(config.get("activation.wake_word.verify_window_ms", 2000)) + 500,
        )

        wake_cfg = config.section("activation.wake_word")
        self._wake = build_wake_detector(wake_cfg)
        # Ikkinchi bosqich: tayyor model bilmagan iboralarni matn bilan tanish.
        self._phrases = build_matcher(wake_cfg)
        self._verify_cooldown = float(wake_cfg.get("verify_cooldown_sec", 3.0))
        self._verify_window_ms = int(wake_cfg.get("verify_window_ms", 2000))
        self._last_verify = -1e9

        clap_cfg = config.section("activation.clap")
        self._clap = (
            ClapDetector(
                sample_rate=config.sample_rate,
                onset_ratio=float(clap_cfg.get("onset_ratio", 8.0)),
                min_gap_sec=float(clap_cfg.get("min_gap_sec", 0.12)),
                max_gap_sec=float(clap_cfg.get("max_gap_sec", 0.70)),
                cooldown_sec=float(clap_cfg.get("cooldown_sec", 2.0)),
            )
            if clap_cfg.get("enabled", True)
            else None
        )

        # Uzluksiz suhbat va gapni bo'lish — ikkisi ham bitta bo'limda.
        self._talk = config.section("conversation")
        self._barge_in = build_barge_in(
            self._talk,
            detector=build_speech_detector(
                config.section("audio.endpointing"), config.sample_rate
            ),
            frame_ms=int(config.get("audio.frame_ms", 20)),
        )

        # Tinglayotganda tizim ovozini pasaytirish — musiqa ustidan
        # eshitilishi uchun.
        self._duck_enabled = bool(config.get("audio.duck_while_listening", True))
        self._duck_level = int(config.get("audio.duck_volume", 20))

        # Ovozli tasdiq: tugma bosish shart emas, «ha» / «yo'q» deyish yetadi.
        vc = config.section("safety.voice_confirm")
        self._vc_enabled = bool(vc.get("enabled", True))
        self._vc_listen = float(vc.get("listen_sec", 8))
        self._vc_attempts = int(vc.get("attempts", 2))
        self._confirm_task: asyncio.Task[None] | None = None

        self._activate = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._greeted = False
        self._interrupted = False
        self._heartbeat: asyncio.Task[None] | None = None

    # --- Hayot sikli ---

    async def start(self) -> None:
        self.config.ensure_dirs()

        self.ui.on("activate", self._on_activate)
        self.ui.on("stop", self._on_stop)
        self.ui.on("text", self._on_text_input)
        self.ui.on("audio", self._on_phone_audio)

        # Tasdiq so'rovlarini ovoz bilan ham hal qilish uchun shinani tinglaymiz.
        if self._vc_enabled:
            self.bus.subscribe(self._on_bus_message)

        await self.ui.start()
        await self.brain.start()
        await self.scheduler.start()
        await self.mic.start()
        await self._check_mic_delivers_audio()
        await self._mark_ready()
        self._heartbeat = asyncio.create_task(self.health.heartbeat())
        await self.bus.set_state(State.IDLE)

        stats = self.memory.stats()
        log.info(
            "Jarvis tayyor — %d fakt, %d vazifa, %d loyiha",
            stats["faktlar"], len(self.agenda.list_tasks()), len(self.agenda.list_projects()),
        )

        if self.ui._is_loopback(self.ui.host):
            log.info("Telefonni ulash uchun `ui.host` ni 0.0.0.0 qiling va token qo'ying")
        else:
            log.info("Telefonda oching: %s", self._lan_phone_url())

        await self.bus.log_line("Jarvis tayyor. «Hey Jarvis» deb chaqiring.")

    async def _check_mic_delivers_audio(self, seconds: float = 1.0) -> None:
        """Mikrofon ochildi — lekin ovoz kelyaptimi?

        macOS mikrofon ruxsatini ishga tushiruvchi dastur bo'yicha beradi.
        Terminaldan ishga tushirilganda javobgar Terminal bo'ladi va ruxsat
        bor; `launchd` ostida esa javobgar boshqa va ruxsat yo'q. Bunday
        holatda macOS so'ramaydi ham, xato ham bermaydi — oqim ochiladi va
        ichida faqat NOL keladi.

        Natijada Jarvis «tayyor» deb turadi, chaqiruvni esa hech qachon
        eshitmaydi. Shuning uchun boshida bir soniya tinglab, aynan shu
        holatni ajratamiz: mutlaq nol — bu jimlik emas, bu bloklangan
        mikrofon (haqiqiy jimlikda ham shovqin fon bo'ladi).
        """
        frames = int(seconds * self.config.sample_rate / self.config.frame_samples)
        loudest = 0.0

        try:
            async with asyncio.timeout(seconds + 2):
                seen = 0
                async for frame in self.mic.frames():
                    loudest = max(loudest, float(np.max(np.abs(frame.astype(np.int32)))))
                    seen += 1
                    if seen >= frames:
                        break
        except (TimeoutError, asyncio.CancelledError):
            pass

        if loudest > 0:
            return

        reason = "macOS mikrofonni bloklayapti (faqat nol keladi)"
        log.error(
            "MIKROFON OVOZ BERMAYAPTI. Oqim ochildi, lekin ichida faqat nol.\n"
            "Bu deyarli har doim macOS ruxsati: Tizim sozlamalari > Maxfiylik "
            "va xavfsizlik > Mikrofon.\n"
            "Ro'yxatga Python qo'shilganini tekshiring: %s\n"
            "Vaqtinchalik yechim: `./scripts/autostart.sh off` va terminaldan "
            "`./scripts/run.sh` — terminal ruxsatga ega.",
            sys.executable,
        )
        await self.health.mark(System.MIC, Status.DOWN, reason)
        await self.bus.log_line(f"Mikrofon: {reason}", level="error")

    async def _mark_ready(self) -> None:
        """Ishga tushgach har bir bo'g'inning boshlang'ich holatini belgilaydi.

        «Tayyor» degani — qism qurildi va ishlashga shay. Haqiqatan ishlashini
        birinchi chaqiruv ko'rsatadi: xato chiqsa, siferblat qizarib qoladi.
        """
        for system in (System.MIC, System.STT, System.BRAIN, System.TTS,
                       System.MEMORY, System.AGENDA, System.TOOLS, System.NET):
            await self.health.mark(system, Status.READY, quiet=True)

        if self._wake is None:
            await self.health.mark(
                System.WAKE, Status.UNKNOWN, "o'chirilgan — qarsak va tugma ishlaydi",
                quiet=True,
            )
        else:
            await self.health.mark(System.WAKE, Status.READY, quiet=True)

        # Hammasi belgilangach — bitta to'liq manzara yuboramiz.
        await self.bus.emit(self.health.snapshot())

    def _lan_phone_url(self) -> str:
        """Telefonda ochish uchun tarmoqdagi haqiqiy manzil.

        `0.0.0.0` — bu "hamma interfeyslarda tingla" degani, telefonga
        yozib bo'ladigan manzil emas. Shuning uchun mashinaning LAN IP'sini
        topib beramiz.
        """
        import socket

        host = self.ui.host
        if host == "0.0.0.0":
            try:
                # Tashqi manzilga "ulanish" — paket yuborilmaydi, lekin OS
                # qaysi interfeys ishlatilishini aytadi.
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.connect(("8.8.8.8", 80))
                    host = probe.getsockname()[0]
            except OSError:
                host = socket.gethostname()

        token = f"?t={self.ui.token}" if self.ui.token else ""
        return f"http://{host}:{self.ui.port}/{token}"

    def request_stop(self) -> None:
        """Asosiy siklga to'xtash haqida xabar beradi."""
        self._shutdown.set()

    async def stop(self) -> None:
        self._shutdown.set()
        self.speaker.stop()
        if self._confirm_task is not None and not self._confirm_task.done():
            self._confirm_task.cancel()
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        await self.scheduler.stop()
        await self.mic.stop()
        await self.brain.stop()
        await self.ui.stop()
        await self.stt.aclose()
        await self.tts.aclose()
        if self._wake is not None:
            self._wake.close()
        self.agenda.close()
        self.memory.close()
        log.info("Jarvis to'xtadi")

    # --- UI buyruqlari ---

    async def _on_activate(self, message: dict[str, Any]) -> None:
        """Orb bosildi — uyg'otuvchi so'zsiz ishga tushirish."""
        self._activate.set()

    async def _on_stop(self, message: dict[str, Any]) -> None:
        """Foydalanuvchi to'xtatdi."""
        self.speaker.stop()
        await self.brain.interrupt()
        await self.bus.set_state(State.IDLE)

    async def _on_text_input(self, message: dict[str, Any]) -> None:
        """Orb'dan matn keldi — ovozsiz rejim (shovqinli joyda foydali)."""
        text = str(message.get("text", "")).strip()
        if text:
            await self._handle_utterance(text, remote=message.get("_client"))

    async def _on_phone_audio(self, message: dict[str, Any]) -> None:
        """Telefondan yozib olingan audio keldi.

        Javob shu telefonga qaytariladi — kompyuter dinamigidan gapirmaydi,
        chunki foydalanuvchi uning oldida emas.
        """
        client = str(message.get("_client", ""))
        pcm = base64_to_pcm(str(message.get("pcm", "")))
        rate = int(message.get("sample_rate", self.config.sample_rate))

        if pcm.size == 0:
            await self.ui.send_to(client, {"type": "error", "text": "Audio bo'sh keldi"})
            return

        log.info("Telefondan audio: %.1f s", pcm.size / rate)
        await self.bus.set_state(State.THINKING)

        async with self.health.busy(System.STT):
            text = await transcribe_guarded(self.stt, pcm, rate)
        if not text:
            await self._speak(random.choice(NOT_UNDERSTOOD), remote=client)
            await self.bus.set_state(State.IDLE)
            return

        await self.bus.transcript(text)
        await self.ui.send_to(client, {"type": "you_said", "text": text})
        await self._handle_utterance(text, remote=client)

    # --- Asosiy sikl ---

    async def run(self) -> None:
        await self.start()
        try:
            await self._listen_loop()
        finally:
            await self.stop()

    async def _listen_loop(self) -> None:
        """Doimiy tinglash: uyg'otuvchi so'z va qarsakni kutadi."""
        frame_count = 0

        async for frame in self.mic.frames():
            if self._shutdown.is_set():
                return

            # Jarvis gapirayotganda o'z ovozidan uyg'onmasligi kerak.
            if self.speaker.speaking:
                continue

            # Har uchinchi kadrda daraja yuboramiz — UI'ni ortiqcha yuklamaslik uchun.
            frame_count += 1
            if frame_count % 3 == 0 and self.bus.state is State.IDLE:
                await self.bus.level(frame_level(frame))
            # Kadr kelayotgan bo'lsa mikrofon tirik. Jimlikni ham "ishlayapti"
            # deb ko'rsatmaslik uchun faqat ovoz bo'lganda chaqnaydi.
            if frame_count % 10 == 0 and frame_level(frame) > 0.04:
                await self.health.ping(System.MIC)

            # Rejalashtiruvchidan kelgan eslatmalar — faqat bo'sh vaqtda,
            # foydalanuvchining gapini bo'lmasdan.
            if not self.proactive.empty() and self.bus.state is State.IDLE:
                await self._deliver_proactive()
                continue

            source = ""
            if self._activate.is_set():
                self._activate.clear()
                source = "bosish"
            elif self._wake is not None and (hit := self._wake.push(frame)) is not None:
                if await self._wake_confirmed(hit):
                    source = "so'z"
                    await self.health.ping(System.WAKE)
            elif self._clap is not None and self._clap.push(frame):
                source = "qarsak"

            if source:
                await self._session(source)
                # Seansdan keyin buferlarni tozalaymiz — eski audio yangi
                # seansga o'tib ketmasin.
                self.mic.drain()
                if self._wake is not None:
                    self._wake.reset()
                if self._clap is not None:
                    self._clap.reset()

    # --- Ovozli tasdiq ---

    async def _on_bus_message(self, message: dict[str, Any]) -> None:
        """Shinadagi tasdiq so'rovlarini ovozli oqimga uzatadi."""
        kind = message.get("type")
        if kind == "confirm":
            # Darvoza javobni kutib turadi, ya'ni bir paytda faqat bitta
            # tasdiq bo'ladi. Har ehtimolga eski vazifani to'xtatamiz.
            if self._confirm_task is not None and not self._confirm_task.done():
                self._confirm_task.cancel()
            self._confirm_task = asyncio.create_task(self._voice_confirm(message))
        elif kind == "confirm_closed" and self._confirm_task is not None:
            # Tugma bosildi yoki vaqt tugadi — tinglashning ma'nosi qolmadi.
            if not self._confirm_task.done():
                self._confirm_task.cancel()

    async def _voice_confirm(self, message: dict[str, Any]) -> None:
        """Tasdiq savolini ovozda beradi va javobni ovozdan o'qiydi.

        Foydalanuvchi kompyuter oldida bo'lmasligi mumkin — tugma bosishni
        talab qilsak, jarayon shu yerda qotib qolardi. Tugmalar ishlashda
        davom etadi: qaysi biri oldin javob bersa, o'sha hal qiladi.
        """
        confirm_id = str(message.get("id", ""))
        action = str(message.get("action", ""))
        detail = str(message.get("detail", ""))

        try:
            await self._speak(consent_prompt(action, detail))

            for _attempt in range(self._vc_attempts):
                if self._shutdown.is_set() or not self.bus.is_pending(confirm_id):
                    return

                # Savolning aks sadosi javob sifatida o'qilmasin.
                self.mic.drain()
                text = await self._capture_utterance(patience_sec=self._vc_listen)

                if not self.bus.is_pending(confirm_id):
                    return

                verdict = parse_consent(text)
                if verdict is None:
                    # Noaniq javobda hech qachon "ruxsat" deb taxmin qilmaymiz.
                    if text:
                        await self._speak("Ha yoki yo'q deb ayting.")
                    continue

                self.bus.resolve_confirm(confirm_id, verdict)
                await self._speak("Xo'p." if verdict else "Bekor qildim.")
                return

            # Javob ololmadik — qaror tugmaga yoki vaqt tugashiga qoladi
            # (vaqt tugasa, darvoza RAD deb hisoblaydi).
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Ovozli tasdiqda xato")

    async def _wake_confirmed(self, hit: WakeHit) -> bool:
        """Chaqiruv haqiqiy ekanini aniqlaydi.

        Ishonchli ball — darhol `True`. Shubhali ball esa matn bilan
        tekshiriladi: oxirgi ikki soniyani STT'ga beramiz va «jarvis» ismi
        eshitildimi, shuni qaraymiz. Shu yo'l bilan tayyor model bilmagan
        «salom jarvis» va «hi jarvis» ham ishlaydi, chegarani pasaytirib
        televizor ovozidan uyg'onib ketmasdan.
        """
        if hit.confident:
            log.info("Chaqiruv: ishonchli (ball %.2f)", hit.score)
            return True
        if self._phrases is None:
            return False

        now = asyncio.get_running_loop().time()
        if now - self._last_verify < self._verify_cooldown:
            log.debug("Shubhali chaqiruv (%.2f) — tekshirish sovish davrida", hit.score)
            return False
        self._last_verify = now

        audio = self.mic.recent(ms=self._verify_window_ms)
        if audio.size == 0:
            return False

        async with self.health.busy(System.STT):
            text = await transcribe_guarded(self.stt, audio, self.config.sample_rate)

        if self._phrases.matches(text):
            log.info("Chaqiruv tasdiqlandi: «%s» (ball %.2f)", text, hit.score)
            return True

        log.info("Shubhali chaqiruv rad etildi: «%s» (ball %.2f)", text, hit.score)
        return False

    async def _deliver_proactive(self) -> None:
        """Navbatdagi eslatmalarni aytadi."""
        while not self.proactive.empty():
            item = self.proactive.get_nowait()
            log.info("Proaktiv xabar (%s): %s", item.kind, item.text)

            if item.notify:
                try:
                    if sys.platform == "darwin":
                        from .tools import macos as platform_tools
                    else:
                        from .tools import windows as platform_tools
                    await platform_tools.notify("Jarvis", item.text[:200])
                except Exception:
                    log.debug("Bildirishnoma ko'rsatilmadi", exc_info=True)

            await self._play_chime()
            await self._speak(item.text)

            if self._interrupted:
                # Foydalanuvchi eslatmani bo'lib javob bermoqda — bu suhbat
                # boshlanishi. Qolgan eslatmalar navbatda turadi, keyin aytiladi.
                log.info("Eslatma bo'lindi — foydalanuvchi javob bermoqda")
                await self._converse()
                return

        await self.bus.set_state(State.IDLE)

    async def _session(self, source: str = "so'z") -> None:
        """Uyg'onish: signal, salomlashish, so'ng suhbat."""
        await self.bus.set_state(State.WAKE)
        # To'liq ekranli sahnani aniq buyruq bilan ochamiz — chaqiruv qanday
        # kelganidan qat'i nazar (so'z, qarsak yoki tugma).
        await self.bus.hud("show")
        await self._play_chime()

        if not self._greeted:
            self._greeted = True
            await self._speak(
                FIRST_GREETING.format(
                    user=self.config.get("identity.user_name", ""),
                    name=self.config.get("identity.name", "Jarvis"),
                )
            )
        else:
            # Qarsak bilan chaqirilganda kinodagidek rasmiyroq javob beramiz.
            pool = CLAP_GREETINGS if source == "qarsak" else GREETINGS
            await self._speak(random.choice(pool))

        await self._converse()

    async def _converse(self) -> None:
        """Tinglash -> javob -> yana tinglash.

        Javobdan keyin Jarvis darhol jim bo'lib qolmaydi — bir necha soniya
        tinglab turadi. Foydalanuvchi davom etsa, uyg'otuvchi so'zni qaytadan
        aytish shart emas; jim bo'lsa, o'zi kutish holatiga qaytadi.
        """
        follow_up = bool(self._talk.get("follow_up", True))
        max_turns = int(self._talk.get("max_turns", 12))
        turn = 0
        closed = False

        while not self._shutdown.is_set():
            # Gapni bo'lgan foydalanuvchi allaqachon gapirib turgan bo'ladi —
            # unda kutish oynasi kerak emas, darhol tinglaymiz.
            text = await self._capture_utterance(patience_sec=self._patience(turn))

            if not text:
                # Birinchi navbatda jimlik — "eshitmadim" deb aytamiz.
                # Keyingi navbatlarda bu shunchaki suhbat tugagani.
                if turn == 0:
                    await self._speak(random.choice(NOT_UNDERSTOOD))
                else:
                    log.info("Suhbat tugadi: %d navbat", turn)
                break

            # «To'xta» — gapirishni to'xtat, lekin suhbatni yopma. Ikkisini
            # ajratmasak, foydalanuvchi har «to'xta» deganda qaytadan
            # chaqirishga majbur bo'lardi.
            if is_stop_speaking(text):
                log.info("Ovoz bilan to'xtatildi: «%s»", text)
                self.speaker.stop()
                continue

            # «Bekor qil» / «cancel» — bu miyaga bormaydi. Uni Claude'ga
            # yuborib javob kutish sekin ham, keraksiz ham.
            if is_end_of_conversation(text):
                log.info("Suhbat ovoz bilan yakunlandi: «%s»", text)
                await self._speak(random.choice(FAREWELLS))
                closed = True
                break

            await self._handle_utterance(text)
            turn += 1

            if not follow_up:
                break
            if turn >= max_turns:
                log.info("Suhbat navbatlari chegarasiga yetdi (%d)", max_turns)
                break

        # Sahna faqat aniq yakunlanganda yopiladi. Jimlik bilan tugagan
        # suhbatdan keyin oyna ochiq qoladi — foydalanuvchi davom ettirishi
        # mumkin va u har safar qaytadan ochilib-yopilib turmasligi kerak.
        if closed:
            await self.bus.hud("hide")
        await self.bus.set_state(State.IDLE)

    def _patience(self, turn: int) -> float:
        """Foydalanuvchi gapirishini shuncha soniya kutamiz.

        Birinchi navbatda ko'proq kutamiz — odam uyg'otuvchi so'zni aytib,
        keyin o'ylanib qolishi mumkin. Javobdan keyingi oyna qisqaroq,
        aks holda Jarvis xonani keraksiz uzoq tinglab turardi.
        """
        if self._interrupted:
            return 2.0
        if turn == 0:
            return float(self._talk.get("first_wait_sec", 6))
        return float(self._talk.get("follow_up_sec", 8))

    @asynccontextmanager
    async def _ducked(self):
        """Tinglash davomida tizim ovozini pasaytiradi.

        Musiqa chalinayotganda mikrofon uni ham eshitadi va foydalanuvchi
        ovozi uning ichida yo'qoladi — STT musiqani matnga aylantirib,
        «tushunmadim» chiqadi. Aks sadoni dasturiy yo'q qilish (AEC) og'ir
        ish; ovozni pasaytirish esa arzon va aynan shu muammoni yechadi.

        Eski daraja saqlanadi va tinglash tugagach qaytariladi — hatto xato
        bo'lsa ham, aks holda musiqa jim bo'lib qolardi.
        """
        if not self._duck_enabled or sys.platform != "darwin":
            # Ovoz pasaytirish hozircha faqat macOS'da amalga oshiriladi
            # (Windows'da absolyut darajani o'qish/yozish uchun qo'shimcha
            # COM qatlami kerak). Windows'da tinglash shunday davom etadi.
            yield
            return

        from .tools import macos

        previous: int | None = None
        try:
            previous = await macos.get_volume()
            if previous > self._duck_level:
                await macos.set_volume(self._duck_level)
        except macos.MacOsError:
            log.debug("Ovoz balandligi boshqarilmadi", exc_info=True)
            previous = None

        try:
            yield
        finally:
            if previous is not None:
                with suppress(Exception):
                    await macos.set_volume(previous)

    async def _capture_utterance(self, patience_sec: float = 6.0) -> str:
        """Foydalanuvchini tinglaydi va aytganini matnga aylantiradi."""
        async with self._ducked():
            return await self._listen(patience_sec)

    async def _listen(self, patience_sec: float) -> str:
        await self.bus.set_state(State.LISTENING)

        endpoint_cfg = self.config.section("audio.endpointing")
        endpointer = Endpointer(
            detector=build_speech_detector(endpoint_cfg, self.config.sample_rate),
            frame_ms=int(self.config.get("audio.frame_ms", 20)),
            silence_ms=int(endpoint_cfg.get("silence_ms", 900)),
            max_utterance_sec=float(endpoint_cfg.get("max_utterance_sec", 30)),
        )
        # Gapni bo'lgan bo'lsa, aytilgan birinchi so'zlar shu buferda —
        # ular yo'qolmasligi kerak.
        endpointer.prime(self.mic.take_preroll())
        self._interrupted = False

        # Foydalanuvchi umuman gapirmasa, cheksiz kutib qolmaymiz.
        frame_ms = max(1, int(self.config.get("audio.frame_ms", 20)))
        silence_budget = int(patience_sec * 1000 / frame_ms)
        idle_frames = 0

        async for frame in self.mic.frames():
            if self._shutdown.is_set():
                return ""

            endpointer.push(frame)
            await self.bus.level(frame_level(frame))

            if endpointer.finished:
                break
            if not endpointer.started:
                idle_frames += 1
                if idle_frames >= silence_budget:
                    log.info("Hech kim gapirmadi, seans bekor qilindi")
                    return ""

        await self.bus.set_state(State.THINKING)
        audio = endpointer.result()
        async with self.health.busy(System.STT):
            text = await transcribe_guarded(self.stt, audio, self.config.sample_rate)

        if text:
            log.info("Eshitildi: %s", text)
            await self.bus.transcript(text)
        return text

    async def _handle_utterance(self, text: str, remote: str | None = None) -> None:
        """Matnni miyaga beradi va javobni ovozga chiqaradi.

        `remote` berilgan bo'lsa, javob o'sha mijozga (telefonga) yuboriladi,
        kompyuter dinamigidan chiqmaydi.
        """
        await self.bus.set_state(State.THINKING)
        spoke_anything = False
        answer = self.brain.ask(text)

        try:
            # Miya siferblati javob oqib kelayotgan butun davr davomida yonadi.
            async with self.health.busy(System.BRAIN):
                async for sentence in answer:
                    if self._shutdown.is_set():
                        return
                    spoke_anything = True
                    await self._speak(sentence, remote=remote)
                    if self._interrupted:
                        # Foydalanuvchi gapni bo'ldi — qolgan gaplarni aytmaymiz.
                        # Javobning to'liq matni `last_reply` da qoladi.
                        log.info("Javob to'xtatildi — foydalanuvchi gapirdi")
                        break
        except Exception as exc:
            log.exception("Javob olishda xato")
            await self.bus.set_state(State.ERROR)
            # Xatoning turi ("APIError") foydalanuvchiga hech nima demaydi.
            # Tanish sabab bo'lsa, nima qilish kerakligini aytamiz.
            advice = hint_for(f"{type(exc).__name__}: {exc}")
            if advice:
                await self.bus.log_line(advice, level="error")
            await self._speak(ERROR_REPLY.format(error=type(exc).__name__), remote=remote)
        finally:
            # Yarmida to'xtatilgan generator SDK seansini ochiq qoldirmasin.
            await answer.aclose()
            if not spoke_anything and self.brain.last_reply and not self._interrupted:
                await self._speak(self.brain.last_reply, remote=remote)
            if remote is not None:
                await self.ui.send_to(remote, {"type": "done"})
            # Bo'lingan bo'lsa, sikl darhol tinglashga o'tadi — oradagi
            # IDLE chaqnashi HUD'da keraksiz sakrash bo'lardi.
            if not self._interrupted:
                await self.bus.set_state(State.IDLE)

    # --- Ovoz chiqarish ---

    async def _speak(self, text: str, remote: str | None = None) -> None:
        """Matnni ovozga chiqaradi va UI'ga ko'rsatadi.

        `remote` berilgan bo'lsa, ovoz kompyuterda chalinmaydi — PCM sifatida
        o'sha mijozga (telefonga) yuboriladi.
        """
        text = text.strip()
        if not text:
            return

        await self.bus.set_state(State.SPEAKING)
        await self.bus.say(text)
        log.info("Jarvis: %s", text)

        if remote is not None:
            await self._speak_remote(text, remote)
            return

        def on_level(value: float) -> None:
            # `Speaker` buni event loop oqimidan chaqiradi, shuning uchun bu yerda
            # to'g'ridan-to'g'ri vazifa yaratsa bo'ladi. Kutmaymiz: daraja yangilanishi
            # animatsiya uchun, kechiksa yoki yo'qolsa ham ijroni to'xtatmasligi kerak.
            asyncio.ensure_future(self.bus.level(value))

        # Ijro davomida mikrofonni kuzatamiz: foydalanuvchi gapirsa to'xtaymiz.
        # Bu paytda asosiy sikl `_speak`ni kutib turadi, ya'ni kadrlarni
        # boshqa hech kim o'qimaydi — navbat uchun kurash bo'lmaydi.
        watcher = None
        if self._barge_in is not None:
            watcher = asyncio.create_task(self._watch_for_barge_in())

        try:
            async with self.health.busy(System.TTS):
                await self.speaker.play(self.tts.stream(text), self.tts.sample_rate, on_level)
        except Exception:
            log.exception("Ovozga chiqarib bo'lmadi")
            await self.bus.log_line(f"[ovozsiz] {text}", level="warn")
        finally:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher

    async def _watch_for_barge_in(self) -> None:
        """Ijro paytida foydalanuvchi gapini kutadi va gapirsa ijroni to'xtatadi."""
        assert self._barge_in is not None
        self._barge_in.reset()

        # Bu yerda o'qilgan kadrlar yo'qolmaydi: preroll buferi har bir kadrni
        # o'qilganidan qat'i nazar saqlaydi, shuning uchun bo'lingandan keyin
        # `_capture_utterance` aytilgan birinchi so'zni ham oladi.
        async for frame in self.mic.frames():
            if self._barge_in.push(frame):
                self._interrupted = True
                self.speaker.stop()
                await self.health.ping(System.MIC)
                return

    async def _speak_remote(self, text: str, client: str) -> None:
        """Javobni sintez qilib, mijozga PCM sifatida yuboradi."""
        try:
            async with self.health.busy(System.TTS):
                chunks = [chunk async for chunk in self.tts.stream(text) if chunk.size]
        except Exception:
            log.exception("Telefon uchun ovoz sintez qilinmadi")
            # Ovoz bo'lmasa ham matnni yuboramiz — foydalanuvchi o'qiy oladi.
            await self.ui.send_to(client, {"type": "reply_text", "text": text})
            return

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        await self.ui.send_to(client, {"type": "reply_text", "text": text})
        if audio.size:
            await self.ui.send_audio(client, audio, self.tts.sample_rate)

    async def _play_chime(self) -> None:
        """Uyg'onish signalini chaladi."""

        async def one_chunk():
            yield chime(self.tts.sample_rate)

        try:
            await self.speaker.play(one_chunk(), self.tts.sample_rate)
        except Exception:
            log.debug("Signal chalinmadi", exc_info=True)


async def amain() -> int:
    config = load_config()
    jarvis = Jarvis(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def request_stop() -> None:
        log.info("To'xtatish signali olindi")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass  # Windows

    runner = asyncio.create_task(jarvis.run())
    waiter = asyncio.create_task(stop_event.wait())

    done, pending = await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)

    if runner not in done:
        jarvis.request_stop()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
    for task in pending:
        task.cancel()

    if runner in done and runner.exception() is not None:
        log.error("Jarvis yiqildi: %s", runner.exception())
        return 1
    return 0
