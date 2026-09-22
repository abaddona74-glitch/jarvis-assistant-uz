"""Gemini miyasi — kaliti bepul (AI Studio) variant.

Claude Agent SDK o'rniga Google Gemini API'ni ishlatadi. Interfeys
`Brain` bilan bir xil (duck-typing): start/stop/ask/last_reply/interrupt —
shuning uchun `app.py` config'ga qarab ikkinchisini tanlaydi.

Imkoniyatlar:
  * oqimli javob (gap-gap chiqarish — kechikish kam);
  * suhbat tarixi (seans ichida chat sessiyasi);
  * Jarvis vositalari: xotira (remember/recall/search), vazifalar
    (add_task/list_tasks/complete_task/daily_brief), tizim (open_url/
    open_app/say_now) — Gemini native function calling orqali.

Cheklovlar: fayl yozish/shell kagi agent vositalari yo'q (xavfsizlik
nuqtai nazaridan ham soddaroq); SafetyGate shu yerda ishtirok etmaydi.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any, Callable

from ..bus import EventBus
from ..config import Config, require_env
from ..brain.agenda import Agenda
from ..brain.memory import Memory
from .agent import MIN_SENTENCE_CHARS, SENTENCE_END, clean_for_speech
from .prompts import build_system_prompt

log = logging.getLogger("jarvis.brain.gemini")

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiBrain:
    """Gemini API ustida qurilgan miya — Brain interfeysining o'rnini bosadi."""

    def __init__(
        self,
        config: Config,
        bus: EventBus,
        memory: Memory,
        agenda: Agenda,
        gate: Any,  # SafetyGate — Gemini yo'lida hozircha ishlatilmaydi
        announce: Callable[[str], Any],
    ) -> None:
        self.config = config
        self.bus = bus
        self.memory = memory
        self.agenda = agenda
        self.gate = gate
        self.announce = announce

        self._client: Any = None
        self._chat: Any = None
        self._session_id: str = "jarvis-gemini"
        self._last_reply: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interrupted = threading.Event()

    # --- Hayot sikli ---

    async def start(self) -> None:
        from google import genai
        from google.genai import types

        api_key = require_env(
            "GEMINI_API_KEY",
            "Gemini miya uchun (aistudio.google.com/apikey — bepul)",
        )
        self._loop = asyncio.get_running_loop()
        self._client = genai.Client(api_key=api_key)

        model = str(self.config.get("brain.model", DEFAULT_MODEL)) or DEFAULT_MODEL
        self._chat = self._client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=self._build_prompt(),
                temperature=0.6,
                tools=self._tools(),
            ),
        )
        self._session_id = f"jarvis-gemini-{int(asyncio.get_event_loop().time())}"
        log.info("Miya ulandi (gemini: %s)", model)

    async def stop(self) -> None:
        self._chat = None
        self._client = None
        log.info("Miya uzildi (gemini)")

    def _build_prompt(self) -> str:
        prompt = build_system_prompt(
            name=str(self.config.get("identity.name", "Jarvis")),
            user_name=str(self.config.get("identity.user_name", "foydalanuvchi")),
            workspace=str(self.config.workspace),
            memory_context=self.memory.context_block(),
        )
        prompt += (

            "\n\n# Vositalar haqida\n"
            "Sizga vazifa, xotira, aloqa va tizim vositalari berilgan. "
            "«Eslatib qo'y», «ertaga ... degin» kabi gaplar uchun add_task ni "
            "chaqiring. «Meni ... deb eslab qo'y» uchun remember. "
            "«... och» (ilova yoki sayt) uchun open_app yoki open_url. "
            "Javobingiz qisqa va o'zbekcha bo'lsin — u ovozda eshitiladi."
        )
        return prompt

    # --- Jarvis vositalari (Gemini auto function calling) ---

    def _tools(self) -> list[Callable[..., str]]:
        memory = self.memory
        agenda = self.agenda
        loop = self._loop

        def remember(kalit: str, matn: str) -> str:
            """Foydalanuvchi haqidagi faktni uzoq xotiraga saqlaydi.

            Args:
                kalit: qisqa nom, masalan "ismi" yoki "ish joyi".
                matn: faktning o'zi, masalan "ismi Maxmudbek".
            """
            memory.remember(kalit, matn)
            return f"Saqlandi: {kalit}"

        def recall(kalit: str) -> str:
            """Faktni xotiradan kalit bo'yicha o'qiydi."""
            value = memory.recall(kalit)
            return value if value is not None else "Xotirada bunday fakt yo'q"

        def search_memory(query: str) -> str:
            """Xotiradan so'z bo'yicha faktrlarni qidiradi."""
            facts = memory.search(query, limit=5)
            if not facts:
                return "Hech narsa topilmadi"
            return "\n".join(f"{f.key}: {f.value}" for f in facts)

        def add_task(title: str, when: str = "", project: str = "") -> str:
            """Vazifa/eslatma qo'shadi. Vaqt ko'rsatilsa o'sha paytda aytadi.

            Args:
                title: vazifa matni, masalan "Alisherga qo'ng'iroq qil".
                when: vaqt insonga tushunarli ko'rinishda, masalan
                    "ertaga 09:00" yoki "bugun 18:30". Bo'sh bo'lsa vaqtsiz.
                project: loyiha nomi (ixtiyoriy).
            """
            from .agenda import parse_when

            stamp = parse_when(when or None)
            task = agenda.add_task(title, project=project or None, remind_at=stamp)
            return f"Vazifa #{task.id} qo'shildi: {task.title}"

        def list_tasks() -> str:
            """Ochiq vazifalar ro'yxatini beradi."""
            tasks = agenda.list_tasks(status="open", limit=10)
            if not tasks:
                return "Ochiq vazifa yo'q"
            return "\n".join(f"#{t.id} {t.title}" for t in tasks)

        def complete_task(task_id: int) -> str:
            """Vazifani bajarilgan deb belgilaydi."""
            task = agenda.complete_task(task_id)
            return f"Bajarildi: {task.title}" if task else "Bunday vazifa topilmadi"

        def daily_brief() -> str:
            """Bugungi kun xulosasi: vazifalar va faol loyihalar."""
            return agenda.daily_brief()

        def open_url(havola: str) -> str:
            """Havolani standart brauzerda ochadi. «Google och» uchun
            havola='https://www.google.com'."""
            import sys

            if sys.platform == "darwin":
                from ..tools import macos as platform
            else:
                from ..tools import windows as platform
            _schedule(platform.open_url(havola))
            return f"Ochildi: {havola}"

        def open_app(nom: str) -> str:
            """Kompyuterda ilova ochadi, masalan 'notepad', 'calc', 'chrome'."""
            import sys

            if sys.platform == "darwin":
                from ..tools import macos as platform
            else:
                from ..tools import windows as platform
            _schedule(platform.open_app(nom))
            return f"Ochildi: {nom}"

        def say_now(matn: str) -> str:
            """Jarvis'ga foydalanuvchiga darhol gapirg'izadi (uzoq ish tugaganda)."""
            _schedule(self.announce(matn))
            return "Aytildi"

        def _schedule(coro: Any) -> None:
            assert loop is not None
            asyncio.run_coroutine_threadsafe(coro, loop)

        return [
            remember, recall, search_memory,
            add_task, list_tasks, complete_task, daily_brief,
            open_url, open_app, say_now,
        ]

    # --- Suhbat ---

    async def ask(self, text: str) -> AsyncIterator[str]:
        """Savol beradi, javobni gap-gap qaytaradi (Brain.ask bilan bir xil shartnoma)."""
        if self._chat is None:
            raise RuntimeError("Miya ishga tushirilmagan — avval `start()` chaqiring")

        from datetime import datetime

        from .agent import WEEKDAYS_UZ

        moment = datetime.now()
        stamped = (
            f"[hozir: {moment.strftime('%Y-%m-%d %H:%M')}, {WEEKDAYS_UZ[moment.weekday()]}]\n"
            f"{text}"
        )

        self._interrupted.clear()
        self.memory.add_turn(self._session_id, "user", text)

        # API oqimini alohida ipda yuritamiz — asosiy sikl bloklanmasin.
        q: queue.Queue[tuple[str, str]] = queue.Queue()

        def worker() -> None:
            try:
                for chunk in self._chat.send_message_stream(stamped):
                    piece = getattr(chunk, "text", None)
                    if piece:
                        q.put(("chunk", piece))
                q.put(("done", ""))
            except Exception as exc:  # noqa: BLE001
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        buffer = ""
        full: list[str] = []

        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "error":
                raise RuntimeError(f"Gemini xatosi: {payload}")
            if kind == "done" or self._interrupted.is_set():
                break

            full.append(payload)
            buffer += payload
            while True:
                match = SENTENCE_END.search(buffer)
                if match is None:
                    break
                sentence = buffer[: match.start()].strip()
                buffer = buffer[match.end():]
                spoken = clean_for_speech(sentence)
                if len(spoken) >= MIN_SENTENCE_CHARS:
                    yield spoken
                elif spoken:
                    buffer = spoken + " " + buffer

        tail = clean_for_speech(buffer)
        if tail and not self._interrupted.is_set():
            yield tail

        self._last_reply = "".join(full).strip()
        if self._last_reply:
            self.memory.add_turn(self._session_id, "assistant", self._last_reply)

    @property
    def last_reply(self) -> str:
        return self._last_reply

    async def interrupt(self) -> None:
        """Joriy javobni to'xtatadi (foydalanuvchi «to'xta» desa)."""
        self._interrupted.set()

    async def refresh_memory_context(self) -> None:
        """Yangi faktlarni chat'ga oddiy xabar sifatida yuboradi."""
        if self._chat is None:
            return
        context = self.memory.context_block()
        if context:
            self._chat.send_message("[tizim eslatmasi] Xotira yangilandi:\n" + context)
