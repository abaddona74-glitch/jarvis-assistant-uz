"""Jarvis'ning o'z asboblari — Claude Agent SDK ichidagi MCP serveri sifatida.

Bu yerdagi asboblar Claude'ning tayyor asboblari (Read/Write/Bash/WebSearch)
ustiga qo'shiladi. To'rt guruh:

    xotira    — nimani bilaman (faktlar, oldingi suhbatlar)
    agenda    — nima qilish kerak (loyihalar, vazifalar, eslatmalar)
    aloqalar  — kim bilan bog'lanaman (Telegram, telefon)
    tizim     — macOS, Shortcuts, kanallar
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

from ..brain.agenda import Agenda, format_when, parse_when
from ..brain.memory import Memory

from . import channels, media

# Platforma moduli: macOS'da AppleScript, Windows'da PowerShell/win32.
# Interfeys bir xil (duck-typing) — open_app, open_url, notify, frontmost_app,
# playpause... Istisno klassi ham xuddi shu nomda (MacOsError), shuning uchun
# `except MacOsError` bloklari ikkala platformada ishlaydi.
if sys.platform == "darwin":
    from . import macos as platform
else:
    from . import windows as platform

log = logging.getLogger("jarvis.tools")

SERVER_NAME = "jarvis"

# Tashqi dunyoga ta'sir qilmaydigan asboblar — darvoza ularni tez o'tkazadi.
READ_ONLY = ToolAnnotations(readOnlyHint=True)

# Tasdiq so'ramasdan ishlatiladigan asboblar ro'yxati (`allowed_tools` uchun).
READ_ONLY_TOOLS = [
    "recall", "search_memory",
    "list_projects", "list_tasks", "daily_brief",
    "list_contacts", "find_contact",
    "frontmost_app", "list_shortcuts",
    "open_url",
]


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _fail(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _json(payload: Any) -> dict[str, Any]:
    return _ok(json.dumps(payload, ensure_ascii=False, indent=2))


def _memory_tools(memory: Memory) -> list[Any]:
    """Nimani bilaman."""

    @tool(
        "remember",
        "Foydalanuvchi haqidagi barqaror faktni xotiraga saqlaydi. Keyingi suhbatlarda "
        "ham eslab qolinadi. Masalan: kalit='ish_vaqti', qiymat='9:00 dan 18:00 gacha'.",
        {"kalit": str, "qiymat": str, "toifa": str},
    )
    async def remember(args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("kalit", "")).strip()
        value = str(args.get("qiymat", "")).strip()
        if not key or not value:
            return _fail("`kalit` va `qiymat` bo'sh bo'lmasligi kerak")
        memory.remember(key, value, str(args.get("toifa") or "umumiy"))
        return _ok(f"Eslab qoldim: {key} = {value}")

    @tool("recall", "Xotiradan faktni kalit bo'yicha o'qiydi.", {"kalit": str},
          annotations=READ_ONLY)
    async def recall(args: dict[str, Any]) -> dict[str, Any]:
        value = memory.recall(str(args.get("kalit", "")))
        return _ok(value if value is not None else "Bunday fakt xotirada yo'q")

    @tool("forget", "Faktni xotiradan o'chiradi.", {"kalit": str})
    async def forget(args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("kalit", ""))
        return _ok(f"O'chirildi: {key}" if memory.forget(key) else "Bunday fakt topilmadi")

    @tool(
        "search_memory",
        "Xotiradagi faktlar va oldingi suhbatlar ichidan matn bo'yicha qidiradi. "
        "«kechagi hisobotni yana yubor» kabi so'rovlarda ishlating.",
        {"soz": str},
        annotations=READ_ONLY,
    )
    async def search_memory(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("soz", "")).strip()
        if not query:
            return _fail("Qidiruv so'zi bo'sh")
        facts = memory.search(query, limit=15)
        turns = memory.search_turns(query, limit=10)
        if not facts and not turns:
            return _ok("Hech narsa topilmadi")
        return _json({
            "faktlar": [{"kalit": f.key, "qiymat": f.value} for f in facts],
            "suhbatlar": turns,
        })

    return [remember, recall, forget, search_memory]


def _agenda_tools(agenda: Agenda, announce: Callable[[str], Any]) -> list[Any]:
    """Nima qilish kerak — loyihalar, vazifalar, eslatmalar."""

    @tool(
        "set_project",
        "Loyihani yaratadi yoki holatini yangilaydi. Faqat o'zgartirmoqchi bo'lgan "
        "maydonlarni bering. Holat: faol | kutilmoqda | tugagan | to'xtatilgan. "
        "Muddat formati: 2026-08-15 yoki 2026-08-15T18:00.",
        {"nom": str, "holat": str, "tavsif": str, "keyingi_qadam": str, "muddat": str},
    )
    async def set_project(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("nom", "")).strip()
        if not name:
            return _fail("Loyiha nomi kerak")
        try:
            project = agenda.upsert_project(
                name,
                status=str(args["holat"]) if args.get("holat") else None,
                description=str(args["tavsif"]) if args.get("tavsif") else None,
                next_step=str(args["keyingi_qadam"]) if args.get("keyingi_qadam") else None,
                deadline=parse_when(args.get("muddat")),
            )
        except ValueError as exc:
            return _fail(str(exc))
        return _ok(f"Loyiha saqlandi: {project.name} ({project.status})")

    @tool(
        "list_projects",
        "Loyihalar ro'yxatini va ularning holatini beradi. «Loyihalarim qaysi "
        "bosqichda?» degan savolga shu bilan javob bering.",
        {"holat": str},
        annotations=READ_ONLY,
    )
    async def list_projects(args: dict[str, Any]) -> dict[str, Any]:
        projects = agenda.list_projects(str(args["holat"]) if args.get("holat") else None)
        if not projects:
            return _ok("Loyihalar yo'q")
        return _json([
            {
                "nom": p.name,
                "holat": p.status,
                "keyingi_qadam": p.next_step,
                "muddat": format_when(p.deadline),
                "tavsif": p.description,
            }
            for p in projects
        ])

    @tool(
        "add_task",
        "Vazifa qo'shadi. `vaqt` berilsa, Jarvis o'sha vaqtda o'zi eslatadi — "
        "foydalanuvchi so'ramasa ham. Format: 2026-08-09T10:00. "
        "Takror: kunlik | ish_kunlari | haftalik | oylik.",
        {"vazifa": str, "vaqt": str, "loyiha": str, "takror": str},
    )
    async def add_task(args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("vazifa", "")).strip()
        if not title:
            return _fail("Vazifa matni kerak")
        try:
            task = agenda.add_task(
                title,
                project=str(args["loyiha"]) if args.get("loyiha") else None,
                remind_at=parse_when(args.get("vaqt")),
                repeat=str(args.get("takror") or ""),
            )
        except ValueError as exc:
            return _fail(str(exc))

        when = format_when(task.remind_at)
        return _ok(f"Qo'shildi (#{task.id}): {task.title}" + (f" — {when}" if when else ""))

    @tool(
        "list_tasks",
        "Bajarilmagan vazifalar ro'yxati. Loyiha nomi berilsa, faqat o'shanikilari.",
        {"loyiha": str},
        annotations=READ_ONLY,
    )
    async def list_tasks(args: dict[str, Any]) -> dict[str, Any]:
        tasks = agenda.list_tasks(project=str(args["loyiha"]) if args.get("loyiha") else None)
        if not tasks:
            return _ok("Bajarilmagan vazifa yo'q")
        return _json([
            {
                "id": t.id,
                "vazifa": t.title,
                "loyiha": t.project or "",
                "vaqt": format_when(t.remind_at),
                "takror": t.repeat,
            }
            for t in tasks
        ])

    @tool("complete_task", "Vazifani bajarilgan deb belgilaydi (id bo'yicha).", {"id": int})
    async def complete_task(args: dict[str, Any]) -> dict[str, Any]:
        task = agenda.complete_task(int(args.get("id", 0)))
        if task is None:
            return _fail("Bunday vazifa topilmadi")
        suffix = " Takrorlanuvchi — keyingisi rejaga qo'shildi." if task.repeat else ""
        return _ok(f"Bajarildi: {task.title}.{suffix}")

    @tool("delete_task", "Vazifani o'chiradi (id bo'yicha).", {"id": int})
    async def delete_task(args: dict[str, Any]) -> dict[str, Any]:
        ok = agenda.delete_task(int(args.get("id", 0)))
        return _ok("O'chirildi") if ok else _fail("Bunday vazifa topilmadi")

    @tool(
        "daily_brief",
        "Bugungi kun uchun qisqa xulosa: qanday ishlar bor, qaysi loyihalar faol.",
        {},
        annotations=READ_ONLY,
    )
    async def daily_brief(args: dict[str, Any]) -> dict[str, Any]:
        return _ok(agenda.daily_brief())

    @tool(
        "say_now",
        "Foydalanuvchiga darhol ovozli xabar aytadi. Uzoq ish tugaganda yoki "
        "muhim narsa yuz berganda ishlating — foydalanuvchi so'ramasa ham.",
        {"matn": str},
    )
    async def say_now(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("matn", "")).strip()
        if not text:
            return _fail("Xabar matni kerak")
        await announce(text)
        return _ok("Aytildi")

    return [set_project, list_projects, add_task, list_tasks, complete_task,
            delete_task, daily_brief, say_now]


def _contact_tools(agenda: Agenda) -> list[Any]:
    """Kim bilan bog'lanaman."""

    @tool(
        "save_contact",
        "Aloqani saqlaydi: ism va Telegram chat ID (yoki @username) hamda telefon raqami. "
        "Shundan keyin «Alisherga yoz» deyish kifoya.",
        {"ism": str, "telegram": str, "telefon": str, "izoh": str},
    )
    async def save_contact(args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("ism", "")).strip()
        if not name:
            return _fail("Ism kerak")
        agenda.upsert_contact(
            name,
            telegram=str(args.get("telegram") or ""),
            phone=str(args.get("telefon") or ""),
            note=str(args.get("izoh") or ""),
        )
        return _ok(f"Aloqa saqlandi: {name}")

    @tool("list_contacts", "Saqlangan aloqalar ro'yxati.", {}, annotations=READ_ONLY)
    async def list_contacts(args: dict[str, Any]) -> dict[str, Any]:
        contacts = agenda.list_contacts()
        return _json(contacts) if contacts else _ok("Aloqalar yo'q")

    @tool(
        "find_contact",
        "Aloqani ism bo'yicha topadi. Xabar yuborishdan oldin shu bilan tekshiring.",
        {"ism": str},
        annotations=READ_ONLY,
    )
    async def find_contact(args: dict[str, Any]) -> dict[str, Any]:
        contact = agenda.find_contact(str(args.get("ism", "")))
        if contact is None:
            return _fail(
                "Bunday aloqa topilmadi yoki bir nechta mos keldi. "
                "`list_contacts` bilan ro'yxatni ko'ring."
            )
        return _json(contact)

    return [save_contact, list_contacts, find_contact]


def _system_tools(agenda: Agenda) -> list[Any]:
    """macOS, Shortcuts va tashqi kanallar."""

    @tool(
        "notify",
        "Ekranda macOS bildirishnomasini ko'rsatadi. Foydalanuvchi kompyuter oldida "
        "bo'lmasligi mumkin bo'lgan uzoq ishlarda foydali.",
        {"sarlavha": str, "matn": str},
    )
    async def notify(args: dict[str, Any]) -> dict[str, Any]:
        try:
            await platform.notify(str(args.get("sarlavha", "Jarvis")), str(args.get("matn", "")))
            return _ok("Bildirishnoma ko'rsatildi")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool("open_app", "Kompyuterda ilovani ochadi. Masalan: 'chrome', 'notepad', 'calc'.", {"nom": str})
    async def open_app(args: dict[str, Any]) -> dict[str, Any]:
        try:
            await platform.open_app(str(args.get("nom", "")))
            return _ok(f"{args.get('nom')} ochildi")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "open_url",
        "Havolani standart brauzerda ochadi. «Google och», «youtube.com och» kabi "
        "buyruqlar uchun shu ishlatiladi — masalan havola='https://www.google.com'.",
        {"havola": str},
    )
    async def open_url(args: dict[str, Any]) -> dict[str, Any]:
        try:
            await platform.open_url(str(args.get("havola", "")))
            return _ok("Havola ochildi")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "play_youtube",
        "YouTube'dan qo'shiq yoki videoni topib, brauzerda qo'yadi. "
        "`nima` — qo'shiq nomi va ijrochi. `sekund` berilsa, o'sha joydan boshlaydi. "
        "Masalan: nima='Roshka Ishondingmi', sekund=20.",
        {"nima": str, "sekund": int},
    )
    async def play_youtube(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("nima", "")).strip()
        start = int(args.get("sekund") or 0)
        try:
            video_id, title = await media.find_video(query)
            await platform.open_url(media.watch_url(video_id, start))
        except (media.MediaError, platform.MacOsError) as exc:
            return _fail(str(exc))

        what = title or query
        when = f", {start}-sekunddan" if start else ""
        return _ok(f"Qo'yildi: {what}{when}")

    @tool("close_youtube", "YouTube ochilgan varaqlarni yopadi.", {})
    async def close_youtube(args: dict[str, Any]) -> dict[str, Any]:
        try:
            count = await media.close_youtube()
        except media.MediaError as exc:
            return _fail(str(exc))
        return _ok(f"{count} ta varaq yopildi" if count else "YouTube ochiq emas edi")

    @tool("playpause", "Ijroni to'xtatadi yoki davom ettiradi.", {})
    async def playpause(args: dict[str, Any]) -> dict[str, Any]:
        try:
            await media.playpause()
        except (media.MediaError, platform.MacOsError) as exc:
            return _fail(str(exc))
        return _ok("Bajarildi")

    @tool("frontmost_app", "Hozir qaysi ilova faol ekanini aytadi.", {}, annotations=READ_ONLY)
    async def frontmost_app(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return _ok(await platform.frontmost_app())
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "send_message",
        "SMS/iMessage yuboradi (faqat macOS'da ishlaydi). `kimga` — saqlangan aloqa ismi, "
        "telefon raqami yoki Apple ID. Ism berilsa, aloqalardan raqami topiladi.",
        {"kimga": str, "matn": str},
    )
    async def send_message(args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("kimga", "")).strip()
        text = str(args.get("matn", "")).strip()
        if not target or not text:
            return _fail("`kimga` va `matn` kerak")

        # Ism berilgan bo'lsa, aloqalardan raqamni topamiz.
        if not any(ch.isdigit() for ch in target) and "@" not in target:
            contact = agenda.find_contact(target)
            if contact is None or not contact["telefon"]:
                return _fail(
                    f"«{target}» uchun telefon raqami topilmadi. "
                    f"`save_contact` bilan saqlang yoki raqamni to'g'ridan-to'g'ri bering."
                )
            target = contact["telefon"]

        try:
            await platform.send_imessage(target, text)
            return _ok(f"Xabar yuborildi: {target}")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "send_telegram",
        "Telegram orqali xabar yuboradi. `kimga` bo'sh bo'lsa — foydalanuvchining "
        "o'ziga; ism berilsa — saqlangan aloqaga (uning nomidan yozadi).",
        {"matn": str, "kimga": str},
    )
    async def send_telegram(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("matn", "")).strip()
        if not text:
            return _fail("Xabar matni kerak")

        chat_id = ""
        target = str(args.get("kimga") or "").strip()
        if target:
            contact = agenda.find_contact(target)
            if contact is None or not contact["telegram"]:
                return _fail(
                    f"«{target}» uchun Telegram manzili topilmadi. "
                    f"`save_contact` bilan saqlang."
                )
            chat_id = contact["telegram"]

        try:
            await channels.send_telegram(text, chat_id=chat_id)
            return _ok(f"Telegram'ga yuborildi{f' ({target})' if target else ''}")
        except channels.ChannelError as exc:
            return _fail(str(exc))

    @tool(
        "list_shortcuts",
        "Mavjud macOS Shortcuts qisqa yo'llari ro'yxati. Telefonda amal bajarish "
        "uchun avval shu ro'yxatdan mos qisqa yo'lni toping.",
        {},
        annotations=READ_ONLY,
    )
    async def list_shortcuts(args: dict[str, Any]) -> dict[str, Any]:
        try:
            names = await platform.list_shortcuts()
            return _ok("\n".join(names) if names else "Qisqa yo'llar topilmadi")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "run_shortcut",
        "macOS Shortcuts qisqa yo'lini ishga tushiradi. iCloud orqali sinxronlangan "
        "qisqa yo'llar telefonda ham amal bajarishi mumkin.",
        {"nom": str, "kirish": str},
    )
    async def run_shortcut(args: dict[str, Any]) -> dict[str, Any]:
        try:
            output = await platform.run_shortcut(
                str(args.get("nom", "")), str(args.get("kirish") or "")
            )
            return _ok(output or "Qisqa yo'l bajarildi")
        except platform.MacOsError as exc:
            return _fail(str(exc))

    @tool(
        "call_n8n",
        "n8n webhook'ini chaqiradi va javobini qaytaradi — mavjud avtomatlashtirish "
        "jarayonlarini ishga tushirish uchun.",
        {"malumot": str},
    )
    async def call_n8n(args: dict[str, Any]) -> dict[str, Any]:
        raw = str(args.get("malumot", "{}"))
        try:
            payload = json.loads(raw) if raw.strip().startswith("{") else {"text": raw}
        except json.JSONDecodeError:
            payload = {"text": raw}
        try:
            return _ok(await channels.call_n8n(payload))
        except channels.ChannelError as exc:
            return _fail(str(exc))

    return [notify, open_app, open_url, play_youtube, close_youtube, playpause,
            frontmost_app, send_message, send_telegram,
            list_shortcuts, run_shortcut, call_n8n]


def build_server(
    memory: Memory, agenda: Agenda, announce: Callable[[str], Any]
) -> Any:
    """MCP serverini yaratadi.

    `announce` — Jarvis'ga darhol gapirish imkonini beruvchi qayta chaqiruv
    (asinxron). Uni yadro beradi, chunki ovoz chiqarish yadroga tegishli.
    """
    tools = [
        *_memory_tools(memory),
        *_agenda_tools(agenda, announce),
        *_contact_tools(agenda),
        *_system_tools(agenda),
    ]
    return create_sdk_mcp_server(name=SERVER_NAME, version="0.2.0", tools=tools)


def read_only_tool_names() -> list[str]:
    """Tasdiq so'ramasdan ishlatsa bo'ladigan asboblarning to'liq nomlari."""
    return [f"mcp__{SERVER_NAME}__{name}" for name in READ_ONLY_TOOLS]
