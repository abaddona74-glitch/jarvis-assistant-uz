"""YouTube va media boshqaruvi.

«Falon qo'shiqni qo'y» — ovozli yordamchidan eng ko'p so'raladigan ishlardan
biri. Uni bajarish uchun qidiruv natijasidan birinchi videoning ID'sini
olish kerak.

Nega API kaliti ishlatilmadi: YouTube Data API kalit talab qiladi, kunlik
kvotasi bor va ro'yxatdan o'tish kerak. Qidiruv sahifasining o'zida esa
kerakli ID allaqachon bor — uni o'qish yetarli. Bu yo'lning kamchiligi ham
bor va uni yashirmaymiz: YouTube sahifa tuzilishini o'zgartirsa, naqsh
ishlamay qoladi. Shuning uchun ajratish alohida funksiyada va testlari bilan.
"""

from __future__ import annotations

import logging
import re
import sys
from urllib.parse import quote_plus

log = logging.getLogger("jarvis.tools.media")

# Platforma moduli (server.py bilan bir xil tanlov).
if sys.platform == "darwin":
    from . import macos as platform
else:
    from . import windows as platform


class MediaError(RuntimeError):
    """Media buyrug'i bajarilmadi."""


# YouTube sahifasi ichida video ID'lari shu ko'rinishda uchraydi. ID har doim
# 11 belgi: harflar, raqamlar, `-` va `_`.
_VIDEO_ID = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')

# Sarlavha ham o'sha JSON ichida — foydalanuvchiga nima qo'yilganini aytamiz.
_TITLE = re.compile(r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"')

SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def first_video(html: str) -> tuple[str, str] | None:
    """Qidiruv sahifasidan birinchi videoning (id, sarlavha) sini oladi.

    Sarlavha topilmasa ham ID qaytariladi — asosiysi qo'shiqni qo'yish.
    """
    match = _VIDEO_ID.search(html or "")
    if match is None:
        return None
    title = _TITLE.search(html or "")
    return match.group(1), _unescape(title.group(1)) if title else ""


def _unescape(text: str) -> str:
    """JSON qochish belgilarini ochadi: \\u0026 -> &, \\" -> "."""
    try:
        import json

        return json.loads(f'"{text}"')
    except ValueError:
        return text


def watch_url(video_id: str, start_sec: int = 0) -> str:
    """Ko'rish havolasi. `start_sec` berilsa, o'sha joydan boshlanadi."""
    url = WATCH_URL.format(video_id=video_id)
    if start_sec > 0:
        url += f"&t={int(start_sec)}s"
    return url


async def find_video(query: str) -> tuple[str, str]:
    """YouTube'dan qidiradi va birinchi natijani qaytaradi."""
    import httpx

    if not query.strip():
        raise MediaError("Nimani qidirishni ayting")

    url = SEARCH_URL.format(query=quote_plus(query))
    headers = {
        # Oddiy brauzer sifatida so'raymiz — aks holda YouTube boshqa,
        # ID'siz sahifa qaytarishi mumkin.
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept-Language": "uz,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MediaError(f"YouTube'ga ulanib bo'lmadi: {exc}") from exc

    found = first_video(response.text)
    if found is None:
        raise MediaError(f"«{query}» bo'yicha hech nima topilmadi")
    return found


# --- Brauzerni boshqarish (macOS) ---

# Qaysi brauzer ochiq bo'lsa, o'shani boshqaramiz. Chrome birinchi — u
# ko'pchilikda standart, lekin Safari ham qo'llab-quvvatlanadi.
BROWSERS = ("Google Chrome", "Safari")

_CLOSE_CHROME = '''
tell application "Google Chrome"
    set closed to 0
    repeat with w in windows
        set i to 1
        repeat while i ≤ (count of tabs of w)
            if URL of tab i of w contains "youtube.com" then
                close tab i of w
                set closed to closed + 1
            else
                set i to i + 1
            end if
        end repeat
    end repeat
    return closed
end tell
'''

_CLOSE_SAFARI = '''
tell application "Safari"
    set closed to 0
    repeat with w in windows
        repeat with t in (tabs of w)
            if URL of t contains "youtube.com" then
                close t
                set closed to closed + 1
            end if
        end repeat
    end repeat
    return closed
end tell
'''


async def close_youtube() -> int:
    """YouTube ochilgan barcha varaqlarni yopadi. Nechtasi yopilganini qaytaradi."""
    if sys.platform != "darwin":
        raise MediaError("Varaq bo'yicha yopish hozircha faqat macOS'da ishlaydi")
    from . import macos

    total = 0
    for name, script in (("Google Chrome", _CLOSE_CHROME), ("Safari", _CLOSE_SAFARI)):
        if not await _is_running(name):
            continue
        try:
            result = await macos.osascript(script)
            total += int(result.strip() or 0)
        except (macos.MacOsError, ValueError):
            log.debug("%s da varaq yopilmadi", name, exc_info=True)
    return total


async def _is_running(app_name: str) -> bool:
    """Ilova ochiqmi? Yopiq ilovani AppleScript bilan so'roq qilish uni ochib
    yuboradi — shuning uchun avval tekshiramiz."""
    from . import macos

    script = f'tell application "System Events" to (name of processes) contains "{app_name}"'
    try:
        return (await macos.osascript(script)).strip() == "true"
    except macos.MacOsError:
        return False


async def playpause() -> None:
    """Ijroni to'xtatadi yoki davom ettiradi.

    Windows'da media tugmasi yuboriladi — brauzer old planda bo'lmasa ham ishlaydi.
    macOS'da esa YouTube probel tugmasi bilan boshqariladi, shuning uchun avval
    brauzerni old planga chiqaramiz. Media tugmasini yuborish ham mumkin edi,
    lekin u boshqa pleyerlarga (Music, Spotify) tegib ketardi.
    """
    if sys.platform != "darwin":
        await platform.playpause()
        return

    from . import macos

    for name in BROWSERS:
        if await _is_running(name):
            await macos.osascript(f'tell application "{name}" to activate')
            await macos.osascript('tell application "System Events" to key code 49')
            return
    raise MediaError("Ochiq brauzer topilmadi")
