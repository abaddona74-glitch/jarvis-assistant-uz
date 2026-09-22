"""Windows bilan ishlash: bildirishnoma, ilovalar, brauzer, media tugmalari.

`macos.py` bilan bir xil interfeys (duck-typing): server.py platformaga qarab
shu yoki o'sha modulni ishlatadi. Istisno klassi ham xuddi shu nomda —
`except MacOsError` bloklari ikkala platformada ishlashi uchun.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess

log = logging.getLogger("jarvis.tools.windows")


class MacOsError(RuntimeError):
    """Windows amali bajarilmadi (nom macOS versiyasi bilan mos ushlab turilgan)."""


# PowerShell'ga satr xavfsiz uzatish uchun qo'shtirnoq ichiga olamiz.
def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


async def _ps(script: str, timeout: float = 20.0) -> str:
    """PowerShell skriptini bajaradi va stdout qaytaradi."""

    def call() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )

    try:
        result = await asyncio.to_thread(call)
    except subprocess.TimeoutExpired as exc:
        raise MacOsError(f"Buyruq {timeout:.0f} soniyada tugamadi") from exc

    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise MacOsError(message[:300] or f"Buyruq {result.returncode} kodi bilan tugadi")
    return (result.stdout or "").strip()


# --- Bildirishnomalar ---

async def notify(title: str, message: str, subtitle: str = "") -> None:
    """Windows toast bildirishnomasini ko'rsatadi."""
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] | Out-Null; "
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
        f"$t.SelectSingleNode(\"//text[@id='1']\").AppendChild("
        f"$t.CreateTextNode({_ps_quote(title)})) | Out-Null; "
        f"$t.SelectSingleNode(\"//text[@id='2']\").AppendChild("
        f"$t.CreateTextNode({_ps_quote(message)})) | Out-Null; "
        "$n = [Windows.UI.Notifications.ToastNotification]::new($t); "
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        + _ps_quote("Jarvis") + ").Show($n)"
    )
    try:
        await _ps(script)
    except MacOsError:
        # Toast ishlamasa jim o'tamiz — bildirishnoma muhim amal emas.
        log.debug("Toast bildirishnoma chiqmadi", exc_info=True)


# --- Ilovalar ---

async def open_app(name: str) -> None:
    """Ilovani nomi (yoki bajariladigan fayli) bilan ochadi."""
    name = name.strip()
    if not name:
        raise MacOsError("Ilova nomi bo'sh")

    # `start` ilova nomi bilan ham, fayl/buyruq bilan ham ishlaydi.
    await asyncio.to_thread(_open_via_start, name)


def _open_via_start(name: str) -> None:
    subprocess.Popen(
        ["cmd", "/c", "start", "", name],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
    )


async def quit_app(name: str) -> None:
    result = await _ps(
        f"Get-Process -Name {_ps_quote(name)} -ErrorAction SilentlyContinue | Stop-Process -Force"
    )
    del result


async def frontmost_app() -> str:
    """Hozir faol oynani egallagan jarayon nomini qaytaradi."""
    result = await _ps(
        "$sig = '[DllImport(\"user32.dll\")] public static extern IntPtr "
        "GetForegroundWindow(); [DllImport(\"user32.dll\")] public static extern "
        "int GetWindowThreadProcessId(IntPtr h, out int id);'; "
        "Add-Type -MemberDefinition $sig -Name W -Namespace Win; "
        "$h = [Win.W]::GetForegroundWindow(); "
        "$procId = 0; [Win.W]::GetWindowThreadProcessId($h, [ref]$procId) | Out-Null; "
        "(Get-Process -Id $procId).ProcessName"
    )
    return result or "noma'lum"


async def send_imessage(recipient: str, text: str) -> None:
    """Windows'da iMessage yo'q — tushunarli xato beradi."""
    raise MacOsError("iMessage faqat macOS'da ishlaydi. Telegram yuborish (send_telegram) ishlatilsin.")


async def open_url(url: str) -> None:
    """Havolani standart brauzerda ochadi."""
    if not url.startswith(("http://", "https://")):
        raise MacOsError("Faqat http/https havolalari ochiladi")
    _open_via_start(url)


# --- Media tugmalari ---

VK_MEDIA_PLAY_PAUSE = 0xB3


async def _key(vk: int) -> None:
    script = (
        "$sh = New-Object -ComObject WScript.Shell; "
        f"$sh.SendKeys([char]{vk})"
    )
    await _ps(script)


async def playpause() -> None:
    """Ijroni to'xtatadi yoki davom ettiradi (media tugmasi)."""
    await _key(VK_MEDIA_PLAY_PAUSE)
