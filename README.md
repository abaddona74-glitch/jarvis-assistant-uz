# Jarvis Assistant — Windows + o'zbekcha ovoz varianti

Ovozli yordamchi: **"Hey Jarvis"** deb murojaat qiling, o'zbekcha buyruq ayting — Jarvis eshitadi, o'ylaydi va javob beradi.

Bu loyiha [kamolbeek/jarvis-assistant](https://github.com/kamolbeek/jarvis-assistant) fork'idir va **Windows uchun moslashtirilgan**:

| O'zgarish | Tafsilot |
|---|---|
| 🇺🇿 **Lokal o'zbekcha STT** | Yangi `faster_whisper_uz` provayderi — faster-whisper (CTranslate2) bilan o'zbekchaga fine-tune qilingan Whisper ishlaydi. Internetsiz, kalitsiz, bepul. |
| 🔊 **Windows TTS** | Yangi `windows` provayderi — SAPI (PowerShell) orqali lokal ovoz chiqarish. Kalitsiz, bepul. |
| 🪟 **Windows'ga moslash** | Apple'ga bog'liq `mlx-whisper` o'rniga CPU'da ishlaydigan faster-whisper. |

Asl loyiha haqida: macOS'ga qaratilgan Jarvis — Claude Agent SDK bilan miya, openWakeWord bilan uyg'otuvchi so'z, Telegram uslubidagi web UI, xotira, rejalashtiruvchi va xavfsizlik darvozasi.

## Bizning variant nima beradi?

- **Eshitish (STT)**: to'liq lokal va tekin — [maqsudxo1ja/uz-whisper-small-stt-v2](https://huggingface.co/maqsudxo1ja/uz-whisper-small-stt-v2) (250 MB int8, o'rtacha PC'da real vaqtga yaqin) yoki kattaroq [hostmepanda/whisper-large-v3-turbo-uzbek-ct2](https://huggingface.co/hostmepanda/whisper-large-v3-turbo-uzbek-ct2) (~800 MB, sifatliroq, biroz sekinroq).
- **Gapirish (TTS)**: Windows SAPI (lokal). Haqiqiy o'zbekcha ovoz uchun Azure F0 bepul tarif tavsiya etiladi (`uz-UZ-SardorNeural`, oyiga 500 000 belgi).
- **Miya (LLM)**: Claude Agent SDK — ANTHROPIC_API_KEY yoki Claude Pro/Max obunasi (asl loyiha shartligicha).

> ⚠️ STT+TTS bepul, lekin "miya" pullik — bu asl loyihaning arxitekturasi.

## Windows'da o'rnatish

Talablar: **Python 3.12** (3.14 hali wheel chiqarmagan: `webrtcvad`, `onnxruntime`), Node.js (orb HUD uchun, ixtiyoriy), mikrofon.

```powershell
git clone https://github.com/abaddona74-glitch/jarvis-assistant-uz.git
cd jarvis-assistant-uz

# 1. Virtual muhit
py -3.12 -m venv .venv
.venv\Scripts\activate

# 2. Paketlar (webrtcvad kompilyatsiya xatosi bersa: pip install webrtcvad-wheels)
pip install -e .

# 3. Sozlanma
copy .env.example .env      # ANTHROPIC_API_KEY ni yozing
copy config\jarvis.example.yaml config\jarvis.yaml
```

`config\jarvis.yaml` ichida (past quvvatli CPU uchun tavsiya):

```yaml
audio:
  input_device: 8          # python -c "import sounddevice as sd; print(sd.query_devices())" bilan toping
  input_gain: 2.0

voice:
  stt:
    provider: "faster_whisper_uz"
    language: "uz"
    model: "maqsudxo1ja/uz-whisper-small-stt-v2"   # kattaroq: hostmepanda/whisper-large-v3-turbo-uzbek-ct2
    device: "cpu"
    compute_type: "int8"
    beam_size: 1
    cpu_threads: 4
  tts:
    provider: "windows"    # lokal SAPI; o'zbekcha ovoz uchun Azure F0: "azure"
```

### Ishga tushirish

```powershell
python -m jarvis doctor    # diagnostika
python -m jarvis           # asosiy rejim — "Hey Jarvis" deb murojaat qiling
```

Brauzer: `http://127.0.0.1:8765` — Telegram uslubidagi chat sahifasi.

## Model tanlash bo'yicha maslahat (CPU)

| Model | int8 hajmi | 7s audio, 4 oqim | Sifat |
|---|---|---|---|
| `maqsudxo1ja/uz-whisper-small-stt-v2` | ~250 MB | ~7s (real vaqt) | yaxshi |
| `hostmepanda/whisper-large-v3-turbo-uzbek-ct2` | ~800 MB | ~35s | eng yaxshi (sekin CPU'lar uchun tavsiya etilmaydi) |

Sinov natijasi (i5-4570T, 4 oqim, int8): small model 7.6s audio → 7.1s transkripsiya.

## Nima ishlaydi / nima ishlamaydi (Windows)

| Qism | Holat |
|---|---|
| Wake word, audio, xotira, scheduler, web UI, xavfsizlik | ✅ |
| Lokal o'zbekcha STT, Windows TTS | ✅ (bu fork qo'shgan) |
| Miya (Claude) | ✅ (kalit/obuna kerak) |
| macOS ilovalar, iMessage, Shortcuts, `say` | ❌ macOS'gina |

## Litsenziya

Asl loyiha bilan bir xil (MIT). STT modellarining litsenziyalari Hugging Face sahifalarida ko'rsatilgan.
