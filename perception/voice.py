"""Voice loop: spoken question -> Whisper (local, on the GPU) -> vlm.ask over the memory log -> spoken answer.

    python voice.py runs/live/memory.jsonl                        # Enter to start talking, Enter to stop
    python voice.py runs/live/memory.jsonl --text "where is my medicine?"
    python voice.py runs/live/memory.jsonl --arm                  # "bring me my medicine" / "stop" drive the arm
"""
import argparse, queue, re, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

# mlx_whisper and sounddevice are imported where they are used, not here. mlx_whisper is
# Apple-Silicon only, so importing it at module load made the whole file unusable on the
# team's Windows laptop -- including `--text`, which needs no microphone at all.
STT_MODEL = "mlx-community/whisper-base.en-mlx"  # 0.05 s per question once loaded on the M3 Pro
RATE = 16000
FETCH = re.compile(r"\b(bring|fetch|get|grab|hand)\b.*\b(medicine|meds|pills?|bottle|it)\b", re.I)
STOP = re.compile(r"^\W*stop\b", re.I)


def record():
    """Push to talk: records from the default mic between two Enter presses."""
    import sounddevice as sd

    chunks = queue.Queue()
    input("Press Enter and ask your question...")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=lambda d, *_: chunks.put(d.copy())):
        input("Listening. Press Enter when done.")
    return np.concatenate(list(chunks.queue)).ravel() if not chunks.empty() else np.zeros(0, np.float32)


def arm_command(question, url):
    """Fetch and stop requests go to the arm (vla/arm_client.py, served by vla/run_policy.py); None for the rest."""
    if not (STOP.search(question) or FETCH.search(question)):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vla.arm_client import arm, fetch

    try:
        if STOP.search(question):
            arm("stop", url)
            return "Stopping."
        status = fetch(url)
    except OSError:
        return "I can't reach the arm."
    return "Getting it for you." if status["active"] else f"I can't start the arm: {status['last_error']}"


def speak(text):
    """Say the answer out loud, on whichever OS this is. Printing is the fallback:
    a demo that prints the answer is fine, one that crashes on an unknown platform is not."""
    if sys.platform == "darwin":
        subprocess.Popen(["say", "-v", "Samantha", text])
    elif sys.platform == "win32":
        # SAPI via PowerShell: no extra dependency. Single quotes are the only escape needed.
        script = ("Add-Type -AssemblyName System.Speech; "
                  f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text.replace(chr(39), chr(39) * 2)}')")
        subprocess.Popen(["powershell", "-NoProfile", "-Command", script])
    elif shutil.which("espeak"):
        subprocess.Popen(["espeak", text])
    # else: main() has already printed the answer, which is enough to keep a demo going.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("memory", help="memory.jsonl written by memory_pipeline.py")
    ap.add_argument("--text", help="skip the mic and ask this")
    ap.add_argument("--arm", nargs="?", const="http://127.0.0.1:8020", help="send fetch/stop requests to the arm")
    args = ap.parse_args()

    from vlm import ask  # needs ANTHROPIC_API_KEY in perception/.env

    mlx_whisper = None
    if not args.text:
        import mlx_whisper  # Apple Silicon only; --text is the way in on other platforms
        mlx_whisper.transcribe(np.zeros(RATE, np.float32), path_or_hf_repo=STT_MODEL)  # load before the first question
    while True:
        if args.text:
            question, t_end = args.text, time.perf_counter()
        else:
            audio = record()
            t_end = time.perf_counter()
            question = mlx_whisper.transcribe(audio, path_or_hf_repo=STT_MODEL)["text"].strip()
        t_stt = time.perf_counter()
        print(f"Q: {question}")
        if not question:
            continue
        if not (args.arm and (answer := arm_command(question, args.arm))):
            answer, _ = ask(question, args.memory)
        t_answer = time.perf_counter()
        speak(answer)
        print(f"A: {answer}\n(speech-to-text {t_stt - t_end:.2f}s, answer {t_answer - t_stt:.1f}s, "
              f"question to first audio ~{t_answer - t_end:.1f}s)")
        if args.text:
            break


if __name__ == "__main__":
    main()
