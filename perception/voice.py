"""Voice loop: spoken question -> Whisper (local, on the GPU) -> vlm.ask over the memory log -> spoken answer.

    python voice.py runs/live/memory.jsonl                        # Enter to start talking, Enter to stop
    python voice.py runs/live/memory.jsonl --text "where is my medicine?"
"""
import argparse, queue, subprocess, time

import mlx_whisper
import numpy as np
import sounddevice as sd

STT_MODEL = "mlx-community/whisper-base.en-mlx"  # 0.05 s per question once loaded on the M3 Pro
RATE = 16000


def record():
    """Push to talk: records from the default mic between two Enter presses."""
    chunks = queue.Queue()
    input("Press Enter and ask your question...")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=lambda d, *_: chunks.put(d.copy())):
        input("Listening. Press Enter when done.")
    return np.concatenate(list(chunks.queue)).ravel() if not chunks.empty() else np.zeros(0, np.float32)


def speak(text):
    subprocess.Popen(["say", "-v", "Samantha", text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("memory", help="memory.jsonl written by memory_pipeline.py")
    ap.add_argument("--text", help="skip the mic and ask this")
    args = ap.parse_args()

    from vlm import ask  # needs ANTHROPIC_API_KEY in perception/.env
    mlx_whisper.transcribe(np.zeros(RATE, np.float32), path_or_hf_repo=STT_MODEL)  # load the model before the first question
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
        answer, _ = ask(question, args.memory)
        t_answer = time.perf_counter()
        speak(answer)
        print(f"A: {answer}\n(speech-to-text {t_stt - t_end:.2f}s, answer {t_answer - t_stt:.1f}s, "
              f"question to first audio ~{t_answer - t_end:.1f}s)")
        if args.text:
            break


if __name__ == "__main__":
    main()
