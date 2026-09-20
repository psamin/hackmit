"""Full facial-recognition test: enrol from the webcam, then identify.

    python facetest.py Aneesh

Captures three shots with countdowns, enrols the first two under the name, then
identifies the third and reports the similarity numbers the threshold is set from.
Restores the gallery to its previous state at the end unless --keep is passed.
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import faces
import face_tools

name = sys.argv[1] if len(sys.argv) > 1 else "Test"
keep = "--keep" in sys.argv

# Point the module at a SCRATCH gallery unless --keep. The earlier version backed up
# the real file, deleted it, and restored the backup at the end -- which silently threw
# away anyone enrolled through Pam in the meantime. A test should never be able to
# destroy real data, and "restore a backup" is not the same as "do not touch it".
REAL = Path(faces.GALLERY)
if not keep:
    faces.GALLERY = REAL.with_name("faces.test.json")
    faces.GALLERY.unlink(missing_ok=True)
G = Path(faces.GALLERY)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("camera busy - stop the pipeline first")


def grab(msg, secs=6):
    for n in range(secs, 0, -1):
        print(f"  {msg}  {n}...", flush=True)
        t = time.perf_counter()
        while time.perf_counter() - t < 1.0:
            cap.read()
    ok, f = cap.read()
    return f if ok else None


print(f"Enrolling '{name}'. Look straight at the camera, face filling a good part of the frame.\n")
shots = [grab("shot 1 of 3 - look straight at it"),
         grab("shot 2 of 3 - turn your head slightly"),
         grab("shot 3 of 3 - look straight again (this one is the TEST)")]
cap.release()

for i, f in enumerate(shots, 1):
    if f is not None:
        cv2.imwrite(f"facetest_{i}.jpg", f)

print("\n--- framing ---")
usable = 0
for i, f in enumerate(shots, 1):
    d = faces.faces_in(f, enrolling=True) if f is not None else []
    if not d:
        print(f"  shot {i}: NO FACE FOUND")
    else:
        verdict = d[0]["reject"] or "usable"
        usable += d[0]["reject"] is None
        print(f"  shot {i}: {d[0]['px']}px det={d[0]['det']:.2f} blur={d[0]['blur']:.0f} -> {verdict}")

if usable == 0:
    print("\n  STOP: no shot was good enough to enrol from.")
    print("  The face must be at least", faces.ENROL_MIN_FACE_PX, "px. Tilt the screen so you")
    print("  fill more of the frame, and check facetest_1.jpg to see what the camera saw.")
    sys.exit()

print("\n--- enrol + identify ---")
print(f"  (writing to {G.name})")
for i in (0, 1):
    if shots[i] is not None:
        print(f"  save_face -> {face_tools.save_face(name, shots[i])['say']}")
print(f"  who_is_this -> {face_tools.who_is_this(shots[2])['say']}")

g = faces.load()
if g.get(name):
    d3 = faces.faces_in(shots[2])
    if d3 and not d3[0]["reject"]:
        sim = max(float(v @ d3[0]["vec"]) for v in g[name])
        print(f"\n  same-person similarity: {sim:.3f}")
        print(f"  MATCH_THR {faces.MATCH_THR}, MARGIN {faces.MARGIN}")
        print(f"  -> {'comfortably above' if sim > faces.MATCH_THR + 0.15 else 'above' if sim > faces.MATCH_THR else 'BELOW - would not be recognised'}")

if not keep:
    G.unlink(missing_ok=True)
    print(f"\n  scratch gallery removed; the real one ({REAL.name}) was never touched")
    print("  pass --keep to enrol into the real gallery instead")
else:
    print(f"\n  enrolled into {REAL.name}: {[f'{k} x{len(v)}' for k, v in faces.load().items()]}")
