"""Copy real pump clips into TACIT's Ear module.

Usage:  python3 src/import_audio.py <folder with .wav files>

Files whose name contains "anomaly" (or "abnormal") go to assets/audio/anomaly, files containing "normal"
go to assets/audio/normal (DCASE 2020 names: normal_id_00_00000000.wav / anomaly_id_00_00000000.wav;
MIMII: put the files from id_00/normal and id_00/abnormal in one folder, or run this once per folder).
Restart the server afterwards; the Ear page then says "Real pump recordings".
"""
import os, sys, glob, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = sys.argv[1] if len(sys.argv) > 1 else None
if not src or not os.path.isdir(src): sys.exit("Give the folder that contains the .wav files, e.g. python3 src/import_audio.py ~/Downloads/pump_sample")
counts = {"normal": 0, "anomaly": 0}
for f in sorted(glob.glob(os.path.join(src, "**", "*.wav"), recursive=True)):
    low = f.lower(); kind = "anomaly" if ("anomaly" in low or "abnormal" in low) else ("normal" if "normal" in low else None)
    if not kind: continue
    d = os.path.join(ROOT, "assets", "audio", kind); os.makedirs(d, exist_ok=True)
    shutil.copy(f, os.path.join(d, os.path.basename(f))); counts[kind] += 1
print(f"Copied {counts['normal']} normal and {counts['anomaly']} faulty clips into assets/audio/. Restart the server.")
