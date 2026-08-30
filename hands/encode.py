"""Encode a directory of numbered JPG frames to H.264 MP4 with PyAV (no ffmpeg binary needed)."""
import sys, glob, av, numpy as np
from PIL import Image
frames_dir, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
first = np.asarray(Image.open(files[0]).convert("RGB"))
c = av.open(out, "w"); s = c.add_stream("libx264", rate=int(round(fps)))
s.width, s.height, s.pix_fmt = first.shape[1], first.shape[0], "yuv420p"; s.options = {"crf": "23", "preset": "fast"}
for f in files:
    fr = av.VideoFrame.from_ndarray(np.asarray(Image.open(f).convert("RGB")), format="rgb24")
    for pkt in s.encode(fr): c.mux(pkt)
for pkt in s.encode(): c.mux(pkt)
c.close(); print("wrote", out, len(files), "frames")
