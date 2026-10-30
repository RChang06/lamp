import sys, time
sys.path.insert(0, ".")
from brain.vision import Vision

v = Vision()
v.start()
print("look at the camera, then look away. 20 seconds.")
last = None
t0 = time.time()
while time.time() - t0 < 20:
    lk = v.look
    tag = (lk.present, lk.attending)
    if tag != last:
        print(f"{time.time()-t0:5.1f}s present={lk.present} attending={lk.attending} "
              f"az={lk.az:+.2f} el={lk.el:+.2f} yaw={lk.yaw:+5.1f} size={lk.size:.2f}")
        last = tag
    time.sleep(0.05)
lk = v.look
print(f"\nframes {v.frames} in 20s = {v.frames/20:.1f} hz, landmarker {lk.latency_ms:.1f} ms/frame")
v.close()
