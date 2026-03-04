import time
import numpy as np
import sounddevice as sd

DEVICE = 3          # 你的 device id
REQ_SR = 16000      # 先尝试 16000
CHANNELS = 1
BLOCK_MS = 50       # 每 50ms 一块

def rms_raw(x):
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))

def try_open(device, sr):
    block_frames = int(sr * BLOCK_MS / 1000)
    print(f"\n[TRY] Open InputStream device={device} sr={sr} ch={CHANNELS} block={BLOCK_MS}ms ({block_frames} frames)")
    stream = sd.InputStream(
        samplerate=sr,
        channels=CHANNELS,
        dtype="int16",
        device=device,
        blocksize=block_frames,
    )
    stream.start()
    return stream, block_frames

print("==== PortAudio / sounddevice quick probe ====")
default_in, default_out = sd.default.device
print(f"Default input id:  {default_in}")
print(f"Default output id: {default_out}")

dev = sd.query_devices(DEVICE)
print("\n==== Device info ====")
print(f"DEVICE={DEVICE}")
print(f"name: {dev.get('name')}")
print(f"hostapi: {dev.get('hostapi')}")
print(f"max_input_channels: {dev.get('max_input_channels')}")
print(f"max_output_channels: {dev.get('max_output_channels')}")
print(f"default_samplerate: {dev.get('default_samplerate')}")
print("=============================================")

if dev.get("max_input_channels", 0) < 1:
    raise SystemExit(
        f"Device {DEVICE} has no input channels (max_input_channels=0). "
        f"Use a device id that has input. Run a device list first."
    )

# Try requested SR, then fallback to default SR
sr_candidates = [REQ_SR]
default_sr = int(dev.get("default_samplerate") or 0)
if default_sr and default_sr not in sr_candidates:
    sr_candidates.append(default_sr)

stream = None
block_frames = None
last_err = None

for sr in sr_candidates:
    try:
        stream, block_frames = try_open(DEVICE, sr)
        print("[OK] Stream opened.")
        break
    except Exception as e:
        last_err = e
        print(f"[FAIL] Could not open with sr={sr}. Error: {type(e).__name__}: {e}")

if stream is None:
    raise SystemExit(
        f"\nCould not open input stream for device={DEVICE} with SR candidates {sr_candidates}.\n"
        f"Last error: {type(last_err).__name__}: {last_err}\n\n"
        f"Likely causes:\n"
        f"  - The device doesn't support the requested sample rates (try its default_samplerate).\n"
        f"  - Another app is using the device in exclusive mode (WASAPI exclusive).\n"
        f"  - Windows privacy permission blocks mic access for desktop apps.\n"
        f"  - Driver issue.\n"
    )

print("\nSpeak now... (Ctrl+C to stop)")
try:
    while True:
        data, _ = stream.read(block_frames)
        r = rms_raw(data)
        maxv = int(data.max())
        minv = int(data.min())
        print(f"RMS={r:8.2f}  max={maxv:6d}  min={minv:6d}")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass