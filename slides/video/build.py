"""Build pitch_video.mp4: one narrated clip per slide (slideN.png + line N of script.txt)."""
import asyncio, pathlib, subprocess, sys
import edge_tts

VOICE, RATE, PAD = "en-US-AndrewNeural", "+5%", 0.6
here = pathlib.Path(__file__).parent
lines = [l.strip() for l in (here / "script.txt").read_text().splitlines() if l.strip()]

def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))

async def tts():
    for i, text in enumerate(lines, 1):
        for attempt in range(4):
            try:
                await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(here / f"n{i}.mp3"))
                break
            except edge_tts.exceptions.NoAudioReceived:
                if attempt == 3:
                    raise
                await asyncio.sleep(2)

asyncio.run(tts())
clips = []
for i in range(1, len(lines) + 1):
    clip = here / f"clip{i}.mp4"
    t = round(dur(here / f"n{i}.mp3") + PAD, 2)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30", "-i", str(here / f"slide{i}.png"),
                    "-i", str(here / f"n{i}.mp3"), "-af", "apad", "-t", str(t), "-c:v", "libx264", "-tune", "stillimage",
                    "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(clip)], check=True)
    clips.append(clip)
    print(f"slide {i}: {t:.1f}s")
(here / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
out = here / "pitch_video.mp4"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(here / "list.txt"), "-c", "copy", str(out)], check=True)
for c in clips: c.unlink()
(here / "list.txt").unlink()
total = dur(out)
print(f"total {total:.1f}s")
sys.exit(0 if total < 120 else 1)
