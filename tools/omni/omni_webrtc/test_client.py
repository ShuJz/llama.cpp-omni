import argparse
import asyncio
import logging
import os
import wave
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-client")


async def run(audio_file):
    if not os.path.exists(audio_file):
        # Generate dummy audio
        logger.info(f"Generating dummy audio: {audio_file}")
        import numpy as np
        sample_rate = 16000
        duration = 3 # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Convert to int16
        audio = (audio * 32767).astype(np.int16)

        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

    pc = RTCPeerConnection()

    # Send audio
    player = MediaPlayer(audio_file)
    pc.addTrack(player.audio)

    # Record response
    recorder = MediaRecorder("response.wav")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Receiving track: {track.kind}")
        recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            logger.info("Track ended")
            await recorder.stop()

    # Create Offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send Offer
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8081/offer", json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as resp:
            if resp.status != 200:
                logger.error(f"Error: {await resp.text()}")
                return
            answer = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=answer["sdp"],
        type=answer["type"]
    ))

    logger.info("Connected. Sending audio...")
    await recorder.start()

    # Wait for audio to finish playing + some time for response
    # In a real app, we'd wait for VAD or user stop.
    # Here, MediaPlayer stops when file ends.
    # We need to keep the connection open to receive response.

    # Wait until recorder stops or timeout
    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        pass

    await pc.close()
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test_input.wav", help="Audio file to send")
    args = parser.parse_args()

    asyncio.run(run(args.audio))
