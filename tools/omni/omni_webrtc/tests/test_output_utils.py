import os
import tempfile
import unittest

from omni_webrtc.output_utils import clear_output_dir


class TestClearOutputDir(unittest.TestCase):
    def test_clear_only_when_basename_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "round_000", "tts_wav"), exist_ok=True)
            with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as f:
                f.write("x")
            clear_output_dir(tmp, expected_basename="omni_output")
            self.assertTrue(os.path.exists(os.path.join(tmp, "a.txt")))
            self.assertTrue(os.path.isdir(os.path.join(tmp, "round_000")))

    def test_clear_contents(self):
        with tempfile.TemporaryDirectory() as base:
            out = os.path.join(base, "omni_output")
            os.makedirs(os.path.join(out, "round_000", "tts_wav"), exist_ok=True)
            with open(os.path.join(out, "round_000", "tts_wav", "wav_0000.wav"), "wb") as f:
                f.write(b"\x00\x01")
            with open(os.path.join(out, "root.txt"), "w", encoding="utf-8") as f:
                f.write("root")

            clear_output_dir(out)
            self.assertEqual(os.listdir(out), [])


if __name__ == "__main__":
    unittest.main()
