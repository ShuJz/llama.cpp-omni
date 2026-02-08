import os
import shutil


def clear_output_dir(path: str, expected_basename: str = "omni_output") -> None:
    try:
        if not path:
            return
        abs_path = os.path.abspath(path)
        if expected_basename and os.path.basename(abs_path) != expected_basename:
            return
        if not os.path.isdir(abs_path):
            return
        for name in os.listdir(abs_path):
            child = os.path.join(abs_path, name)
            try:
                if os.path.isdir(child) and not os.path.islink(child):
                    shutil.rmtree(child)
                else:
                    os.remove(child)
            except FileNotFoundError:
                continue
    except Exception:
        return
