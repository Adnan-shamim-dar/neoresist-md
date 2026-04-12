from pathlib import Path
import runpy


TARGET_APP = Path(__file__).resolve().parent / "neoresist-md" / "app.py"

if not TARGET_APP.exists():
    raise FileNotFoundError(f"Updated app entrypoint not found: {TARGET_APP}")

# Keep root launch command stable while always executing the updated app.
runpy.run_path(str(TARGET_APP), run_name="__main__")
