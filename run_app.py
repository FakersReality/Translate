# run_app.py
from pathlib import Path
from streamlit.web.bootstrap import run
import sys

# When bundled by PyInstaller, files are unpacked to sys._MEIPASS
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
app_script = BASE / "MainLang.py"

# Launch Streamlit
run(str(app_script), "", [], {})
