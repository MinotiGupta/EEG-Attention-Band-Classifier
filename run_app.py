#!/usr/bin/env python3
"""
Script to run the Real-time EEG Attention Analyzer
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit application"""

    print("🧠 Starting Real-time EEG Attention Analyzer...")
    print("=" * 50)

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    main_app = script_dir / "main.py"

    # Check if main.py exists
    if not main_app.exists():
        print(f"❌ Error: {main_app} not found!")
        sys.exit(1)

    print(f"📁 App location: {main_app}")
    print("🚀 Launching Streamlit dashboard...")
    print("\n💡 The dashboard will open in your web browser")
    print("🔗 Default URL: http://localhost:8501")
    print("\n⚡ Features:")
    print("  - Real-time EEG attention monitoring")
    print("  - Live frequency band analysis")
    print("  - Attention state detection")
    print("  - Interactive visualizations")
    print("\n" + "=" * 50)

    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(main_app),
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure Streamlit is installed: pip install streamlit")
        print("  2. Check if all dependencies are installed: pip install -r requirements.txt")
        print("  3. Verify Python version compatibility")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
        print("👋 Thanks for using the EEG Attention Analyzer!")
        sys.exit(0)


if __name__ == "__main__":
    main()
