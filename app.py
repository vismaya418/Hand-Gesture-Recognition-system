import sys
import subprocess
from pathlib import Path

def main():
    # Resolve the absolute path to the dashboard.py file
    root_dir = Path(__file__).resolve().parent
    dashboard_path = root_dir / "src" / "dashboard.py"
    
    print("=" * 60)
    print("      LAUNCHING GESTURE RECOGNITION SYSTEM & DASHBOARD")
    print("=" * 60)
    print(f"Dashboard File: {dashboard_path}")
    print("Press Ctrl+C in this terminal window to stop the dashboard server.")
    print("=" * 60)
    
    try:
        # Run streamlit command as a subprocess
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            str(dashboard_path)
        ], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run dashboard: {e}")
        print("Please ensure you have all requirements installed via 'pip install -r requirements.txt'")

if __name__ == "__main__":
    main()
