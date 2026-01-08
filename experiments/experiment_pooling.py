import sys
import yaml
import subprocess
from pathlib import Path


def run_script(script_path, experiments_dir):
    """Run a single script and return success status."""
    print(f"\nRunning: {script_path.name}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(experiments_dir),
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.returncode == 0:
            print(f"SUCCESS: {script_path.name}")
            return True
        else:
            print(f"FAILED: {script_path.name}")
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"ERROR: {script_path.name} - {e}")
        return False


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "script_pooling.yaml"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    scripts = config.get("scripts", [])
    max_retries = config.get("max_retries", 3)
    experiments_dir = Path(__file__).parent

    # Delete status file if exists
    status_file = experiments_dir / ".experiment_status.json"
    if status_file.exists():
        status_file.unlink()

    # Run scripts
    for script_name in scripts:
        script_path = experiments_dir / script_name
        if not script_path.exists():
            print(f"Script not found: {script_name}")
            continue

        # Try with retries
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Retry {attempt}/{max_retries-1}")

            if run_script(script_path, experiments_dir):
                break

    print("\nDone")


if __name__ == "__main__":
    main()

