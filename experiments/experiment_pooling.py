import sys
import yaml
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.discord_utils import send_discord_notification


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
            # Send success notification
            message = f"✅ **Script Completed**\n📄 `{script_path.name}`"
            send_discord_notification(message, color=0x00FF00)
            return True
        else:
            print(f"FAILED: {script_path.name}")
            if result.stderr:
                print(result.stderr)
            # Send failure notification
            message = f"❌ **Script Failed**\n📄 `{script_path.name}`"
            if result.stderr:
                # Limit error message to first 500 chars
                error_msg = result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr
                message += f"\n```\n{error_msg}\n```"
            send_discord_notification(message, color=0xFF0000)
            return False
    except Exception as e:
        print(f"ERROR: {script_path.name} - {e}")
        # Send error notification
        message = f"⚠️ **Script Error**\n📄 `{script_path.name}`\n```\n{str(e)}\n```"
        send_discord_notification(message, color=0xFFA500)
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

    # Track results
    completed_scripts = []
    failed_scripts = []
    total_scripts = len(scripts)

    # Run scripts
    for script_name in scripts:
        script_path = experiments_dir / script_name
        if not script_path.exists():
            print(f"Script not found: {script_name}")
            failed_scripts.append(script_name)
            continue

        # Try with retries
        success = False
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Retry {attempt}/{max_retries-1}")
                message = f"🔄 **Retrying Script**\n📄 `{script_name}`\n🔄 Attempt: {attempt + 1}/{max_retries}"
                send_discord_notification(message, color=0xFFA500)

            if run_script(script_path, experiments_dir):
                success = True
                break

        if success:
            completed_scripts.append(script_name)
        else:
            failed_scripts.append(script_name)

    # Send batch summary
    failed_list = "\n".join([f"  • `{script}`" for script in failed_scripts]) if failed_scripts else "  None"
    message = (
        f"📊 **Batch Execution Summary**\n"
        f"✅ Completed: {len(completed_scripts)}/{total_scripts}\n"
        f"❌ Failed: {len(failed_scripts)}/{total_scripts}\n"
        f"\n**Failed Scripts:**\n{failed_list}"
    )
    color = 0x00FF00 if len(failed_scripts) == 0 else 0xFF0000
    send_discord_notification(message, color=color)

    print("\nDone")


if __name__ == "__main__":
    main()

