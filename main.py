import subprocess

from backend import singleton

if __name__ == "__main__":

    from backend.singleton_params import instantiate_singleton_params

    instantiate_singleton_params()

    cmd = ["streamlit",
            "run",
            "frontend/ui.py",
            "--browser.gatherUsageStats",
            "false"]

    process = subprocess.Popen(cmd)
    with process as p:
        try:
            for line in p.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            print("Killing streamlit app")
            p.kill()
            p.wait()
            raise
