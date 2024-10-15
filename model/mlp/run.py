import multiprocessing
import subprocess

# List of your Python scripts
scripts = ["mlp_detailed.py", "mlp_systematic.py"]

# Function to run each script
def run_script(script):
    subprocess.run(["python", script])

if __name__ == "__main__":
    processes = []
    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script,))
        p.start()  # Start each process
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish