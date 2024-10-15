import multiprocessing
import subprocess

# List of your Python scripts
scripts = ["gru_detailed.py", "lstm_detailed.py", "gru_systematic.py", "lstm_systematic.py"]

# Function to run each script
def run_script(script):
    try:
        # Run the script and capture output and errors
        result = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()

        # Print output and errors if any
        print(f"Output of {script}:\n{output.decode()}")
        if error:
            print(f"Error in {script}:\n{error.decode()}")
    except Exception as e:
        print(f"Exception occurred while running {script}: {e}")

if __name__ == "__main__":
    processes = []
    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script,))
        p.start()  # Start each process
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish