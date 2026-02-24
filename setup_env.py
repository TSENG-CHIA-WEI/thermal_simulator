import os
import sys
import subprocess
import shutil

def run_command(cmd, shell=False):
    process = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode

def check_python():
    print("[1/4] Checking Python installation...")
    try:
        subprocess.run([sys.executable, "--version"], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("[ERROR] Python is not installed properly.")
        sys.exit(1)

def setup_venv():
    print("\n[2/4] Setting up Python virtual environment (venv)...")
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", venv_dir])
    else:
        print("Virtual environment already exists.")

def install_dependencies():
    print("\n[3/4] Installing CPU dependencies...")
    
    # Determine the venv Python executable
    if sys.platform == "win32":
        venv_python = os.path.join("venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join("venv", "bin", "python")

    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    run_command([venv_python, "-m", "pip", "install", "-r", "requirements.txt"])

def check_and_install_cuda():
    print("\n[4/4] Checking for NVIDIA GPU and CUDA toolkit...")
    nvcc_path = shutil.which("nvcc")
    
    if sys.platform == "win32":
        venv_pip = os.path.join("venv", "Scripts", "pip.exe")
    else:
        venv_pip = os.path.join("venv", "bin", "pip")

    if nvcc_path:
        print("[INFO] CUDA Toolkit found on system.")
        result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
        output = result.stdout
        
        if "release 11." in output:
            print("[INFO] Detected CUDA 11.x. Installing cupy-cuda11x...")
            run_command([venv_pip, "install", "cupy-cuda11x"])
        elif "release 12." in output:
            print("[INFO] Detected CUDA 12.x. Installing cupy-cuda12x...")
            run_command([venv_pip, "install", "cupy-cuda12x"])
        else:
            print("[WARNING] Unrecognized CUDA version.")
            print("          GPU acceleration will not be available.")
    else:
        print("[INFO] nvcc (CUDA Toolkit) not found in PATH.")
        print("       Skipping CuPy installation. The simulator will run on CPU only.")

def main():
    print("===================================================")
    print("  Thermal Simulator (ThermoSim) Environment Setup")
    print("===================================================")
    
    check_python()
    setup_venv()
    install_dependencies()
    check_and_install_cuda()
    
    print("\n===================================================")
    print("Setup Complete!")
    print("You can now run the simulator. Be sure to activate the environment first:")
    if sys.platform == "win32":
        print("  .\\venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    print("===================================================")

if __name__ == "__main__":
    main()
