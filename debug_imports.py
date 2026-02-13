
print("Step 1: Importing sys/os")
import sys, os
print("Step 2: Importing PyQt5.QtWidgets")
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    print("  Success")
    print("Step 2b: Instantiating QApplication")
    app = QApplication(sys.argv)
    print("  App Created")
    win = QMainWindow()
    print("  Window Created")
except Exception as e:
    print(f"  Failed: {e}")

print("Step 3: Importing PyVista")
try:
    import pyvista as pv
    print("  Success")
except Exception as e:
    print(f"  Failed: {e}")

print("Step 4: Importing QtInteractor")
try:
    from pyvistaqt import QtInteractor
    print("  Success")
except Exception as e:
    print(f"  Failed: {e}")

print("Done.")
