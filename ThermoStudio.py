
import sys
import os
import pyvista as pv
import numpy as np
import configparser
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QPlainTextEdit, QPushButton, QLabel, QTabWidget, 
                             QMessageBox, QFrame, QGroupBox, QDoubleSpinBox, QCheckBox, 
                             QSlider, QRadioButton, QButtonGroup, QFormLayout, QDockWidget,
                             QGridLayout, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import QProcess, Qt, QSize
from pyvistaqt import QtInteractor

class ThermoStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThermoStudio v3.0 (Classic Stable)")
        self.resize(1600, 1000)
        
        # State
        self.mesh = None
        self.project_dir = "projects/demo" # Default
        
        # --- Main Layout ---
        self.vis_frame = QFrame()
        self.setCentralWidget(self.vis_frame)
        v_layout = QVBoxLayout(self.vis_frame)
        v_layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista Plotter (White Background)
        self.plotter = QtInteractor(self.vis_frame)
        v_layout.addWidget(self.plotter.interactor)
        self.init_plotter()
        
        # --- Control Panel (Sidebar) ---
        self.dock_ctrl = QDockWidget("Control Panel", self)
        self.dock_ctrl.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_ctrl)
        
        ctrl_widget = QWidget()
        self.dock_ctrl.setWidget(ctrl_widget)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        
        # 1. Project Info
        grp_proj = QGroupBox("Project")
        l_proj = QVBoxLayout(grp_proj)
        self.lbl_proj = QLabel(f"Project: ...")
        self.btn_load = QPushButton("Load Project")
        self.btn_load.clicked.connect(self.load_project_dialog)
        
        l_proj.addWidget(self.lbl_proj)
        l_proj.addWidget(self.btn_load)
        ctrl_layout.addWidget(grp_proj)
        
        # 2. Visualization
        grp_viz = QGroupBox("Visualization")
        l_viz = QFormLayout(grp_viz)
        
        self.spin_min = QDoubleSpinBox(); self.spin_min.setRange(-500, 5000); self.spin_min.setValue(25.0)
        self.spin_max = QDoubleSpinBox(); self.spin_max.setRange(-500, 5000); self.spin_max.setValue(100.0)
        self.spin_min.valueChanged.connect(self.update_scene)
        self.spin_max.valueChanged.connect(self.update_scene)
        
        l_viz.addRow("Min Temp:", self.spin_min)
        l_viz.addRow("Max Temp:", self.spin_max)
        
        # Toggles
        self.chk_edges = QCheckBox("Show Mesh Edges"); self.chk_edges.toggled.connect(self.update_scene)
        l_viz.addRow("", self.chk_edges)
        
        self.chk_env = QCheckBox("Show Env Box"); self.chk_env.setChecked(True); self.chk_env.toggled.connect(self.update_scene)
        l_viz.addRow("", self.chk_env)
        
        self.chk_bounds = QCheckBox("Show Component Bounds"); self.chk_bounds.setChecked(False); self.chk_bounds.toggled.connect(self.update_scene)
        l_viz.addRow("", self.chk_bounds)

        ctrl_layout.addWidget(grp_viz)

        # 2b. Component Filter
        grp_comp = QGroupBox("Components")
        l_comp = QVBoxLayout(grp_comp)
        self.list_comps = QListWidget()
        self.list_comps.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_comps.itemSelectionChanged.connect(self.update_scene)
        l_comp.addWidget(self.list_comps)
        ctrl_layout.addWidget(grp_comp)
        
        # 2.5 Camera Views
        grp_cam = QGroupBox("Camera Views")
        l_cam = QGridLayout(grp_cam)
        btn_xp = QPushButton("+X"); btn_xp.clicked.connect(lambda: self.plotter.view_yz())
        btn_xn = QPushButton("-X"); btn_xn.clicked.connect(lambda: self.plotter.view_yz(negative=True))
        btn_yp = QPushButton("+Y"); btn_yp.clicked.connect(lambda: self.plotter.view_xz())
        btn_yn = QPushButton("-Y"); btn_yn.clicked.connect(lambda: self.plotter.view_xz(negative=True))
        btn_zp = QPushButton("+Z"); btn_zp.clicked.connect(lambda: self.plotter.view_xy())
        btn_zn = QPushButton("-Z"); btn_zn.clicked.connect(lambda: self.plotter.view_xy(negative=True))
        btn_iso = QPushButton("Iso"); btn_iso.clicked.connect(lambda: self.plotter.view_isometric())
        
        l_cam.addWidget(btn_xp, 0, 0); l_cam.addWidget(btn_xn, 1, 0)
        l_cam.addWidget(btn_yp, 0, 1); l_cam.addWidget(btn_yn, 1, 1)
        l_cam.addWidget(btn_zp, 0, 2); l_cam.addWidget(btn_zn, 1, 2)
        l_cam.addWidget(btn_iso, 0, 3, 2, 1)
        
        ctrl_layout.addWidget(grp_cam)
        
        # 3. Slicing
        grp_slice = QGroupBox("Slice / Clip")
        l_slice = QVBoxLayout(grp_slice)
        
        # Mode
        h_mode = QHBoxLayout()
        self.bg_slice = QButtonGroup()
        self.rad_off = QRadioButton("Full"); self.rad_off.setChecked(True)
        self.rad_x = QRadioButton("X")
        self.rad_y = QRadioButton("Y")
        self.rad_z = QRadioButton("Z")
        for r in [self.rad_off, self.rad_x, self.rad_y, self.rad_z]:
            h_mode.addWidget(r); self.bg_slice.addButton(r); r.toggled.connect(self.update_slice_mode)
        l_slice.addLayout(h_mode)
        
        # Slider
        self.lbl_pos = QLabel("Pos: 0.00 mm")
        l_slice.addWidget(self.lbl_pos)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_slice_label)
        self.slider.sliderReleased.connect(self.update_scene)
        l_slice.addWidget(self.slider)
        
        # Clip
        self.chk_clip = QCheckBox("Clip Volume")
        self.chk_clip.setChecked(False) # Stable default
        self.chk_clip.toggled.connect(self.update_scene)
        l_slice.addWidget(self.chk_clip)
        
        ctrl_layout.addWidget(grp_slice)
        
        # 4. Simulation
        grp_sim = QGroupBox("Simulation")
        l_sim = QVBoxLayout(grp_sim)
        self.btn_run = QPushButton("Run Solver")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_run.clicked.connect(self.run_simulation)
        l_sim.addWidget(self.btn_run)
        
        self.tabs = QTabWidget()
        self.txt_log = QPlainTextEdit(); self.txt_log.setReadOnly(True)
        self.tabs.addTab(self.txt_log, "Log")
        l_sim.addWidget(self.tabs)
        
        ctrl_layout.addWidget(grp_sim)
        ctrl_layout.addStretch()

        # Init
        self.detect_default_project()
        self.init_app()

    def init_plotter(self):
        # Classic White
        self.plotter.set_background('white')
        self.plotter.add_axes()
        # No EDL by default (Safe Mode)
        
    def detect_default_project(self):
        # Default to the new verification stack for the user
        target = "projects/chip_stack"
        if os.path.exists(target):
            self.project_dir = target
        elif os.path.exists("projects/demo"):
             self.project_dir = "projects/demo"
        self.refresh_ui_label()

    def refresh_ui_label(self):
        self.lbl_proj.setText(f"Project: {os.path.basename(self.project_dir)}")

    def load_project_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Select Project Directory", self.project_dir)
        if d:
            self.project_dir = d
            self.refresh_ui_label()
            self.init_app() # Reloads VTK and scene
            # Verify if config exists
            self.draw_component_bounds() # Reload bounds

    def load_components_map(self):
        self.list_comps.clear()
        
        # Search for ANY .config file in the project dir
        try:
             files = [f for f in os.listdir(self.project_dir) if f.endswith(".config")]
        except:
             return
             
        if not files: return
        
        # Priority
        if "box_sim.config" in files:
            cfg_name = "box_sim.config"
        else:
            cfg_name = files[0]
            
        cfg_path = os.path.join(self.project_dir, cfg_name)
        
        config = configparser.ConfigParser()
        config.read(cfg_path)
        
        idx = 0
        for sec in config.sections():
            if sec.lower().startswith("box:"):
                name = sec.split(":")[1]
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, idx)
                
                # Checkbox for visibility? No, just selection.
                self.list_comps.addItem(item)
                item.setSelected(True)
                idx += 1
                
    def init_app(self):
        # Clear Scene
        self.plotter.clear()
        self.plotter.add_axes()
        
        # Load components list first
        self.load_components_map()
        
        # Load output.vtk if exists
        vtk_path = os.path.join(self.project_dir, "output.vtk")
        if os.path.exists(vtk_path):
            self.load_result(vtk_path)
        else:
            self.txt_log.appendPlainText("No output.vtk found in this project.")
            self.plotter.reset_camera()
            
        self.draw_component_bounds()

    def load_result(self, filename):
        try:
            self.mesh = pv.read(filename)
            t_min, t_max = self.mesh['Temperature'].min(), self.mesh['Temperature'].max()
            
            self.spin_min.setValue(t_min)
            self.spin_max.setValue(t_max)
            
            self.update_slice_mode()
            self.update_scene()
            self.plotter.view_isometric()
            self.statusBar().showMessage(f"Loaded {filename}")
            
        except Exception as e:
            self.txt_log.appendPlainText(f"Error loading mesh: {e}")

    def run_simulation(self):
        self.txt_log.clear()
        self.txt_log.appendPlainText("Running...")
        self.btn_run.setEnabled(False)
        
        self.process = QProcess()
        self.process.setWorkingDirectory(self.project_dir)
        
        # Absolute path fix
        root_dir = os.path.dirname(os.path.abspath(__file__))
        solver_script = os.path.join(root_dir, "ThermoSim.py")
        
        abs_proj = os.path.abspath(self.project_dir)
        
        # Detect Params Config
        params_file = "params_sample.config"
        if os.path.exists(os.path.join(abs_proj, "params_stack.config")):
             params_file = "params_stack.config"
             
        cmd_args = [
            os.path.join(abs_proj, "box_sim.config"),
            os.path.join(abs_proj, params_file)
        ]
        
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.start(sys.executable, [solver_script] + cmd_args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        self.txt_log.insertPlainText(bytes(data).decode('utf8'))
        self.txt_log.moveCursor(self.txt_log.textCursor().End)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        self.txt_log.insertPlainText("ERR: " + bytes(data).decode('utf8'))
        self.txt_log.moveCursor(self.txt_log.textCursor().End)

    def on_process_finished(self):
        self.btn_run.setEnabled(True)
        if self.process.exitCode() == 0:
            self.txt_log.appendPlainText("Done.")
            self.load_result(os.path.join(self.project_dir, "output.vtk"))
        else:
            self.txt_log.appendPlainText("Failed.")

    def update_slice_mode(self):
        if self.mesh is None: return
        bounds = self.mesh.bounds
        factor = 1e6
        
        idx = -1
        if self.rad_x.isChecked(): idx = 0
        if self.rad_y.isChecked(): idx = 2
        if self.rad_z.isChecked(): idx = 4
        
        if idx != -1:
            vmin, vmax = bounds[idx], bounds[idx+1]
            self.slider.setRange(int(vmin*factor), int(vmax*factor))
            self.slider.setValue(int((vmin+vmax)/2 * factor))
            self.slider.setEnabled(True)
        else:
            self.slider.setEnabled(False)
        self.update_slice_label()
        self.update_scene()

    def update_slice_label(self):
        val = self.slider.value() / 1000.0
        self.lbl_pos.setText(f"Pos: {val:.3f} mm")

    def update_scene(self):
        if self.mesh is None: return
        self.plotter.clear()
        self.plotter.add_axes()
        
        vmin, vmax = self.spin_min.value(), self.spin_max.value()
        
        mesh_to_plot = self.mesh
             
        # Filter by Component Selection
        if self.mesh and "BoxID" in self.mesh.cell_data:
            sel_items = self.list_comps.selectedItems()
            # If subset selected (and not all)
            if sel_items and len(sel_items) < self.list_comps.count():
                sel_ids = [item.data(Qt.UserRole) for item in sel_items]
                # mesh.cell_data['BoxID'] is a pyvista array
                # Use numpy for speed
                box_ids = self.mesh.cell_data['BoxID']
                mask = np.isin(box_ids, sel_ids)
                if np.any(mask):
                    mesh_to_plot = self.mesh.extract_cells(mask)
                else:
                    # User selected something but no cells match
                    pass
        elif self.mesh and sel_items:
             # User selected items, but 'BoxID' is missing in mesh
             pass # Fail silently or Print?
             # Let's not spam popups in update_scene (called frequently)
             # But if it's the first time...
             pass
        if not self.rad_off.isChecked():
            val = self.slider.value() / 1e6
            normal = 'z'; origin = (0,0,val)
            if self.rad_x.isChecked(): normal = 'x'; origin = (val,0,0)
            if self.rad_y.isChecked(): normal = 'y'; origin = (0,val,0)
            if self.rad_z.isChecked(): normal = 'z'; origin = (0,0,val)
            
            if self.chk_clip.isChecked():
                mesh_to_plot = self.mesh.clip(normal=normal, origin=origin, invert=True)
            else:
                mesh_to_plot = self.mesh.slice(normal=normal, origin=origin)
            
            # Add Environment Box if checked
            if self.chk_env.isChecked():
                 self.plotter.add_mesh(self.mesh.outline(), color='black')
        else:
            # Full View
            if self.chk_env.isChecked():
                 self.plotter.add_mesh(self.mesh.outline(), color='black')

        if mesh_to_plot.n_points > 0:
            self.plotter.add_mesh(mesh_to_plot, scalars='Temperature', cmap='jet', clim=[vmin, vmax],
                                  show_edges=self.chk_edges.isChecked(), edge_color='black',
                                  scalar_bar_args={'title': 'Temp (C)', 'color': 'black'})
        
        # Component Boundaries
        if self.chk_bounds.isChecked():
             self.draw_component_bounds()
        
        self.plotter.reset_camera_clipping_range()

    def draw_component_bounds(self):
        # We need to parse box_sim.config from project dir
        try:
             import configparser
             config = configparser.ConfigParser()
             
             # Dynamic Config Finding
             try:
                 files = [f for f in os.listdir(self.project_dir) if f.endswith(".config")]
             except:
                 return

             if not files: return
            
             if "box_sim.config" in files:
                cfg_name = "box_sim.config"
             else:
                cfg_name = files[0]
                
             cfg_path = os.path.join(self.project_dir, cfg_name)
             
             config.read(cfg_path)
             
             # Need logic to parse box limits. This is redundant with config_parser.py
             # Simple parser here for visualization
             for sec in config.sections():
                 if sec.lower().startswith("box"):
                     try:
                         origin = [float(x.strip()) for x in config[sec]['Origin'].split(',')]
                         size = [float(x.strip()) for x in config[sec]['Size'].split(',')]
                         
                         # Create box geometry
                         box = pv.Box(bounds=(origin[0], origin[0]+size[0], 
                                              origin[1], origin[1]+size[1], 
                                              origin[2], origin[2]+size[2]))
                         self.plotter.add_mesh(box, style='wireframe', color='blue', line_width=2)
                     except:
                         pass
        except Exception as e:
             print(f"Failed to draw bounds: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ThermoStudio()
    w.show()
    sys.exit(app.exec_())
