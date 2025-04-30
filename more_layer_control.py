# multi_layer_fft_viz_fixed4.py
# -*- coding: utf-8 -*-
import sys, time, threading
import numpy as np
from scipy.ndimage import convolve
from numpy.fft import fftn, ifftn, fftfreq, fftshift

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFrame, QMessageBox,
    QCheckBox, QGroupBox
)
import pyvista as pv
from pyvistaqt import QtInteractor

# --- Simulation Engine ---
class EmergentParticleSimulator3D:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.params = {
            'dt': 0.04,
            'damping': 0.001,
            'base_c_sq': 1.0,
            'tension_factor': 5.0,
            'potential_lin': 1.0,
            'potential_cub': 0.2,
            'biharmonic_gamma': 0.02
        }
        self.update_interval_ms = 50
        self.phi = np.zeros((grid_size,)*3, np.float64)
        self.phi_old = np.zeros_like(self.phi)
        self.lock = threading.Lock()
        self.step_count = 0

        # build 3D Laplacian kernel
        k = np.zeros((3,3,3), np.float64)
        k[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            k[dx,dy,dz] = 1
        self.kern = k

        self.initialize_field()

    def initialize_field(self):
        with self.lock:
            N = self.grid_size
            x = np.arange(N); y = np.arange(N); z = np.arange(N)
            X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
            cx = cy = cz = N//2
            r2 = (N/8.0)**2
            self.phi[:] = 2.0 * np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)/(2*r2))
            self.phi_old[:] = self.phi
            self.step_count = 0

    def step(self):
        with self.lock:
            lap   = convolve(self.phi,   self.kern, mode='wrap')
            bih   = convolve(lap,        self.kern, mode='wrap')

            # **Corrected** potential derivative using GUI parameters:
            Vp    = -self.params['potential_lin'] * self.phi \
                    + self.params['potential_cub'] * (self.phi**3)

            c2    = self.params['base_c_sq'] / (
                      1.0 + self.params['tension_factor']*(self.phi**2) + 1e-9
                    )
            accel = c2*lap - Vp - self.params['biharmonic_gamma']*bih
            vel   = self.phi - self.phi_old
            phi_new = self.phi + (1.0 - self.params['damping']*self.params['dt'])*vel \
                      + (self.params['dt']**2)*accel

            self.phi_old[:] = self.phi
            self.phi[:]     = phi_new
            self.step_count += 1

    def get_phi_copy(self):
        with self.lock:
            return self.phi.copy()

    def update_params(self, d):
        with self.lock:
            for k,v in d.items():
                if k in self.params:
                    self.params[k] = float(v)

    def reset_simulation(self):
        self.initialize_field()


# --- Simulation Thread ---
class SimulationWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            t0 = time.perf_counter()
            self.sim.step()
            dt = (time.perf_counter()-t0)*1000
            to_sleep = max(1, self.sim.update_interval_ms - dt)/1000.0
            time.sleep(to_sleep)
        self.finished.emit()

    def stop(self):
        self._running = False


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("Interactive 3D Sim – Harmonic Components")
        self.resize(1300,800)

        # Layout
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)

        # PyVista canvas
        pf = QFrame()
        pf.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
        pv_layout = QVBoxLayout(pf)
        self.plotter = QtInteractor(pf)
        pv_layout.addWidget(self.plotter)
        main_layout.addWidget(pf,stretch=3)

        # Controls
        ctrl = QWidget(); ctrl.setFixedWidth(350)
        cl = QVBoxLayout(ctrl)
        main_layout.addWidget(ctrl,stretch=1)

        # Start/Stop/Reset
        hl = QHBoxLayout()
        self.start_b = QPushButton("Start")
        self.stop_b  = QPushButton("Stop"); self.stop_b.setEnabled(False)
        self.reset_b = QPushButton("Reset")
        hl.addWidget(self.start_b); hl.addWidget(self.stop_b); hl.addWidget(self.reset_b)
        cl.addLayout(hl)
        self.start_b.clicked.connect(self.start)
        self.stop_b.clicked.connect(self.stop)
        self.reset_b.clicked.connect(self.reset)

        # Parameter sliders
        cl.addWidget(QLabel("Parameters:"))
        self.sliders = {}
        for name in self.sim.params:
            h = QHBoxLayout()
            lbl = QLabel(name)
            sld = QSlider(Qt.Horizontal)
            mn,mx,sc,pr = self._slider_conf(name)
            sld.setRange(int(mn*sc),int(mx*sc))
            sld.setValue(int(self.sim.params[name]*sc))
            val_lbl = QLabel(f"{self.sim.params[name]:.{pr}f}")
            sld.valueChanged.connect(lambda v,n=name,sc=sc,pr=pr: self._param_changed(n,v/sc,pr))
            h.addWidget(lbl); h.addWidget(sld); h.addWidget(val_lbl)
            cl.addLayout(h)
            self.sliders[name] = {'s':sld,'l':val_lbl,'sc':sc,'pr':pr}

        # Band controls
        grp = QGroupBox("Bands")
        gl = QVBoxLayout(grp)
        self.layer_vis = {}
        self.layer_iso = {}
        self.colors    = ['blue','red','green','purple']
        for i in range(4):
            h = QHBoxLayout()
            cb = QCheckBox(f"Band {i}"); cb.setChecked(True)
            iso_s = QSlider(Qt.Horizontal); iso_s.setRange(-300,300); iso_s.setValue(100)
            iso_l = QLabel("1.00")
            iso_s.valueChanged.connect(lambda v,i=i: self._iso_changed(i,v/100))
            h.addWidget(cb); h.addWidget(QLabel("Iso:")); h.addWidget(iso_s); h.addWidget(iso_l)
            gl.addLayout(h)
            self.layer_vis[i] = cb
            self.layer_iso[i] = {'s':iso_s,'l':iso_l,'v':1.0}
        cl.addWidget(grp)
        cl.addStretch()

        # FFT prep & scene
        self._init_fft()
        self._scene_setup()

        # Timer
        self.timer = QTimer(self); 
        self.timer.timeout.connect(self._refresh); 
        self.timer.setInterval(100)
        self.worker = None

    def _slider_conf(self,n):
        if n=='dt': return 0.01,0.2,1000,3
        if n=='damping': return 0,0.01,10000,4
        if n=='base_c_sq': return 0.1,5,100,2
        if n=='tension_factor': return 0,20,10,1
        if n=='potential_lin': return 0.1,2,100,2
        if n=='potential_cub': return 0,1,100,2
        if n=='biharmonic_gamma': return 0,0.1,10000,4
        return 0,1,100,2

    def _param_changed(self,n,v,pr):
        self.sim.update_params({n:v})
        self.sliders[n]['l'].setText(f"{v:.{pr}f}")

    def _iso_changed(self,i,v):
        self.layer_iso[i]['v'] = v
        self.layer_iso[i]['l'].setText(f"{v:.2f}")
        if not (self.worker and self.worker._running):
            self._draw_once()

    def _init_fft(self):
        N = self.sim.grid_size
        f = fftshift(fftfreq(N))
        self.kx,self.ky,self.kz = np.meshgrid(f,f,f,indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        km = 0.5
        self.bands = [(0,km/8),(km/8,km/4),(km/4,km/2),(km/2,km)]
        print("FFT bands:", self.bands)

    def _scene_setup(self):
        self.plotter.add_axes()
        phi0 = self.sim.get_phi_copy()
        g = pv.ImageData(dimensions=np.array(phi0.shape)+1)
        self.plotter.add_mesh(g.outline(),color='grey',name='outline')
        self.plotter.camera_position='xy'; self.plotter.reset_camera()

    def _filter(self,fftphi,i):
        kmin,kmax = self.bands[i]
        mask = (self.k2>=kmin)&(self.k2<kmax)
        f2 = fftphi.copy(); f2[~mask] = 0
        return f2

    def _draw_once(self):
        phi = self.sim.get_phi_copy()
        fftphi = fftshift(fftn(phi))

        # rebuild grid
        dim = np.array(phi.shape)+1
        grid = pv.ImageData(dimensions=dim)
        grid.cell_data['phi'] = phi.ravel(order='F')

        # remove old band actors
        for i in range(4):
            nm = f'band{i}'
            if nm in self.plotter.actors:
                self.plotter.remove_actor(nm,render=False)

        # add new ones
        for i in range(4):
            if not self.layer_vis[i].isChecked():
                continue
            comp = np.real(ifftn(fftshift(self._filter(fftphi,i))))
            grid.cell_data[f'band{i}'] = comp.ravel(order='F')
            pdata = grid.cell_data_to_point_data()
            mesh = pdata.contour([self.layer_iso[i]['v']], scalars=f'band{i}')
            if mesh.n_points>0 and mesh.n_faces>0:
                self.plotter.add_mesh(mesh, name=f'band{i}', color=self.colors[i], opacity=0.4)
        self.plotter.render()

    def _refresh(self):
        if self.worker and self.worker._running:
            self._draw_once()

    def start(self):
        if not (self.worker and self.worker._running):
            self.worker = SimulationWorker(self.sim)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.thread.start()
            self.timer.start()
            self.start_b.setEnabled(False)
            self.stop_b.setEnabled(True)

    def stop(self):
        if self.worker: self.worker.stop()
        self.timer.stop()
        self.start_b.setEnabled(True)
        self.stop_b.setEnabled(False)

    def reset(self):
        self.stop()
        self.sim.reset_simulation()
        self._draw_once()

    def closeEvent(self, ev):
        self.stop()
        super().closeEvent(ev)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = EmergentParticleSimulator3D(grid_size=64)
    w = MainWindow(sim)
    w.show()
    sys.exit(app.exec_())
