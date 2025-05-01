import threading, time, queue
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import marching_cubes
import colorsys

from ursina import (
    Ursina, window, color, Entity, Mesh, EditorCamera, application,
    camera, Slider, Text
)
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight, AmbientLight

# ── mini solver ───────────────────────────────────────────
class MiniWoW:
    def __init__(self, N=32, dt=0.04, damping=0.001,
                 tension=5., pot_lin=1., pot_cub=0.2):
        self.N, self.dt, self.damp = N, dt, damping
        self.tension, self.pot_lin, self.pot_cub = tension, pot_lin, pot_cub

        self.lock = threading.Lock()
        self.phi   = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)

        # gaussian pulse
        x = np.arange(N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r2      = (N/6)**2
        c       = N//2
        self.phi[:] = 2*np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2)/(2*r2))

        # 6-point laplacian kernel
        self.kern = np.zeros((3,3,3), np.float32)
        self.kern[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            self.kern[dx,dy,dz] = 1

    def step(self, n_steps=1):
        for _ in range(n_steps):
            lap = convolve(self.phi, self.kern, mode='wrap')
            Vp  = -self.pot_lin*self.phi + self.pot_cub*self.phi**3
            c2  = 1.0/(1.0 + self.tension*self.phi**2 + 1e-6)
            acc = c2*lap - Vp

            vel      = self.phi - self.phi_o
            phi_next = self.phi + (1-self.damp*self.dt)*vel + self.dt**2*acc
            self.phi_o[:] = self.phi
            self.phi[:]   = phi_next

# ── simulation thread ─────────────────────────────────────
paused = False
def sim_worker(sim, q, stop_evt):
    global paused
    while not stop_evt.is_set():
        if not paused:
            sim.step(2)                    # ~80 Hz physics
            with sim.lock:
                if not q.full():
                    q.put(sim.phi.copy())
        time.sleep(0.01)

sim      = MiniWoW()
field_q  = queue.Queue(maxsize=2)
stop_evt = threading.Event()
threading.Thread(target=sim_worker,
                 args=(sim, field_q, stop_evt),
                 daemon=True).start()

# ── Ursina setup ──────────────────────────────────────────
app = Ursina(fullscreen=True, development_mode=False)
window.color               = color.rgb(2,2,10)
window.fps_counter.enabled = False
window.title               = 'Enhanced Magic Box'

EditorCamera()  # WASD + RMB

sun  = DirectionalLight(rotation=(45,-45,45), color=color.white, shadows=True)
AmbientLight(color=color.rgba(0.3,0.3,0.5,0.1))

container = Entity()
surface   = Entity(parent=container, double_sided=True,
                   shader=lit_with_shadows_shader)

ground = Entity(model='plane', scale=100, y=-15,
                color=color.dark_gray, texture='white_cube',
                texture_scale=(100,100))

# ── globals & helpers ─────────────────────────────────────
iso_val, time_val = 1.0, 0.0

def update_mesh(phi):
    global surface
    try:
        v, f, n, _ = marching_cubes(phi, level=iso_val)
        if v.size == 0:
            surface.visible = False
            return
        v -= v.mean(0)                     # center mesh
        surface.model = Mesh(
            vertices=v.tolist(),
            triangles=f.flatten().tolist(),
            normals=n.tolist(),
            mode='triangle'
        )
        surface.visible = True
    except Exception as e:
        print('marching-cubes error:', e)
        surface.visible = False

def update_color():
    hue = (time_val*0.1)%1.0
    r,g,b = colorsys.hsv_to_rgb(hue,0.8,0.9)
    return color.rgba(r,g,b,0.9)

# ── main update loop ──────────────────────────────────────
def update():
    global time_val
    if paused:                         # physics frozen, but UI still runs
        return
    time_val += time.dt
    container.rotation_y += time.dt*5

    try:                               # newest field first
        while True:
            phi = field_q.get_nowait()
            update_mesh(phi)
    except queue.Empty:
        pass

    if time_val % 1.0 < 0.02:
        surface.color = update_color()

# ── keyboard / UI ─────────────────────────────────────────
def input(key):
    global iso_val, paused
    if key == 'left arrow':
        iso_val = max(-2.0, iso_val-0.05)
        print('Iso →', f'{iso_val:.2f}')
        with sim.lock:
            update_mesh(sim.phi.copy())      # live refresh even if paused
    elif key == 'right arrow':
        iso_val = min( 2.0, iso_val+0.05)
        print('Iso →', f'{iso_val:.2f}')
        with sim.lock:
            update_mesh(sim.phi.copy())
    elif key == 'p':
        paused = not paused
        print('Paused' if paused else 'Resumed')
    elif key == 'escape':
        stop_evt.set()
        application.quit()
    elif key == 'f':
        window.fullscreen = not window.fullscreen

# ── slider factory ────────────────────────────────────────
def add_slider(label, attr, rng, y):
    txt = Text(text=f'{label}: {getattr(sim,attr):.3f}',
               x=-0.83, y=y+0.04, parent=camera.ui, scale=0.75)
    sld = Slider(min=rng[0], max=rng[1], default=getattr(sim,attr),
                 step=(rng[1]-rng[0])/200,
                 x=-0.85, y=y, scale=0.3, parent=camera.ui)
    def changed():                          # no args from Ursina
        val = sld.value
        setattr(sim, attr, val)
        txt.text = f'{label}: {val:.3f}'
    sld.on_value_changed = changed

add_slider('dt',       'dt',    (0.01,0.2),   0.35)
add_slider('damping',  'damp',  (0.0,0.05),   0.25)
add_slider('tension',  'tension',(0.0,20.0),  0.15)
add_slider('pot_lin',  'pot_lin',(0.0,2.0),   0.05)
add_slider('pot_cub',  'pot_cub',(0.0,1.0),  -0.05)

Text(
    text='WASD+RMB fly | ←/→ iso | P pause | ESC quit',
    y=-0.45, x=0, origin=(0,0),
    background=True, background_color=color.rgba(0,0,0,128),
    parent=camera.ui
)

app.run()
