import sys, os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('TkAgg') # Tkinter backend for Matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox

# --- PyInstaller --windowed keyboard issue workaround ------------------------
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')
# -----------------------------------------------------------------------------

plt.rcParams['toolbar'] = 'None'

# =============================================================================
# 1. Physical Constants
# =============================================================================
q_elem = 1.602e-19    # Elementary charge (C)
m_proton = 1.672e-27  # Proton mass (kg)

# =============================================================================
# 2. Physics Engine
# =============================================================================
def calculate_trajectory(r_in_m, r_out_m, alpha_deg, energy_kev, v_in_mag, ion_type):
    q_sign = 1.0 if ion_type == 'Cation (+)' else -1.0
    q = q_elem * q_sign
    
    energy_joules = energy_kev * 1000.0 * q_elem
    if energy_joules <= 0: energy_joules = 1e-20
    v0 = np.sqrt(2 * energy_joules / m_proton)
    
    r_center = (r_in_m + r_out_m) / 2.0
    x_center = 0.0
    
    alpha_rad = np.radians(alpha_deg)
    initial_state = [x_center, r_center, v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)]
    
    def ion_motion(t, state):
        px, py, vx, vy = state
        r = np.sqrt(px**2 + py**2)
        if r == 0: return [0, 0, 0, 0]
        
        v_in_volts = -v_in_mag if ion_type == 'Cation (+)' else v_in_mag
        E_r = (v_in_volts * r_in_m * r_out_m) / (r_out_m - r_in_m) * (1.0 / r**2)
        
        ax = (q / m_proton) * E_r * (px / r)
        ay = (q / m_proton) * E_r * (py / r)
        return [vx, vy, ax, ay]

    def collision_event(t, state):
        px, py, _, _ = state
        r = np.sqrt(px**2 + py**2)
        if r <= r_in_m or r >= r_out_m or px < -1e-5 or py < 0:
            return 0
        return 1
    collision_event.terminal = True

    sol = solve_ivp(ion_motion, (0, 1e-5), initial_state, events=collision_event, max_step=1e-9)
    is_detected = (sol.y[1][-1] <= 1e-5) and (r_in_m <= sol.y[0][-1] <= r_out_m)
    return sol.y[0], sol.y[1], is_detected

def calculate_detection(N, dist_type, r_in, r_out, alpha0_deg, E0_kev, v_in_mag, ion_type):
    q_sign = 1.0 if ion_type == 'Cation (+)' else -1.0
    q = q_elem * q_sign
    
    delta_alpha = np.radians(5.0)
    delta_E = E0_kev * 0.2
    r_center = (r_in + r_out) / 2.0
    
    if dist_type == 'Uniform':
        y0 = np.random.uniform(r_in + 1e-5, r_out - 1e-5, N)
        alpha = np.random.uniform(np.radians(alpha0_deg) - delta_alpha, np.radians(alpha0_deg) + delta_alpha, N)
        E = np.random.uniform(E0_kev - delta_E, E0_kev + delta_E, N)
    else: 
        y0 = np.clip(np.random.normal(r_center, (r_out - r_in) / 6.0, N), r_in + 1e-5, r_out - 1e-5)
        alpha = np.random.normal(np.radians(alpha0_deg), delta_alpha / 3.0, N)
        E = np.random.normal(E0_kev, delta_E / 3.0, N)
        
    v0 = np.sqrt(2 * np.maximum(E * 1000.0 * q_elem, 1e-20) / m_proton)
    
    x, y = np.zeros(N), y0
    vx, vy = v0 * np.cos(alpha), v0 * np.sin(alpha)
    
    active = np.ones(N, dtype=bool)
    detected = np.zeros(N, dtype=bool)
    
    const_E = (-v_in_mag if ion_type == 'Cation (+)' else v_in_mag) * r_in * r_out / (r_out - r_in)
    q_over_m = q / m_proton
    dt = ((np.pi / 2.0 * r_center) / np.mean(v0)) / 5000
    
    for _ in range(10000):
        if not np.any(active): break
        xa, ya = x[active], y[active]
        vxa, vya = vx[active], vy[active]
        
        r2 = xa**2 + ya**2
        r = np.sqrt(r2)
        
        vxa += q_over_m * const_E * xa / (r2 * r) * dt
        vya += q_over_m * const_E * ya / (r2 * r) * dt
        xa += vxa * dt
        ya += vya * dt
        
        x[active], y[active] = xa, ya
        vx[active], vy[active] = vxa, vya
        
        r_new = np.sqrt(xa**2 + ya**2)
        hit_wall = (r_new <= r_in) | (r_new >= r_out)
        hit_detector = (ya <= 0) & (xa > 0)
        
        detected_now = hit_detector & (xa >= r_in) & (xa <= r_out)
        active_idx = np.where(active)[0]
        detected[active_idx[detected_now]] = True
        active[active_idx[hit_wall | (ya <= 0)]] = False

    return np.sum(detected), E[detected]

# =============================================================================
# 3. Main GUI Application Class
# =============================================================================
class ESASimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ESA Simulator")
        self.root.geometry("1200x600")
        self.root.configure(bg='#ecf0f1')
        
        # Default parameters
        self.default_v_in_mag = 100.0
        self.default_num_ions = 10000
        self.default_r_in_mm = 37.5
        self.default_r_out_mm = 39.5
        self.default_alpha = 0.0

        self.setup_menu()
        self.create_widgets()
        self.create_matplotlib_canvas()
        
        # Bind the Return key to run simulation
        self.root.bind('<Return>', self.run_simulation)
        
        # Initial simulation run
        self.run_simulation()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About...", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def show_about(self):
        about_msg = (
            "ESA Simulator (v1.0.0-beta)\n\n"
            "A simulator that calculates ion trajectories and the detectable particle energy distribution for science mission planning and instrument design.\n\n"
            "Developer: Kilho Baek (kilho.baek@gmail.com)"
        )
        messagebox.showinfo("About ESA Simulator", about_msg)

    def create_widgets(self):
        # Left panel for controls
        left_panel = tk.Frame(self.root, bg='#ecf0f1', width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        tk.Label(left_panel, text="ESA Parameters", font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=(0, 15))
        
        # Variable initialization
        self.var_ion = tk.StringVar(value="Cation (+)")
        self.var_dist = tk.StringVar(value="Uniform")
        self.var_vin = tk.StringVar(value=str(self.default_v_in_mag))
        self.var_num = tk.StringVar(value=str(self.default_num_ions))
        self.var_rin = tk.StringVar(value=str(self.default_r_in_mm))
        self.var_rout = tk.StringVar(value=str(self.default_r_out_mm))
        self.var_alpha = tk.StringVar(value=str(self.default_alpha))

        # Radio buttons (Ion Type)
        frame_ion = tk.LabelFrame(left_panel, text="Ion Type", bg='#ecf0f1', font=('Arial', 11, 'bold'))
        frame_ion.pack(fill=tk.X, pady=10)
        tk.Radiobutton(frame_ion, text="Cation (+)", variable=self.var_ion, value="Cation (+)", bg='#ecf0f1', font=('Arial', 11)).pack(anchor=tk.W)
        tk.Radiobutton(frame_ion, text="Anion (-)", variable=self.var_ion, value="Anion (-)", bg='#ecf0f1', font=('Arial', 11)).pack(anchor=tk.W)

        # Radio buttons (Distribution Type)
        frame_dist = tk.LabelFrame(left_panel, text="Distribution Type", bg='#ecf0f1', font=('Arial', 11, 'bold'))
        frame_dist.pack(fill=tk.X, pady=10)
        tk.Radiobutton(frame_dist, text="Uniform", variable=self.var_dist, value="Uniform", bg='#ecf0f1', font=('Arial', 11)).pack(anchor=tk.W)
        tk.Radiobutton(frame_dist, text="Gaussian", variable=self.var_dist, value="Gaussian", bg='#ecf0f1', font=('Arial', 11)).pack(anchor=tk.W)

        # Textbox for parameters
        def create_entry(parent, label_text, var):
            f = tk.Frame(parent, bg='#ecf0f1')
            f.pack(fill=tk.X, pady=5)
            tk.Label(f, text=label_text, width=10, anchor='e', bg='#ecf0f1', font=('Arial', 11)).pack(side=tk.LEFT)
            tk.Entry(f, textvariable=var, font=('Arial', 11), width=7).pack(side=tk.LEFT, padx=5)

        # Textbox entries for parameters
        create_entry(left_panel, "|V_in| (V):", self.var_vin)
        create_entry(left_panel, "# of Ions:", self.var_num)
        create_entry(left_panel, "R_in (mm):", self.var_rin)
        create_entry(left_panel, "R_out (mm):", self.var_rout)
        create_entry(left_panel, "α  (°):", self.var_alpha)

        # Button for running simulation
        btn_run = tk.Button(left_panel, text="Run Simulation", font=('Arial', 12, 'bold'), 
                            bg='#2ecc71', fg='white', activebackground='#27ae60', activeforeground='white',
                            command=self.run_simulation, cursor='hand2')
        btn_run.pack(fill=tk.X, pady=25, ipady=10)

        # Contact Info at the bottom left corner
        lbl_contact = tk.Label(left_panel, text="Contact: kilho.baek@gmail.com", font=('Arial', 9, 'italic'), bg='#ecf0f1', fg='#95a5a6')
        lbl_contact.pack(side=tk.BOTTOM, anchor=tk.SW)

    def create_matplotlib_canvas(self):
        # 우측 그래프 패널 생성 (1 row, 2 columns)
        right_panel = tk.Frame(self.root, bg='#ffffff')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_geom, self.ax_hist) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]}, facecolor='#ffffff')
        self.fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.88, wspace=0.2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add Matplotlib toolbar
        self.theta = np.linspace(0, np.pi/2, 100)
        self.inner_line, = self.ax_geom.plot([], [], color='#7f8c8d', lw=2, label='Inner Electrode')
        self.outer_line, = self.ax_geom.plot([], [], color='#7f8c8d', lw=2, label='Outer Electrode (0V)')
        self.entrance_dot, = self.ax_geom.plot([], [], marker='o', color='#9b59b6', markersize=4, label='Ref. Ion')
        self.detector_line, = self.ax_geom.plot([], [], color='#27ae60', lw=6, label='Detector')
        self.trajectory_line, = self.ax_geom.plot([], [], color='#e74c3c', lw=1.5, label='Ref. Trajectory')
        
        self.status_text = self.ax_geom.text(0.05, 0.12, '', transform=self.ax_geom.transAxes, ha='left', va='top', 
                                             fontsize=11, color='#2c3e50',
                                             bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffffff', edgecolor='#bdc3c7', alpha=0.5))

    def run_simulation(self, _event=None):
        try:
            v_in_mag = float(self.var_vin.get())
            n_ions = int(self.var_num.get())
            rin_mm = float(self.var_rin.get())
            rout_mm = float(self.var_rout.get())
            alpha = float(self.var_alpha.get())
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter valid numerical values.")
            return

        rin_m = rin_mm / 1000.0
        rout_m = rout_mm / 1000.0
        r_center_m = (rin_m + rout_m) / 2.0
        
        ion_t = self.var_ion.get()
        dist_t = self.var_dist.get()
        
        if rout_m <= rin_m or rin_m <= 0:
            messagebox.showwarning("Geometry Error", "Invalid ESA dimensions. Ensure R_out > R_in > 0.")
            return
        if n_ions <= 1000:
            messagebox.showwarning("Geometry Error", "Invalid Number of Ions. Please enter a value greater than 1000.")
            return

        # 1. Calculate Trajectory and Detection
        energy_kev = (v_in_mag * rin_m * rout_m) / (2.0 * 1000.0 * r_center_m * (rout_m - rin_m))
        ref_x, ref_y, is_det = calculate_trajectory(rin_m, rout_m, 0.0, energy_kev, v_in_mag, ion_t)
        det_count, det_energies = calculate_detection(n_ions, dist_t, rin_m, rout_m, alpha, energy_kev, v_in_mag, ion_t)

        # 2. Update Geometry
        rc = (rin_m + rout_m) / 2.0
        self.inner_line.set_data(rin_m * np.cos(self.theta), rin_m * np.sin(self.theta))
        self.outer_line.set_data(rout_m * np.cos(self.theta), rout_m * np.sin(self.theta))
        self.entrance_dot.set_data([0.0], [rc])
        self.detector_line.set_data([rin_m, rout_m], [0.0, 0.0])
        self.trajectory_line.set_data(ref_x, ref_y)

        # Geometry plot limits and labels
        self.ax_geom.set_xlim(-0.01, 0.05)
        self.ax_geom.set_ylim(-0.01, 0.05)
        self.ax_geom.set_aspect('equal', adjustable='box')
        self.ax_geom.set_title("ESA Simulator", fontsize=12, fontweight='bold', pad=15)
        self.ax_geom.set_xlabel("X Position (m)")
        self.ax_geom.set_ylabel("Y Position (m)")
        self.ax_geom.grid(True, linestyle='--', color='#bdc3c7', alpha=0.7)
        self.ax_geom.legend(loc='upper right', framealpha=0.5)
        
        # Update status text
        pct = (det_count / n_ions) * 100.0 if n_ions > 0 else 0.0
        ref_status = 'Detected' if is_det else 'Lost'
        text_str = f"Reference Ion ({energy_kev:.3f} keV, 0°): {ref_status}\nBatch Detection: {det_count} / {n_ions} ({pct:.1f}%)"
        self.status_text.set_text(text_str)

        # 3. Update Histogram
        self.ax_hist.clear()
        if len(det_energies) > 0:
            mini = round(min(det_energies)*10)/10.0-0.1
            maxi = round(max(det_energies)*10)/10.0+0.1
            nbin = round((maxi-mini)/0.01)
            
            if len(det_energies) > 100:
                counts, bins, _ = self.ax_hist.hist(det_energies, bins=nbin, range=(mini, maxi), color='#3498db', edgecolor='#2980b9', alpha=0.5)
                def gauss(x, *p):
                    A, mu, sigma = p
                    return A * np.exp(-(x - mu)**2 / (2. * sigma**2))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                p0 = [np.max(counts), np.mean(det_energies), np.std(det_energies)] 
                
                try:
                    coeff, _ = curve_fit(gauss, bin_centers, counts, p0=p0)
                    mu, sigma = coeff[1], coeff[2]
                    fwhm = 2.355 * abs(sigma)
                    resolution = (fwhm / mu) * 100.0  
                    
                    x_fit = np.linspace(bins[0], bins[-1], 200)
                    self.ax_hist.plot(x_fit, gauss(x_fit, *coeff), color='#e74c3c', lw=2.5, label='Gaussian Fit')
                    
                    fit_info = f"$\\mathbf{{\\mu}}$ = {mu:.3f} keV\n$\\mathbf{{\\Delta E/E}}$ = {resolution:.1f}%"
                    self.ax_hist.text(0.95, 0.95, fit_info, transform=self.ax_hist.transAxes, ha='right', va='top', fontsize=11, fontweight='bold', color='#c0392b',
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffffff', edgecolor='#e74c3c', alpha=0.5))
                except (RuntimeError, ValueError, OverflowError):
                    pass
            else:
                self.ax_hist.hist(det_energies, bins=nbin, range=(mini, maxi), color='#3498db', edgecolor='#2980b9', alpha=0.5)
                
        self.ax_hist.set_title("Detected Energy Distribution", fontsize=12, fontweight='bold', pad=15)
        self.ax_hist.set_xlabel("Energy (keV)")
        self.ax_hist.set_ylabel("Counts")
        self.ax_hist.grid(True, linestyle='--', color='#bdc3c7', alpha=0.7)

        # 4. Render the canvas
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ESASimulatorApp(root)
    
    # Window close event to completely terminate the process
    root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0))
    root.mainloop()