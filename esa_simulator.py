"""
ESA Simulator (v1.1.0-alpha)
Developer: Kilho Baek (kilho.baek [at] gmail.com)
"""

import os
import sys

import tkinter as tk
from tkinter import messagebox

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Tkinter backend for Matplotlib


# === PyInstaller --windowed keyboard issue workaround =========================
# Prevents crash when standard output/error are missing in windowed mode
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')
# =============================================================================

plt.rcParams['toolbar'] = 'None'

# =============================================================================
# 1. Physical Constants
# =============================================================================
Q_ELEM = 1.602e-19    # Elementary charge (C)
M_PROTON = 1.672e-27  # Proton mass (kg)

# =============================================================================
# 2. Physics Engine
# =============================================================================
def calculate_trajectory(r_in_m, r_out_m, alpha_deg, energy_kev, v_in_mag, ion_type):
    """
    Calculate the trajectory of a single reference ion.
    """
    q_sign = 1.0 if ion_type == 'Cation (+)' else -1.0
    qq = Q_ELEM * q_sign

    energy_joules = energy_kev * 1000.0 * Q_ELEM
    if energy_joules <= 0:
        energy_joules = 1e-20
    v0 = np.sqrt(2 * energy_joules / M_PROTON)

    r_center = (r_in_m + r_out_m) / 2.0
    x_center = 0.0

    alpha_rad = np.radians(alpha_deg)
    initial_state = [x_center, r_center, v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)]

    def ion_motion(_, state):
        px, py, vx, vy = state
        rr = np.sqrt(px**2 + py**2)
        if rr == 0:
            return [0, 0, 0, 0]

        v_in_volts = -v_in_mag if ion_type == 'Cation (+)' else v_in_mag
        energy_r = (v_in_volts * r_in_m * r_out_m) / (r_out_m - r_in_m) * (1.0 / rr**2)

        ax = (qq / M_PROTON) * energy_r * (px / rr)
        ay = (qq / M_PROTON) * energy_r * (py / rr)
        return [vx, vy, ax, ay]

    def collision_event(_, state):
        px, py, _, _ = state
        r = np.sqrt(px**2 + py**2)
        if r <= r_in_m or r >= r_out_m or px < -1e-5 or py < 0:
            return 0
        return 1
    collision_event.terminal = True

    sol = solve_ivp(ion_motion, (0, 1e-5), initial_state, events=collision_event, max_step=1e-9)
    is_detected = (sol.y[1][-1] <= 1e-5) and (r_in_m <= sol.y[0][-1] <= r_out_m)

    return sol.y[0], sol.y[1], is_detected

@njit(fastmath=True)
def _calculate_detection_numba(num, dist_type_flag, r_in, r_out, energy0_kev, v_in_mag, q_sign):
    """
    High-performance Numba core for Monte Carlo multi-particle simulation.
    """
    qq = Q_ELEM * q_sign

    # Force incident angle to 0 for Monte Carlo (ignore textbox alpha)
    alpha0_rad = 0.0
    delta_alpha = 5.0 * np.pi / 180.0
    delta_e = energy0_kev * 0.2
    r_center = (r_in + r_out) / 2.0

    v_in_volts = -v_in_mag if q_sign > 0 else v_in_mag
    const_e = v_in_volts * r_in * r_out / (r_out - r_in)
    q_over_m = qq / M_PROTON

    # Pre-allocate result array to prevent memory reallocation overhead
    det_energies = np.empty(num, dtype=np.float64)
    det_count = 0

    # Fast independent tracking loop (avoids vectorized memory bottlenecks)
    for _ in range(num):
        # Initialize particle state based on distribution type
        if dist_type_flag == 0:  # Uniform
            y0 = np.random.uniform(r_in + 1e-5, r_out - 1e-5)
            alpha = np.random.uniform(alpha0_rad - delta_alpha, alpha0_rad + delta_alpha)
            energy = np.random.uniform(energy0_kev - delta_e, energy0_kev + delta_e)
        else:  # Gaussian
            y0 = np.random.normal(r_center, (r_out - r_in) / 6.0)
            if y0 < r_in + 1e-5:
                y0 = r_in + 1e-5
            if y0 > r_out - 1e-5:
                y0 = r_out - 1e-5
            alpha = np.random.normal(alpha0_rad, delta_alpha / 3.0)
            energy = np.random.normal(energy0_kev, delta_e / 3.0)

        energy_j = energy * 1000.0 * Q_ELEM
        if energy_j < 1e-20:
            energy_j = 1e-20
        v0 = np.sqrt(2.0 * energy_j / M_PROTON)

        x = 0.0
        y = y0
        vx = v0 * np.cos(alpha)
        vy = v0 * np.sin(alpha)

        # Calculate time step (dt) scaled to initial velocity
        dt = ((np.pi / 2.0 * r_center) / v0) / 5000.0

        # Physics integration loop
        for _ in range(10000):
            r2 = x*x + y*y
            r = np.sqrt(r2)

            ax = q_over_m * const_e * x / (r2 * r)
            ay = q_over_m * const_e * y / (r2 * r)

            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt

            r_new = np.sqrt(x*x + y*y)

            # Stop tracking if particle hits the electrode wall
            if r_new <= r_in or r_new >= r_out:
                break

            # Check if particle passes the detector plane (y <= 0)
            if y <= 0:
                if x > 0 and r_in <= x <= r_out:
                    det_energies[det_count] = energy
                    det_count += 1
                break

    # Return only the populated slice of the array
    return det_count, det_energies[:det_count]

def calculate_detection(num, dist_type, r_in, r_out, energy0_kev, v_in_mag, ion_type):
    """
    Wrapper for the Numba JIT-compiled detection function.
    """
    # Convert string flags to integers/floats for Numba compatibility
    dist_type_flag = 0 if dist_type == 'Uniform' else 1
    q_sign = 1.0 if ion_type == 'Cation (+)' else -1.0

    det_count, det_energies = _calculate_detection_numba(
        num, dist_type_flag, r_in, r_out, energy0_kev, v_in_mag, q_sign
    )

    return det_count, det_energies

class ESASimulatorApp:
    """
    Main Application Class for ESA Simulator GUI
    """
    def __init__(self, tk_root):
        self.root = tk_root
        self.root.title("ESA Simulator")
        self.root.geometry("1200x600")
        self.root.configure(bg='#ecf0f1')

        # Default UI parameters
        self.default_v_in_mag = 100.0
        self.default_num_ions = 10000
        self.default_r_in_mm = 37.5
        self.default_r_out_mm = 39.5
        self.default_alpha = 0.0

        # Internal state variables to preserve values between updates
        self.last_theory_e = 0.0
        self.last_n_ions = self.default_num_ions
        self.last_det_count = 0

        self.setup_menu()
        self.create_widgets()
        self.create_matplotlib_canvas()

        # Bind the Return/Enter key to trigger the simulation globally
        self.root.bind('<Return>', self.run_simulation)

        # Trigger initial calculation on startup
        self.run_simulation()

    def setup_menu(self):
        """
        Set up the top application menu bar.
        """
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About...", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def show_about(self):
        """
        Display application information via About dialog.
        """
        about_msg = (
            "ESA Simulator (v1.1.0-alpha)\n\n"
            "A simulator that calculates ion trajectories and the detectable particle "
            "energy distribution for science mission planning and instrument design.\n\n"
            "Developer: Kilho Baek (kilho.baek@gmail.com)"
        )
        messagebox.showinfo("About ESA Simulator", about_msg)

    def create_widgets(self):
        """
        Construct left-panel UI components (textboxes, buttons, radio buttons).
        """
        left_panel = tk.Frame(self.root, bg='#ecf0f1', width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        tk.Label(
            left_panel, text="ESA Parameters", font=('Arial', 14, 'bold'),
            bg='#ecf0f1', fg='#2c3e50'
        ).pack(pady=(0, 15))

        # Tkinter variables for user inputs
        self.var_ion = tk.StringVar(value="Cation (+)")
        self.var_dist = tk.StringVar(value="Uniform")
        self.var_vin = tk.StringVar(value=str(self.default_v_in_mag))
        self.var_num = tk.StringVar(value=str(self.default_num_ions))
        self.var_rin = tk.StringVar(value=str(self.default_r_in_mm))
        self.var_rout = tk.StringVar(value=str(self.default_r_out_mm))
        self.var_alpha = tk.StringVar(value=str(self.default_alpha))

        # Ion Type radio buttons
        frame_ion = tk.LabelFrame(
            left_panel, text="Ion Type",
            bg='#ecf0f1', font=('Arial', 11, 'bold')
        )
        frame_ion.pack(fill=tk.X, pady=10)
        tk.Radiobutton(
            frame_ion, text="Cation (+)", variable=self.var_ion, value="Cation (+)",
            bg='#ecf0f1', font=('Arial', 11)
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            frame_ion, text="Anion (-)", variable=self.var_ion, value="Anion (-)",
            bg='#ecf0f1', font=('Arial', 11)
        ).pack(anchor=tk.W)

        # Distribution Type radio buttons
        frame_dist = tk.LabelFrame(
            left_panel, text="Distribution Type", bg='#ecf0f1', font=('Arial', 11, 'bold')
        )
        frame_dist.pack(fill=tk.X, pady=10)
        tk.Radiobutton(
            frame_dist, text="Uniform", variable=self.var_dist, value="Uniform",
            bg='#ecf0f1', font=('Arial', 11)
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            frame_dist, text="Gaussian", variable=self.var_dist, value="Gaussian",
            bg='#ecf0f1', font=('Arial', 11)
        ).pack(anchor=tk.W)

        # Helper function for text inputs
        def create_entry(parent, label_text, var):
            f = tk.Frame(parent, bg='#ecf0f1')
            f.pack(fill=tk.X, pady=5)
            tk.Label(
                f, text=label_text, width=10, anchor='e', bg='#ecf0f1', font=('Arial', 11)
            ).pack(side=tk.LEFT)
            tk.Entry(
                f, textvariable=var, font=('Arial', 11), width=7
            ).pack(side=tk.LEFT, padx=5)

        create_entry(left_panel, "|V_in| (V):", self.var_vin)
        create_entry(left_panel, "# of Ions:", self.var_num)
        create_entry(left_panel, "R_in (mm):", self.var_rin)
        create_entry(left_panel, "R_out (mm):", self.var_rout)
        create_entry(left_panel, "α  (°):", self.var_alpha)

        # Main execution button
        btn_run = tk.Button(
            left_panel, text="Run Simulation", font=('Arial', 12, 'bold'),
            bg='#2ecc71', fg='white', activebackground='#27ae60', activeforeground='white',
            command=self.run_simulation, cursor='hand2'
        )
        btn_run.pack(fill=tk.X, pady=25, ipady=10)

        # Developer contact footprint
        lbl_contact = tk.Label(
            left_panel, text="Contact: kilho.baek@gmail.com",
            font=('Arial', 9, 'italic'), bg='#ecf0f1', fg='#95a5a6'
        )
        lbl_contact.pack(side=tk.BOTTOM, anchor=tk.SW)

    def create_matplotlib_canvas(self):
        """
        Embed Matplotlib figures into the right panel (Geometry & Histogram).
        """
        right_panel = tk.Frame(self.root, bg='#ffffff')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_geom, self.ax_hist) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]}, facecolor='#ffffff'
        )

        # Adjust margins to leave room for the interactive slider overlay
        self.fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.88, wspace=0.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Overlay Reference Ion Energy Slider on top of Geometry Plot ---
        # Position: [left, bottom, width, height]
        self.ax_slider = self.fig.add_axes([0.20, 0.18, 0.23, 0.03], facecolor='#ecf0f1')
        self.slider_energy = Slider(
            ax=self.ax_slider, label='Energy',
            valmin=0.1, valmax=2.0, valinit=1.0, valstep=0.01, valfmt='%.2f keV',
            color='#3498db'
        )
        self.slider_energy.on_changed(self.on_slider_change)

        # Initialize base geometry line objects
        self.theta = np.linspace(0, np.pi/2, 100)
        self.inner_line, = self.ax_geom.plot(
            [], [], color='#7f8c8d', lw=2, label='Inner Electrode'
        )
        self.outer_line, = self.ax_geom.plot(
            [], [], color='#7f8c8d', lw=2, label='Outer Electrode (0V)'
        )
        self.entrance_dot, = self.ax_geom.plot(
            [], [], marker='o', color='#9b59b6', markersize=4, label='Ref. Ion'
        )
        self.detector_line, = self.ax_geom.plot(
            [], [], color='#27ae60', lw=6, label='Detector'
        )
        self.trajectory_line, = self.ax_geom.plot(
            [], [], color='#e74c3c', lw=1.5, label='Ref. Trajectory'
        )

        # Interactive DETECTED/LOST text indicator below detector
        self.detector_status_text = self.ax_geom.text(
            0, -0.002, '', ha='center', va='top', fontsize=12, fontweight='bold'
        )

        # Detailed status text box
        self.status_text = self.ax_geom.text(
            0.05, 0.29, '', transform=self.ax_geom.transAxes, ha='left', va='top',
            fontsize=11, color='#2c3e50',
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor='#ffffff', edgecolor='#bdc3c7', alpha=0.5
            )
        )

    def on_slider_change(self, val):
        """
        Fast callback for the energy slider. 
        Only recalculates single reference trajectory, bypassing heavy MC simulation.
        """
        try:
            rin_mm = float(self.var_rin.get())
            rout_mm = float(self.var_rout.get())
            alpha = float(self.var_alpha.get())
            v_in_mag = float(self.var_vin.get())
        except ValueError:
            return

        rin_m = rin_mm / 1000.0
        rout_m = rout_mm / 1000.0
        ion_t = self.var_ion.get()

        energy_kev = float(val)

        # Recalculate single trajectory using the slider's energy and textbox alpha
        ref_x, ref_y, is_det = calculate_trajectory(
            rin_m, rout_m, alpha, energy_kev, v_in_mag, ion_t
        )
        self.trajectory_line.set_data(ref_x, ref_y)

        # Update visual pass/fail indicator below detector
        rc = (rin_m + rout_m) / 2.0
        self.detector_status_text.set_position((rc, -0.002))
        if is_det:
            self.detector_status_text.set_text('DETECTED')
            self.detector_status_text.set_color('#27ae60')
        else:
            self.detector_status_text.set_text('LOST')
            self.detector_status_text.set_color('#e74c3c')

        self.canvas.draw_idle()

    def run_simulation(self, _event=None):
        """
        Execute full Monte Carlo simulation and update all visual elements.
        """
        try:
            v_in_mag = float(self.var_vin.get())
            n_ions = int(self.var_num.get())
            rin_mm = float(self.var_rin.get())
            rout_mm = float(self.var_rout.get())
            alpha = float(self.var_alpha.get())
        except ValueError:
            messagebox.showwarning(
                "Input Error", "Please enter valid numerical values."
            )
            return

        rin_m = rin_mm / 1000.0
        rout_m = rout_mm / 1000.0
        r_center_m = (rin_m + rout_m) / 2.0

        ion_t = self.var_ion.get()
        dist_t = self.var_dist.get()

        if rout_m <= rin_m or rin_m <= 0:
            messagebox.showwarning(
                "Geometry Error", "Invalid ESA dimensions. Ensure R_out > R_in > 0."
            )
            return
        if n_ions <= 1000:
            messagebox.showwarning(
                "Geometry Error", "Invalid Number of Ions. "
                "Please enter a value greater than 1000."
            )
            return

        # 1. Calculate Theoretical Energy
        energy_kev = (v_in_mag * rin_m * rout_m) / (2.0 * 1000.0 * r_center_m * (rout_m - rin_m))
        self.last_theory_e = energy_kev

        # Calculate min/max detectable energy assuming perfect 0 deg incident angle
        e_min_theo = energy_kev * (rin_m / r_center_m)
        e_max_theo = energy_kev * (rout_m / r_center_m)

        # Format slider boundaries and default value to 2 decimal places
        slider_min = round(e_min_theo, 2) - 0.1
        slider_max = round(e_max_theo, 2) + 0.1
        slider_val = round(energy_kev, 2)

        if slider_min >= slider_max:
            slider_min = max(0.1, slider_val - 0.1)
            slider_max = slider_val + 0.1

        # Temporarily suspend slider events to avoid duplicate trajectory calculation
        self.slider_energy.eventson = False
        self.slider_energy.valmin = slider_min
        self.slider_energy.valmax = slider_max
        self.slider_energy.ax.set_xlim(slider_min, slider_max)
        self.slider_energy.set_val(slider_val)
        self.slider_energy.eventson = True

        # 2. Calculate Reference Trajectory (Uses slider energy & textbox alpha)
        ref_x, ref_y, is_det = calculate_trajectory(
            rin_m, rout_m, alpha, slider_val, v_in_mag, ion_t
        )

        # 3. Execute Monte Carlo Batch Detection (Uses theoretical energy & fixed alpha=0)
        det_count, det_energies = calculate_detection(
            n_ions, dist_t, rin_m, rout_m, energy_kev, v_in_mag, ion_t
        )

        self.last_n_ions = n_ions
        self.last_det_count = det_count

        # Update Geometry Plot shapes
        rc = (rin_m + rout_m) / 2.0
        self.inner_line.set_data(rin_m * np.cos(self.theta), rin_m * np.sin(self.theta))
        self.outer_line.set_data(rout_m * np.cos(self.theta), rout_m * np.sin(self.theta))
        self.entrance_dot.set_data([0.0], [rc])
        self.detector_line.set_data([rin_m, rout_m], [0.0, 0.0])
        self.trajectory_line.set_data(ref_x, ref_y)

        # Refresh detector text indicator
        self.detector_status_text.set_position((rc, -0.002))
        if is_det:
            self.detector_status_text.set_text('DETECTED')
            self.detector_status_text.set_color('#27ae60')
        else:
            self.detector_status_text.set_text('LOST')
            self.detector_status_text.set_color('#e74c3c')

        self.ax_geom.set_xlim(-0.01, 0.05)
        self.ax_geom.set_ylim(-0.01, 0.05)
        self.ax_geom.set_aspect('equal', adjustable='box')
        self.ax_geom.set_title("ESA Geometry", fontsize=12, fontweight='bold', pad=15)
        self.ax_geom.set_xlabel("X Position (m)")
        self.ax_geom.set_ylabel("Y Position (m)")
        self.ax_geom.grid(True, linestyle='--', color='#bdc3c7', alpha=0.7)
        self.ax_geom.legend(loc='upper right', framealpha=0.5)

        # Refresh detailed status box
        pct = (det_count / n_ions) * 100.0 if n_ions > 0 else 0.0
        text_str  = f"$E_\\mathrm{{theory}}$ = {self.last_theory_e:.3f} keV, $\\alpha$ = {alpha}°"
        text_str += f"\nDetection: {det_count}/{n_ions} ({pct:.1f}%)"
        self.status_text.set_text(text_str)

        # 4. Update Energy Distribution Histogram
        self.ax_hist.clear()
        if len(det_energies) > 0:
            mini = round(min(det_energies)*10)/10.0-0.1
            maxi = round(max(det_energies)*10)/10.0+0.1
            nbin = round((maxi-mini)/0.01)

            # Require minimum sample size for reliable Gaussian fitting
            if len(det_energies) > 100:
                counts, bins, _ = self.ax_hist.hist(
                    det_energies, bins=nbin, range=(mini, maxi),
                    color='#3498db', edgecolor='#2980b9', alpha=0.5
                )
                def gauss(xx, *pp):
                    aa, mu, sigma = pp
                    return aa * np.exp(-(xx - mu)**2 / (2. * sigma**2))

                bin_centers = (bins[:-1] + bins[1:]) / 2
                p0 = [np.max(counts), np.mean(det_energies), np.std(det_energies)]

                try:
                    coeff, _ = curve_fit(gauss, bin_centers, counts, p0=p0)
                    mu, sigma = coeff[1], coeff[2]
                    fwhm = 2.355 * abs(sigma)
                    resolution = (fwhm / mu) * 100.0

                    x_fit = np.linspace(bins[0], bins[-1], 200)
                    self.ax_hist.plot(
                        x_fit, gauss(x_fit, *coeff), label='Gaussian Fit',
                        color='#e74c3c', lw=2.5
                    )

                    fit_info  = f"$\\mathbf{{\\mu}}$ = {mu:.3f} keV"
                    fit_info += f"\n$\\mathbf{{\\Delta E/E}}$ = {resolution:.1f}%"
                    self.ax_hist.text(
                        0.95, 0.95, fit_info, transform=self.ax_hist.transAxes,
                        ha='right', va='top', fontsize=11, fontweight='bold', color='#c0392b',
                        bbox=dict(boxstyle="round,pad=0.5",
                                  facecolor='#ffffff', edgecolor='#e74c3c', alpha=0.5
                        )
                    )
                except (RuntimeError, ValueError, OverflowError):
                    pass
            else:
                self.ax_hist.hist(
                    det_energies, bins=nbin, range=(mini, maxi),
                    color='#3498db', edgecolor='#2980b9', alpha=0.5
                )

        self.ax_hist.set_title(
            "Detected Energy Distribution", fontsize=12, fontweight='bold', pad=15
        )
        self.ax_hist.set_xlabel("Energy (keV)")
        self.ax_hist.set_ylabel("Counts")
        self.ax_hist.grid(True, linestyle='--', color='#bdc3c7', alpha=0.7)

        # 5. Flush layout and render everything
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ESASimulatorApp(root)

    # Force close the entire Python process when window is closed
    root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0))
    root.mainloop()
