# ===================== Coupling Matrix TUNING TOOL (PYTHONISTA iPad) =====================
# July 2026

import sys, traceback, clipboard

def ai_error_copy(exctype, value, tb):
    err = "".join(traceback.format_exception(exctype, value, tb))
    print(err)
    @on_main_thread
    def set_clipboard():
        clipboard.set(err)
    set_clipboard()

sys.excepthook = ai_error_copy

import ui, dialogs, numpy as np, matplotlib, io, random, os, threading
from objc_util import on_main_thread

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

plot_lock = threading.Lock()


# ===================== DOMAIN OBJECTS =====================
class SimulationResult:
    def __init__(self, freqs, s11, s21, s22, s11_db, s21_db, s22_db, phase_deg, gd_ns,
                 phase_rel=None):
        self.freqs = freqs
        self.s11 = s11
        self.s21 = s21
        self.s22 = s22
        self.s11_db = s11_db
        self.s21_db = s21_db
        self.s22_db = s22_db
        self.phase_deg = phase_deg  # raw for GD etc.
        self.gd_ns = gd_ns
        self.phase_rel = phase_rel if phase_rel is not None else phase_deg


class MeasurementData:
    def __init__(self, loaded_s21_plot=None, loaded_return_db=None, interps=None):
        self.loaded_s21_plot = loaded_s21_plot
        self.loaded_return_db = loaded_return_db
        self.interps = interps or {}


class Marker:
    def __init__(self, freq=None, color=None, enabled=True):
        self.freq = freq
        self.color = color or 'red'
        self.enabled = enabled
        self.value = None
        self.unit = None
        self.trace = None

    def update_value(self, sim: SimulationResult, secondary_plot, ch1_mode, meas_data=None, state=None):
        if not self.freq or self.freq is None:
            return
        freqs = sim.freqs
        if self.freq < freqs[0] or self.freq > freqs[-1]:
            self.value = None
            return
        
        if ch1_mode != 'none':
            if ch1_mode == 's11':
                self.value = np.interp(self.freq, freqs, sim.s11_db)
                self.unit = "dB"
                self.trace = "S11"
            else:
                self.value = np.interp(self.freq, freqs, sim.s22_db)
                self.unit = "dB"
                self.trace = "S22"
        else:
            mode_config = PLOT_MODES.get(secondary_plot, PLOT_MODES['loss'])
            if secondary_plot.startswith('phase'):
                attr = mode_config.get('sim_data_key', 'phase_rel')
                self.value = np.interp(self.freq, freqs, getattr(sim, attr, sim.phase_rel))
            elif secondary_plot == 'gd':
                self.value = np.interp(self.freq, freqs, sim.gd_ns)
            else:
                attr = mode_config.get('sim_data_key', 's21_db')
                self.value = np.interp(self.freq, freqs, getattr(sim, attr, sim.s21_db))
            self.unit = mode_config['unit']
            self.trace = f"S21 ({secondary_plot})"


# ===================== MODEL =====================
class CATState:
    def __init__(self):
        self.n = 5
        self.f0 = 500e6
        self.bw_design = 50e6
        self.fbw_design = self.bw_design / self.f0
        self.qu = 5000.0  # Unloaded Q

        self.Qe_source = None
        self.Qe_load = None

        self.start_freq = 350e6
        self.stop_freq = 650e6
        self.num_points = 801
        self.ripple_dB = 0.01

        self.rl_ymin = -80.0
        self.rl_ymax = 5.0
        self.il_ymin = -80.0
        self.il_ymax = 5.0
        self.phase_ymin = -180.0
        self.phase_ymax = 180.0
        self.gd_ymin = None
        self.gd_ymax = None

        self.show_meas = False
        self.secondary_plot = 'loss'
        self.ch1_mode = 's11'

        self.phase_flatten = False
        self.phase_delay = 0.0
        self.phase_constant = 0.0

        self.DOCS = os.path.join(os.path.expanduser('~/Documents'), 'S2P')
        os.makedirs(self.DOCS, exist_ok=True)

        self.loaded_freqs = None
        self.loaded_s11 = None
        self.loaded_s21 = None
        self.loaded_s22 = None
        self.loaded_s2p_path = None

        self.golden_freqs = None
        self.golden_s21 = None

        self.g = None
        self.M_golden = None
        self.M_current = None

        self.edit_mode = 'design'
        self.markers = None


# Plot Modes
PLOT_MODES = {
    'loss': {'name': 'Insertion Loss', 'ylabel': 'Insertion Loss (dB)', 'color_sim': '#00ff88', 'color_meas': 'orange', 'unit': 'dB', 'sim_data_key': 's21_db', 'ylim': lambda s: (s.il_ymin, s.il_ymax), 'meas_handler': lambda i,f,g,st: i.get('s21db')(f) if i.get('s21db') else None},
    'phase': {'name': 'Relative Phase', 'ylabel': 'Relative Phase (deg)', 'color_sim': '#ffaa00', 'color_meas': 'magenta', 'unit': '°', 'sim_data_key': 'phase_rel', 'ylim': lambda s: (s.phase_ymin, s.phase_ymax), 'meas_handler': lambda i,f,g,st: None},  # Handled explicitly in prepare_measurement_data
    'gd': {'name': 'Group Delay', 'ylabel': 'Group Delay (ns)', 'color_sim': '#00ffff', 'color_meas': 'pink', 'unit': 'ns', 'sim_data_key': 'gd_ns', 'ylim': lambda s: (s.gd_ymin, s.gd_ymax) if s.gd_ymin is not None else (None, None), 'meas_handler': lambda i,f,g,st: i.get('gd')(f) if i.get('gd') else None},
    'delta': {'name': 'Delta Loss', 'ylabel': 'Δ Loss (dB)', 'color_sim': '#ff00ff', 'color_meas': 'lime', 'unit': 'dB', 'sim_data_key': 's21_db', 'ylim': lambda s: (-5, 5), 'meas_handler': lambda i,f,g,st: (i.get('s21db')(f) - linear_interp1d(st.golden_freqs, 20 * np.log10(np.abs(st.golden_s21) + 1e-20))(f)) if g and i.get('s21db') else None}
}

DEFAULT_MODE = 'loss'


# ===================== SERVICES =====================
class SignalProcessor:
    @staticmethod
    def unwrap_phase(s): return np.unwrap(np.angle(s))
    @staticmethod
    def compute_group_delay(freqs, phase_unw):
        gd = np.zeros_like(freqs)
        if len(freqs) > 1:
            gd[1:] = -np.diff(phase_unw) / np.diff(2 * np.pi * freqs)
            gd[0] = gd[1]
        return gd
    @staticmethod
    def estimate_linear_delay(freqs, phase_unw):
        p = np.polyfit(freqs, phase_unw, 1)
        return -p[0] / (2 * np.pi)
    @staticmethod
    def remove_linear_delay(freqs, s, delay):
        phase_corr = np.unwrap(np.angle(s)) + 2 * np.pi * freqs * delay
        return np.abs(s) * np.exp(1j * phase_corr)
    @staticmethod
    def flatten_phase(freqs, phase_deg, delay, constant_offset=0.0):
        return phase_deg + 360 * freqs * delay + constant_offset
    @staticmethod
    def deembed_s11(freqs, s11):
        phi = SignalProcessor.unwrap_phase(s11)
        delay = phi[0] / (2 * np.pi) if len(phi) > 0 else 0.0
        phi_corr = phi - 2 * np.pi * freqs * delay
        return np.abs(s11) * np.exp(1j * phi_corr)
    @staticmethod
    def deembed_s21(freqs, s21):
        phi = SignalProcessor.unwrap_phase(s21)
        delay = SignalProcessor.estimate_linear_delay(freqs, phi)
        return SignalProcessor.remove_linear_delay(freqs, s21, delay)
    @staticmethod
    def compute_phase_and_gd(freqs, s21):
        phase_unw = SignalProcessor.unwrap_phase(s21)
        phase_deg = phase_unw / np.pi * 180
        gd_sec = SignalProcessor.compute_group_delay(freqs, phase_unw)
        return phase_deg, gd_sec * 1e9


class MeasurementService:
    def __init__(self, processor: SignalProcessor):
        self.processor = processor

    def prepare_measurement_data(self, state, sim_freqs):
        if state.loaded_freqs is None or not state.show_meas:
            return None
        
        interps = {}
        if state.loaded_s11 is not None:
            loaded_s11db_raw = 20 * np.log10(np.abs(state.loaded_s11) + 1e-20)
            interps['s11db'] = linear_interp1d(state.loaded_freqs, loaded_s11db_raw)
        if state.loaded_s22 is not None:
            loaded_s22db_raw = 20 * np.log10(np.abs(state.loaded_s22) + 1e-20)
            interps['s22db'] = linear_interp1d(state.loaded_freqs, loaded_s22db_raw)
        
        loaded_s21db_raw = 20 * np.log10(np.abs(state.loaded_s21) + 1e-20)
        interps['s21db'] = linear_interp1d(state.loaded_freqs, loaded_s21db_raw)
        
        # Keep raw unwrapped phase (no deembed). Correction applied later using sim delay/constant.
        loaded_phase_unw = self.processor.unwrap_phase(state.loaded_s21)
        interps['phase_unw'] = linear_interp1d(state.loaded_freqs, loaded_phase_unw)
        
        _, loaded_gd_ns = self.processor.compute_phase_and_gd(state.loaded_freqs, state.loaded_s21)
        interps['gd'] = linear_interp1d(state.loaded_freqs, loaded_gd_ns)
        
        loaded_return_db = None
        if state.ch1_mode == 's11' and interps.get('s11db'):
            loaded_return_db = interps['s11db'](sim_freqs)
        elif state.ch1_mode == 's22' and interps.get('s22db'):
            loaded_return_db = interps['s22db'](sim_freqs)
        
        mode_config = PLOT_MODES.get(state.secondary_plot, PLOT_MODES[DEFAULT_MODE])
        if state.secondary_plot == 'phase':
            # Apply sim delay/constant to raw measurement phase (no prior deembed)
            loaded_phase_deg = interps['phase_unw'](sim_freqs) / np.pi * 180
            loaded_relative = loaded_phase_deg + 360 * sim_freqs * state.phase_delay + state.phase_constant
            loaded_s21_plot = loaded_relative
        else:
            loaded_s21_plot = mode_config['meas_handler'](interps, sim_freqs, 
                                                         state.golden_s21 is not None and state.golden_freqs is not None, state)
        
        return MeasurementData(loaded_s21_plot=loaded_s21_plot, loaded_return_db=loaded_return_db, interps=interps)


class MatrixService:
    def __init__(self, state):
        self.state = state

    def reset_matrix(self):
        self.state.M_current = self.state.M_golden.copy()

    def randomize_matrix(self):
        self.state.M_current = self.state.M_golden.copy()
        for i in range(1, self.state.n):
            self.state.M_current[i, i + 1] += random.uniform(-0.08, 0.08)
        for i in range(1, self.state.n + 1):
            self.state.M_current[i, i] = random.uniform(-0.05, 0.05)


class FileService:
    def __init__(self, state, measurement_service):
        self.state = state
        self.measurement_service = measurement_service

    def export_s2p(self):
        name = dialogs.input_alert('Export As', 'Filename:')
        if not name: return
        if not name.endswith('.s2p'): name += '.s2p'
        path = os.path.join(self.state.DOCS, name)
        try:
            freqs = np.linspace(self.state.start_freq, self.state.stop_freq, self.state.num_points)
            s11, s21, s12, s22 = compute_s(freqs, self.state.M_current, self.state)
            write_s2p(path, freqs, s11, s21, s12=s12, s22=s22)
            dialogs.alert(f'Exported {name}')
        except Exception as e:
            dialogs.alert('Export Error', str(e))

    def load_s2p(self):
        files = sorted(f for f in os.listdir(self.state.DOCS) if f.lower().endswith('.s2p'))
        if not files:
            dialogs.alert('No Files', 'No .s2p files found in S2P folder.')
            return

        choice = dialogs.list_dialog('Load s2p', files)
        if not choice:
            return

        path = os.path.join(self.state.DOCS, choice)
        self.state.loaded_s2p_path = path

        (self.state.loaded_freqs, self.state.loaded_s11,
         self.state.loaded_s21, self.state.loaded_s22) = read_s2p(path)

        if self.state.loaded_freqs is None:
            dialogs.alert('Parse Error', 'Could not read .s2p file.')
            return

        # Do NOT deembed S21 — keep raw phase. Apply sim delay/constant during plotting only.
        if self.state.loaded_s11 is not None:
            self.state.loaded_s11 = self.measurement_service.processor.deembed_s11(
                self.state.loaded_freqs, self.state.loaded_s11)
        if self.state.loaded_s22 is not None:
            self.state.loaded_s22 = self.measurement_service.processor.deembed_s11(
                self.state.loaded_freqs, self.state.loaded_s22)

        self.state.show_meas = True

    def clear_meas(self):
        self.state.loaded_freqs = self.state.loaded_s11 = self.state.loaded_s21 = self.state.loaded_s22 = None
        self.state.show_meas = False
        self.state.loaded_s2p_path = None

    def delete_file(self):
        files = [f for f in os.listdir(self.state.DOCS) if f.endswith(('.npz', '.s2p'))]
        if not files:
            dialogs.alert('No Files', 'No files found.')
            return
        choice = dialogs.list_dialog('Delete File', files)
        if not choice: return
        if dialogs.alert('Confirm Delete', f'Delete {choice}?', button1='Yes', button2='No') == 2: return
        path = os.path.join(self.state.DOCS, choice)
        try:
            os.remove(path)
            dialogs.hud_alert(f'Deleted {choice}')
            if self.state.loaded_s2p_path and os.path.basename(self.state.loaded_s2p_path) == choice:
                self.clear_meas()
        except Exception as e:
            dialogs.alert('Delete Error', str(e))


class MarkerService:
    def __init__(self, state):
        self.state = state

    def update_markers(self, sim: SimulationResult, meas_data=None):
        for marker in self.state.markers:
            if marker.enabled and marker.freq is not None:
                marker.update_value(sim, self.state.secondary_plot, self.state.ch1_mode, meas_data, self.state)

    def set_marker(self, index):
        freq_str = dialogs.input_alert('Set Marker', f'Enter frequency for Marker {index+1} (MHz):', '')
        if freq_str:
            try:
                freq = float(freq_str) * 1e6
                self.state.markers[index].freq = freq
                self.state.markers[index].enabled = True
            except ValueError:
                dialogs.alert('Error', 'Invalid frequency.')

    def clear_markers(self):
        for marker in self.state.markers:
            marker.freq = None
            marker.enabled = False
            marker.value = None


# ===================== CONTROLLER =====================
class CATController:
    def __init__(self, state, view):
        self.state = state
        self.view = view
        self.processor = SignalProcessor()
        
        self.measurement_service = MeasurementService(self.processor)
        self.matrix_service = MatrixService(state)
        self.file_service = FileService(state, self.measurement_service)
        self.marker_service = MarkerService(state)

        self.sim_engine = SimulationEngine(self.processor)

    def compute_simulation(self) -> SimulationResult:
        return self.sim_engine.simulate(self.state)

    def prepare_measurement_data(self, sim_freqs):
        return self.measurement_service.prepare_measurement_data(self.state, sim_freqs)

    def update_markers(self, sim: SimulationResult, meas_data=None):
        self.marker_service.update_markers(sim, meas_data)

    def matrix_changed(self):
        self.view.redraw(immediate=True)

    def redraw(self):
        self.view.redraw()

    def reset_matrix(self):
        self.matrix_service.reset_matrix()

    def randomize_matrix(self):
        self.matrix_service.randomize_matrix()

    def export_s2p(self):
        self.file_service.export_s2p()
        self.redraw()

    def load_s2p(self):
        self.file_service.load_s2p()
        self.redraw()

    def clear_meas(self):
        self.file_service.clear_meas()
        self.redraw()

    def toggle_meas(self):
        if self.state.loaded_freqs is not None:
            self.state.show_meas = not self.state.show_meas
            self.redraw()

    def delete_file(self):
        self.file_service.delete_file()
        self.redraw()

    @ui.in_background
    def update_phase_flatten(self):
        try:
            freqs = np.linspace(self.state.start_freq, self.state.stop_freq, self.state.num_points)
            _, s21, _, _ = compute_s(freqs, self.state.M_current, self.state)
            phase_unw = self.processor.unwrap_phase(s21)
            phase_deg = phase_unw / np.pi * 180
            idx1 = np.argmin(np.abs(freqs - (self.state.f0 - self.state.bw_design / 2)))
            idx2 = np.argmin(np.abs(freqs - (self.state.f0 + self.state.bw_design / 2)))
            p1 = phase_deg[idx1]
            p2 = phase_deg[idx2]
            f1 = freqs[idx1]
            f2 = freqs[idx2]
            slope = (p2 - p1) / (f2 - f1)
            self.state.phase_delay = -slope / 360.0
            corrected_p1 = p1 + 360 * f1 * self.state.phase_delay
            self.state.phase_constant = -corrected_p1
            self.state.phase_flatten = True
            self.redraw()
        except Exception as e:
            @on_main_thread
            def alert(): dialogs.alert('Phase Flatten Error', str(e))
            alert()

    def edit_parameters(self):
        state = self.state
        fields = [
            {'title': 'Number of resonators (n)', 'type': 'number', 'value': str(state.n)},
            {'title': 'Center frequency (MHz)', 'type': 'number', 'value': f'{state.f0/1e6:.1f}'},
            {'title': 'Design bandwidth (MHz)', 'type': 'number', 'value': f'{state.bw_design/1e6:.1f}'},
            {'title': 'Unloaded Q (Qu)', 'type': 'number', 'value': f'{state.qu:.0f}'},
            {'title': 'Number of points', 'type': 'number', 'value': str(state.num_points)},
            {'title': 'Passband ripple (dB)', 'type': 'number', 'value': f'{state.ripple_dB:.3f}'},
        ]
        res = dialogs.form_dialog('Edit Filter Parameters', fields)
        if not res: return
        try:
            new_n = int(float(res['Number of resonators (n)']))
            if new_n < 2: new_n = 2
            new_f0 = float(res['Center frequency (MHz)']) * 1e6
            new_bw = float(res['Design bandwidth (MHz)']) * 1e6
            new_qu = float(res['Unloaded Q (Qu)'])
            new_ripple = float(res['Passband ripple (dB)'])
            new_num_points = int(float(res['Number of points']))
            if new_num_points < 100: new_num_points = 801
        except Exception as e:
            dialogs.alert('Input Error', str(e))
            return
        old_n = state.n
        order_changed = old_n != new_n
        state.n = new_n
        state.f0 = new_f0
        state.bw_design = new_bw
        state.fbw_design = state.bw_design / state.f0
        state.qu = max(100.0, new_qu)
        state.num_points = new_num_points
        state.ripple_dB = new_ripple
        state.g = chebyshev_g_values(state.n, state.ripple_dB)
        state.Qe_source = state.g[0] * state.g[1] / state.fbw_design
        state.Qe_load = state.g[state.n] * state.g[state.n + 1] / state.fbw_design
        state.M_golden = build_matrix(state)
        state.M_current = state.M_golden.copy()
        if order_changed:
            dialogs.alert('Order Changed', f'Filter order changed from {old_n} to {new_n}.\nMatrix has been reset.')
        self.redraw()

    def set_limits(self):
        state = self.state
        fields = [
            {'title': 'Start Frequency (MHz)', 'type': 'number', 'value': f'{state.start_freq/1e6:.0f}'},
            {'title': 'Stop Frequency (MHz)', 'type': 'number', 'value': f'{state.stop_freq/1e6:.0f}'},
            {'title': 'Return Loss y min (dB)', 'type': 'number', 'value': str(state.rl_ymin)},
            {'title': 'Return Loss y max (dB)', 'type': 'number', 'value': str(state.rl_ymax)},
            {'title': 'Through Loss y min (dB)', 'type': 'number', 'value': str(state.il_ymin)},
            {'title': 'Through Loss y max (dB)', 'type': 'number', 'value': str(state.il_ymax)},
            {'title': 'Phase y min (deg)', 'type': 'number', 'value': str(state.phase_ymin)},
            {'title': 'Phase y max (deg)', 'type': 'number', 'value': str(state.phase_ymax)},
            {'title': 'Group Delay y min (ns)', 'type': 'number', 'value': str(state.gd_ymin) if state.gd_ymin is not None else ''},
            {'title': 'Group Delay y max (ns)', 'type': 'number', 'value': str(state.gd_ymax) if state.gd_ymax is not None else ''},
        ]
        res = dialogs.form_dialog('Plot Limits', fields)
        if res:
            try:
                state.start_freq = float(res['Start Frequency (MHz)']) * 1e6
                state.stop_freq = float(res['Stop Frequency (MHz)']) * 1e6
                state.rl_ymin = float(res['Return Loss y min (dB)'])
                state.rl_ymax = float(res['Return Loss y max (dB)'])
                state.il_ymin = float(res['Through Loss y min (dB)'])
                state.il_ymax = float(res['Through Loss y max (dB)'])
                state.phase_ymin = float(res['Phase y min (deg)'])
                state.phase_ymax = float(res['Phase y max (deg)'])
                gd_min_str = res['Group Delay y min (ns)']
                state.gd_ymin = float(gd_min_str) if gd_min_str.strip() else None
                gd_max_str = res['Group Delay y max (ns)']
                state.gd_ymax = float(gd_max_str) if gd_max_str.strip() else None
                self.redraw()
            except ValueError as e:
                dialogs.alert('Error', str(e))

    def reset_limits(self):
        state = self.state
        state.start_freq = 350e6
        state.stop_freq = 650e6
        state.rl_ymin = -80.0
        state.rl_ymax = 5.0
        state.il_ymin = -80.0
        state.il_ymax = 5.0
        state.phase_ymin = -180.0
        state.phase_ymax = 180.0
        state.gd_ymin = state.gd_ymax = None
        self.redraw()

    def reset(self):
        if dialogs.alert('Confirm', 'This will reset all changes. Continue?', button1='Yes', button2='No') == 2: return
        state = self.state
        state.phase_flatten = False
        state.phase_delay = state.phase_constant = 0.0
        self.reset_matrix()
        self.clear_markers()
        self.redraw()

    def randomize(self):
        if dialogs.alert('Confirm', 'This will randomize changes. Continue?', button1='Yes', button2='No') == 2: return
        self.randomize_matrix()
        self.redraw()

    def toggle_ch1(self):
        state = self.state
        modes = ['s11', 's22', 'none']
        state.ch1_mode = modes[(modes.index(state.ch1_mode) + 1) % len(modes)]
        self.redraw()

    def toggle_ch2(self):
        state = self.state
        all_modes = ['loss', 'phase', 'gd', 'delta', 'none']
        current = state.secondary_plot
        idx = all_modes.index(current) if current in all_modes else 0
        state.secondary_plot = all_modes[(idx + 1) % len(all_modes)]
        self.redraw()

    def set_marker(self, index):
        self.marker_service.set_marker(index)
        self.redraw()

    def clear_markers(self):
        self.marker_service.clear_markers()
        self.redraw()


# ===================== PLOT RENDERING LAYER =====================
def get_plot_config(mode):
    return PLOT_MODES.get(mode, PLOT_MODES[DEFAULT_MODE])


class PlotRenderer:
    def create_figure(self):
        plt.close('all')
        fig = plt.figure(figsize=(10, 7.5), facecolor='#1a1a1a')
        ax1 = plt.subplot(1, 1, 1)
        return fig, ax1

    def save(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, dpi=140, facecolor='#1a1a1a')
        return buf.getvalue()

    def cleanup(self):
        plt.clf()
        plt.close('all')


class TraceRenderer:
    @staticmethod
    def draw_return_loss(ax, freq_mhz, sim_db, meas_db=None):
        ax.plot(freq_mhz, sim_db, '#00ddff', lw=2.2, label='S11 Sim')
        if meas_db is not None:
            ax.plot(freq_mhz, meas_db, 'black', lw=1.5, label='S11 Meas')

    @staticmethod
    def draw_s21(ax, freq_mhz, data, color, label):
        ax.plot(freq_mhz, data, color, lw=2.2, label=label)


class AxisRenderer:
    @staticmethod
    def setup(ax, state, ch1_mode):
        ax.set_xlabel('Frequency (MHz)')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)

        if ch1_mode != 'none':
            ax.set_ylabel('Return Loss (dB)', color='#00ddff')
            ax.tick_params(axis='y', colors='#00ddff')
            ax.set_ylim(state.rl_ymin, state.rl_ymax)


class MarkerRenderer:
    @staticmethod
    def draw(ax, sim, state):
        marker_info = []
        for i, marker in enumerate(state.markers):
            if not marker.enabled or marker.freq is None or marker.value is None:
                continue
            if marker.freq < sim.freqs[0] or marker.freq > sim.freqs[-1]:
                continue
            x = marker.freq / 1e6
            y = marker.value
            ax.axvline(x, color=marker.color, linestyle='--', linewidth=1.0, alpha=0.7)
            ax.plot(x, y, marker='o', color=marker.color, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
            marker_info.append((i+1, x, y, marker.unit, marker.trace))
        return marker_info

    @staticmethod
    def draw_table(ax, marker_info):
        if not marker_info: return
        text_lines = ["MARKERS"]
        for mnum, freq, val, unit, trace in marker_info:
            text_lines.append(f"M{mnum}: {freq:.3f} MHz")
            text_lines.append(f"   {trace} = {val:.3f}{unit}")
        marker_text = "\n".join(text_lines)
        ax.text(0.98, 0.98, marker_text, transform=ax.transAxes, color='white', fontsize=9, fontfamily='monospace',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#0d1117', edgecolor='#555555', alpha=0.95))


class LegendRenderer:
    @staticmethod
    def draw(ax1, ax2=None):
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ([], []) if ax2 is None else ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(h1 + h2, l1 + l2, loc='upper left')


# ===================== VIEW =====================
class MatrixEditor:
    def __init__(self, controller):
        self.controller = controller

    def show(self, sender=None):
        state = self.controller.state
        M_edit = state.M_current
        fields = []
        for lbl in get_labels(state):
            if lbl == 'M_S1':
                val = M_edit[0, 1]
            elif lbl == f'M{state.n}L':
                val = M_edit[state.n, state.n + 1]
            elif lbl.startswith('M'):
                i, j = int(lbl[1]), int(lbl[2])
                val = M_edit[i, j]
            else:
                i = int(lbl[1:])
                val = M_edit[i, i]
            fields.append({'title': lbl, 'type': 'number', 'value': repr(val)})
        res = dialogs.form_dialog('Edit Matrix', fields)
        if not res: return
        changed = False
        for lbl in get_labels(state):
            if lbl == 'M_S1': i,j = 0,1
            elif lbl == f'M{state.n}L': i,j = state.n, state.n+1
            elif lbl.startswith('M'):
                i, j = int(lbl[1]), int(lbl[2])
            else:
                i = j = int(lbl[1:])
            try:
                new = float(res[lbl])
                if abs(new - M_edit[i, j]) > 1e-6:
                    changed = True
                    M_edit[i, j] = M_edit[j, i] = new
            except: pass
        if changed:
            state.M_golden = state.M_current.copy()
            self.controller.matrix_changed()


class CAT(ui.View):
    def __init__(self):
        self.background_color = 'black'
        self.flex = 'WH'

        self.state = CATState()
        self.controller = CATController(self.state, self)

        self.img = ui.ImageView()
        self.img.flex = 'WH'
        self.img.content_mode = ui.CONTENT_SCALE_ASPECT_FIT
        self.img.background_color = 'black'
        self.add_subview(self.img)

        self.editor = MatrixEditor(self.controller)

        self.state.markers = [Marker(color='red') for i in range(3)]

        self.state.g = chebyshev_g_values(self.state.n, self.state.ripple_dB)
        self.state.Qe_source = self.state.g[0] * self.state.g[1] / self.state.fbw_design
        self.state.Qe_load = self.state.g[self.state.n] * self.state.g[self.state.n + 1] / self.state.fbw_design
        self.state.M_golden = build_matrix(self.state)
        self.state.M_current = self.state.M_golden.copy()

        main_defs = [
            ('Edit', self.editor.show),
            ('Reset', lambda s: self.controller.reset()),
            ('Random', lambda s: self.controller.randomize()),
            ('Design', lambda s: self.controller.edit_parameters()),
            ('Scales', lambda s: self.controller.set_limits()),
            ('Reset Scales', lambda s: self.controller.reset_limits()),
            ('Ch1', lambda s: self.controller.toggle_ch1()),
            ('Ch2', lambda s: self.controller.toggle_ch2()),
            ('s2p', self.enter_s2p_menu),
            ('Markers', self.enter_markers_menu),
        ]

        s2p_defs = [
            ('Export s2p', lambda s: self.controller.export_s2p()),
            ('Import s2p', lambda s: self.controller.load_s2p()),
            ('Toggle s2p', lambda s: self.controller.toggle_meas()),
            ('Relative Phase', lambda s: self.controller.update_phase_flatten()),
            ('Clear s2p', lambda s: self.controller.clear_meas()),
            ('Delete', lambda s: self.controller.delete_file()),
            ('Return', self.return_to_main),
        ]

        markers_defs = [
            ('Marker 1', lambda s: self.controller.set_marker(0)),
            ('Marker 2', lambda s: self.controller.set_marker(1)),
            ('Marker 3', lambda s: self.controller.set_marker(2)),
            ('Clear Markers', lambda s: self.controller.clear_markers()),
            ('Return', self.return_to_main),
        ]

        self.buttons = []
        self.main_buttons = []
        self.s2p_buttons = []
        self.markers_buttons = []

        for title, action in main_defs:
            b = self.create_button(title, action)
            self.buttons.append(b)
            self.main_buttons.append(b)
            self.add_subview(b)

        for title, action in s2p_defs:
            b = self.create_button(title, action)
            b.hidden = True
            self.buttons.append(b)
            self.s2p_buttons.append(b)
            self.add_subview(b)

        for title, action in markers_defs:
            b = self.create_button(title, action)
            b.hidden = True
            self.buttons.append(b)
            self.markers_buttons.append(b)
            self.add_subview(b)

        self.redraw_timer = None
        self.redraw()

    def create_button(self, title, action):
        b = ui.Button(title=title)
        b.tint_color = '#00ccff'
        b.border_width = 1
        b.border_color = '#444'
        b.corner_radius = 6
        b.action = action
        return b

    def layout(self):
        visible_buttons = [b for b in self.buttons if not b.hidden]
        num_buttons = len(visible_buttons)
        btn_width = 130
        gap = 8
        max_per_row = max(1, int((self.width - 20) / (btn_width + gap)))
        num_rows = (num_buttons + max_per_row - 1) // max_per_row
        buttons_height = 8 + num_rows * 44 + (num_rows - 1) * 4

        for i, b in enumerate(visible_buttons):
            row = i // max_per_row
            col = i % max_per_row
            x = 10 + col * (btn_width + gap)
            y = 8 + row * 44
            b.frame = (x, y, btn_width, 36)

        plot_y = buttons_height
        self.img.frame = (0, plot_y, self.width, self.height - plot_y)

    def redraw(self, sender=None, immediate=False):
        if self.redraw_timer:
            self.redraw_timer.cancel()
        if immediate:
            self._perform_redraw()
        else:
            self.redraw_timer = ui.delay(self._perform_redraw, 0.5)

    def _perform_redraw(self):
        try:
            sim = self.controller.compute_simulation()
            meas = self.controller.prepare_measurement_data(sim.freqs)
            self.controller.update_markers(sim, meas)

            with plot_lock:
                renderer = PlotRenderer()
                fig, ax1 = renderer.create_figure()
                ax2 = ax1.twinx()

                AxisRenderer.setup(ax1, self.state, self.state.ch1_mode)

                freq_mhz = sim.freqs / 1e6

                if self.state.ch1_mode != 'none':
                    ch1_db = sim.s11_db if self.state.ch1_mode == 's11' else sim.s22_db
                    TraceRenderer.draw_return_loss(ax1, freq_mhz, ch1_db, meas.loaded_return_db if meas else None)

                if self.state.secondary_plot != 'none':
                    mode_config = get_plot_config(self.state.secondary_plot)
                    
                    # Right axis for S21/phase/etc.
                    ax2.set_ylabel(mode_config['ylabel'], color=mode_config['color_sim'])
                    ax2.tick_params(axis='y', colors=mode_config['color_sim'])
                    if self.state.secondary_plot == 'loss':
                        ax2.set_ylim(self.state.il_ymin, self.state.il_ymax)
                    elif self.state.secondary_plot == 'phase':
                        ax2.set_ylim(self.state.phase_ymin, self.state.phase_ymax)
                    elif self.state.secondary_plot == 'gd':
                        if self.state.gd_ymin is not None:
                            ax2.set_ylim(self.state.gd_ymin, self.state.gd_ymax)
                    elif self.state.secondary_plot == 'delta':
                        ax2.set_ylim(-5, 5)

                    sim_data = sim.phase_rel if self.state.secondary_plot == 'phase' else getattr(sim, mode_config.get('sim_data_key', 's21_db'), sim.s21_db)
                    meas_data = meas.loaded_s21_plot if meas else None

                    TraceRenderer.draw_s21(ax2, freq_mhz, sim_data, mode_config['color_sim'], f'Sim ({mode_config["name"]})')
                    if meas_data is not None:
                        TraceRenderer.draw_s21(ax2, freq_mhz, meas_data, mode_config['color_meas'], f'Meas ({mode_config["name"]})')

                # Markers on the active axis
                marker_axis = ax1 if self.state.ch1_mode != 'none' else ax2
                marker_info = MarkerRenderer.draw(marker_axis, sim, self.state)
                MarkerRenderer.draw_table(ax1, marker_info)  # Table always on left

                LegendRenderer.draw(ax1, ax2)

                data = renderer.save(fig)
                renderer.cleanup()

                @on_main_thread
                def set_img():
                    self.img.image = ui.Image.from_data(data)
                set_img()

        except Exception as e:
            @on_main_thread
            def alert():
                dialogs.alert('Plot Render Error', str(e))
            alert()

    def enter_s2p_menu(self, sender):
        for b in self.main_buttons: b.hidden = True
        for b in self.markers_buttons: b.hidden = True
        for b in self.s2p_buttons: b.hidden = False
        self.layout()

    def enter_markers_menu(self, sender):
        for b in self.main_buttons: b.hidden = True
        for b in self.s2p_buttons: b.hidden = True
        for b in self.markers_buttons: b.hidden = False
        self.layout()

    def return_to_main(self, sender):
        for b in self.s2p_buttons: b.hidden = True
        for b in self.markers_buttons: b.hidden = True
        for b in self.main_buttons: b.hidden = False
        self.layout()


# ===================== SIMULATION & CORE =====================
class SimulationEngine:
    def __init__(self, processor: SignalProcessor):
        self.processor = processor

    def simulate(self, state) -> SimulationResult:
        freqs = np.linspace(state.start_freq, state.stop_freq, state.num_points)
        s11, s21, s12, s22 = compute_s(freqs, state.M_current, state)
        
        s21_db = 20 * np.log10(np.abs(s21) + 1e-20)
        s11_db = 20 * np.log10(np.abs(s11) + 1e-20)
        s22_db = 20 * np.log10(np.abs(s22) + 1e-20)
        
        phase_deg_raw, gd_ns = self.processor.compute_phase_and_gd(freqs, s21)
        
        phase_rel = phase_deg_raw
        if state.phase_flatten:
            phase_rel = SignalProcessor.flatten_phase(freqs, phase_deg_raw, state.phase_delay, state.phase_constant)
        
        return SimulationResult(
            freqs=freqs, s11=s11, s21=s21, s22=s22,
            s11_db=s11_db, s21_db=s21_db, s22_db=s22_db,
            phase_deg=phase_deg_raw, gd_ns=gd_ns,
            phase_rel=phase_rel
        )


def chebyshev_g_values(n, ripple_db):
    eps = np.sqrt(10**(ripple_db / 10) - 1)
    beta = np.arcsinh(1 / eps) / n
    s = np.sinh(beta)
    g = [1.0]
    a = [None] + [np.sin((2 * i - 1) * np.pi / (2 * n)) for i in range(1, n + 1)]
    b = [None] + [s * s + np.sin(i * np.pi / n)**2 for i in range(1, n + 1)]
    g.append(2 * a[1] / s)
    for i in range(2, n + 1):
        g.append((4 * a[i - 1] * a[i]) / (b[i - 1] * g[i - 1]))
    g.append(1.0)
    return g


def build_matrix(state):
    M = np.zeros((state.n + 2, state.n + 2))
    if hasattr(state, 'Qe_source') and state.Qe_source is not None:
        m_s1 = 1.0 / np.sqrt(state.Qe_source * state.fbw_design)
        M[0, 1] = M[1, 0] = m_s1
    if hasattr(state, 'Qe_load') and state.Qe_load is not None:
        m_nl = 1.0 / np.sqrt(state.Qe_load * state.fbw_design)
        M[state.n, state.n + 1] = M[state.n + 1, state.n] = m_nl
    for i in range(1, state.n):
        M[i, i + 1] = M[i + 1, i] = 1 / np.sqrt(state.g[i] * state.g[i + 1])
    return M


def build_labels(n):
    labels = ['M_S1']
    for i in range(1, n):
        labels += [f'R{i}', f'M{i}{i+1}']
    labels += [f'R{n}', f'M{n}L']
    return labels


def get_labels(state):
    return build_labels(state.n)


def linear_interp1d(x, y, bounds_error=False, fill_value='extrapolate'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        def interpolator(x_new): return np.full_like(x_new, np.nan, dtype=float)
        return interpolator
    idx = np.argsort(x)
    x_s = x[idx]
    y_s = y[idx]
    def interpolator(x_new):
        x_new = np.asarray(x_new)
        result = np.interp(x_new, x_s, y_s, left=np.nan, right=np.nan)
        if fill_value == 'extrapolate':
            left_mask = x_new < x_s[0]
            right_mask = x_new > x_s[-1]
            if np.any(left_mask):
                slope = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0]) if x_s[1] != x_s[0] else 0.0
                result[left_mask] = y_s[0] + slope * (x_new[left_mask] - x_s[0])
            if np.any(right_mask):
                slope = (y_s[-1] - y_s[-2]) / (x_s[-1] - x_s[-2]) if x_s[-1] != x_s[-2] else 0.0
                result[right_mask] = y_s[-1] + slope * (x_new[right_mask] - x_s[-1])
        return result
    return interpolator


def build_system_matrix(state, M, Omega):
    n = state.n
    N = n + 2
    nf = len(Omega)

    A = np.zeros((nf, N, N), dtype=complex)

    U = np.zeros((N, N))
    U[1:n+1, 1:n+1] = np.eye(n)

    Mmat = M.copy()

    R = np.zeros((N, N))
    r = 1.0 / (state.qu * state.fbw_design)
    for k in range(1, n + 1):
        R[k, k] = r

    G = np.zeros((N, N))
    G[0, 0] = 1.0
    G[N-1, N-1] = 1.0

    for i, om in enumerate(Omega):
        A[i] = G + R + 1j * (om * U - Mmat)

    return A


def solve_excitation(A, port):
    nf, N, _ = A.shape
    b = np.zeros((nf, N), dtype=complex)
    b[:, port] = 1.0
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros((nf, N), dtype=complex)


def compute_s(freqs, M, state):
    freqs = np.asarray(freqs, dtype=float)
    Omega = (freqs / state.f0 - state.f0 / freqs) / state.fbw_design
    
    A = build_system_matrix(state, M, Omega)

    x1 = solve_excitation(A, 0)
    s11 = 1 - 2 * x1[:, 0]
    s21 = -2 * x1[:, -1]

    x2 = solve_excitation(A, -1)
    s22 = 1 - 2 * x2[:, -1]
    s12 = -2 * x2[:, 0]

    return s11, s21, s12, s22


def write_s2p(path, freqs, s11, s21, s12=None, s22=None, z0=50.0):
    freqs = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11, dtype=complex)
    s21 = np.asarray(s21, dtype=complex)
    if s12 is None: s12 = s21
    if s22 is None: s22 = s11
    s12 = np.asarray(s12, dtype=complex)
    s22 = np.asarray(s22, dtype=complex)
    try:
        with open(path, 'w') as f:
            f.write("! Exported by Pythonista CAT Tool\n")
            f.write(f"# Hz S RI R {z0:.6g}\n")
            for fr, a11, a21, a12, a22 in zip(freqs, s11, s21, s12, s22):
                f.write(f"{fr:.0f} {a11.real:.6e} {a11.imag:.6e} {a21.real:.6e} {a21.imag:.6e} {a12.real:.6e} {a12.imag:.6e} {a22.real:.6e} {a22.imag:.6e}\n")
    except Exception as e:
        @on_main_thread
        def alert(): dialogs.alert('Error', f'Failed to write .s2p: {str(e)}')
        alert()


def read_s2p(path):
    freqs = []
    s11_list = []
    s21_list = []
    s12_list = []
    s22_list = []
    freq_unit = 'MHZ'
    data_format = 'RI'
    z0 = 50.0
    unit_mult = {'HZ':1.0, 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'): continue
                if line.startswith('#'):
                    parts = line[1:].upper().split()
                    if len(parts) >= 5:
                        freq_unit = parts[0]
                        data_format = parts[2]
                        z0 = float(parts[4])
                    continue
                parts = line.split()
                if len(parts) != 9: continue
                freq = float(parts[0]) * unit_mult.get(freq_unit, 1e6)
                freqs.append(freq)
                def to_complex(a, b):
                    a = float(a); b = float(b)
                    if data_format == 'RI': return complex(a, b)
                    if data_format == 'MA': return a * np.exp(1j * b * np.pi / 180)
                    if data_format == 'DB': return 10**(a/20) * np.exp(1j * b * np.pi / 180)
                s11_list.append(to_complex(parts[1], parts[2]))
                s21_list.append(to_complex(parts[3], parts[4]))
                s12_list.append(to_complex(parts[5], parts[6]))
                s22_list.append(to_complex(parts[7], parts[8]))
        if not freqs: return None, None, None, None
        idx = np.argsort(freqs)
        freqs = np.array(freqs)[idx]
        s11 = np.array(s11_list)[idx]
        s21 = np.array(s21_list)[idx]
        s12 = np.array(s12_list)[idx]
        s22 = np.array(s22_list)[idx]
        s21 = (s21 + s12) / 2.0
        return freqs, s11, s21, s22
    except Exception as e:
        print(f"Error reading s2p: {str(e)}")
        return None, None, None, None


if __name__ == '__main__':
    CAT().present('fullscreen', orientations=['landscape_left', 'landscape_right'])