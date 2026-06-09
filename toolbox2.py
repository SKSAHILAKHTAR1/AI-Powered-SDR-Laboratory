# ==============================================================================
#                 AI-POWERED SPEC-SDR COMMUNICATION LABORATORY
# ==============================================================================
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import hilbert, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtWidgets import (
    QApplication, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout,
    QPushButton, 
    QLabel, 
    QSlider, 
    QComboBox, 
    QTabWidget, 
    QFrame
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# ==============================================================================
# 1. SYSTEM PARAMETERS & CONFIGURATION
# ==============================================================================
FS = 44100          
N = 4096            
t = np.arange(N) / FS
message = np.zeros(N)

# Rolling phase variable to create continuous horizontal snake motion
phase_offset = 0.0  

# Spin up core sound infrastructure
stream = sd.OutputStream(
    samplerate=FS, 
    channels=1, 
    blocksize=1024
)
stream.start()

# ==============================================================================
# 2. MACHINE LEARNING DATA ASSEMBLY
# ==============================================================================
def initialize_ai_classifier():
    X_train, y_train = [], []
    for _ in range(50):
        s_wave = np.sin(2 * np.pi * 5 * t)
        sq_wave = np.sign(s_wave)
        n_wave = np.random.normal(0, 1, N)
        X_train.extend([
            [np.mean(s_wave), np.std(s_wave), np.max(s_wave)],
            [np.mean(sq_wave), np.std(sq_wave), np.max(sq_wave)],
            [np.mean(n_wave),  np.std(n_wave),  np.max(n_wave)]
        ])
        y_train.extend(["SINE", "SQUARE", "NOISE"])
        
    classifier = RandomForestClassifier(
        n_estimators=10, 
        random_state=42
    )
    classifier.fit(X_train, y_train)
    return classifier

clf = initialize_ai_classifier()

# ==============================================================================
# 3. ADVANCED SIGNAL MATHEMATICS & PROCESSING
# ==============================================================================
def normalize(sig):
    mx = np.max(np.abs(sig))
    if mx != 0:
        return sig / mx
    return sig

def calculate_power(sig):
    p_watts = float(np.mean(sig ** 2))
    if p_watts > 1e-12:
        p_dbm = 10 * np.log10(p_watts / 0.001)
    else:
        p_dbm = -100.0
    return p_watts, p_dbm

def lowpass_filter(sig, cutoff=4000):
    nyq = FS / 2
    b, a = butter(5, cutoff / nyq)
    return filtfilt(b, a, sig)

def generate_base_waveform(wave_type, freq, amp, phase):
    """Generates basic waves augmented with a rolling phase to create animation motion"""
    angle = 2 * np.pi * freq * t + phase
    if wave_type == "Sine":
        return amp * np.sin(angle)
    elif wave_type == "Square":
        return amp * np.sign(np.sin(angle))
    elif wave_type == "Triangle":
        # Phase-shifted normalized calculation for clean triangle animation
        return amp * (2 * np.abs(2 * ((t * freq + phase/(2*np.pi)) - np.floor((t * freq + phase/(2*np.pi)) + 0.5))) - 1)
    elif wave_type == "Sawtooth":
        return amp * (2 * ((t * freq + phase/(2*np.pi)) - np.floor((t * freq + phase/(2*np.pi)) + 0.5)))
    return None

# Modulators using updated global animation phase hooks
def am_modulation(msg, fc, ka, phase): 
    return (1 + ka * msg) * np.cos(2 * np.pi * fc * t + phase * (fc / 150.0))

def dsb_sc_modulation(msg, fc, phase): 
    return msg * np.cos(2 * np.pi * fc * t + phase * (fc / 150.0))

def ssb_sc_modulation(msg, fc, phase):
    h = np.imag(hilbert(msg))
    carrier_phase = phase * (fc / 150.0)
    return msg * np.cos(2 * np.pi * fc * t + carrier_phase) - h * np.sin(2 * np.pi * fc * t + carrier_phase)

def fm_modulation(msg, fc, phase, kf=600):
    phase_integral = np.cumsum(msg) / FS
    carrier_phase = phase * (fc / 150.0)
    return np.cos(2 * np.pi * fc * t + carrier_phase + 2 * np.pi * kf * phase_integral)

# Demodulators
def envelope_detector(sig):
    rec = lowpass_filter(np.abs(sig))
    return (rec - np.mean(rec)) / (np.max(np.abs(rec)) + 1e-6)

def coherent_detector(sig, fc, phase):
    carrier_phase = phase * (fc / 150.0)
    rec = lowpass_filter(sig * np.cos(2 * np.pi * fc * t + carrier_phase))
    return (rec - np.mean(rec)) / (np.max(np.abs(rec)) + 1e-6)

def fm_demodulation(sig):
    phase = np.unwrap(np.angle(hilbert(sig)))
    demod = np.append(np.diff(phase), 0)
    demod -= np.mean(demod)
    return demod / (np.max(np.abs(demod)) + 1e-6)

# ==============================================================================
# 4. SDR WORKSTATION APP INTERFACE
# ==============================================================================
class AISDR(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Powered SDR Laboratory")
        self.showMaximized()
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.build_header_widget()
        self.build_control_dock()
        self.build_ai_telemetry_panel()
        self.build_input_display_box()  
        self.build_display_tabs()
        self.build_hardware_hud()
        self.connect_system_events()

    def build_header_widget(self):
        header = QLabel("AI POWERED SDR COMMUNICATION SYSTEM")
        header.setStyleSheet("font-size: 22pt; color: cyan; padding: 5px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(header)

    def build_control_dock(self):
        controls = QHBoxLayout()
        
        self.wave_box = QComboBox()
        self.wave_box.addItems(["Mouse Draw", "Sine", "Square", "Triangle", "Sawtooth"])
        self.wave_box.setCurrentText("Sine")
        
        self.msg_freq_slider = QSlider(Qt.Horizontal)
        self.msg_freq_slider.setRange(10, 400)
        self.msg_freq_slider.setValue(150)

        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(500, 6000)
        self.freq_slider.setValue(2000)

        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(1, 200)
        self.amp_slider.setValue(90)

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 150)
        self.noise_slider.setValue(5)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["AM", "DSB-SC", "SSB-SC", "FM"])

        self.clear_btn = QPushButton("RESET CLEAR")

        controls.addWidget(QLabel("Wave Type:"))   
        controls.addWidget(self.wave_box)
        controls.addWidget(QLabel("Msg Freq:"))   
        controls.addWidget(self.msg_freq_slider)
        controls.addWidget(QLabel("Carrier:"))    
        controls.addWidget(self.freq_slider)
        controls.addWidget(QLabel("Mod Index:"))  
        controls.addWidget(self.amp_slider)
        controls.addWidget(QLabel("Noise level:"))
        controls.addWidget(self.noise_slider)
        controls.addWidget(QLabel("Scheme:"))     
        controls.addWidget(self.mode_box)
        controls.addWidget(self.clear_btn)
        
        self.main_layout.addLayout(controls)

    def build_ai_telemetry_panel(self):
        ai_panel = QFrame()
        ai_panel.setStyleSheet("border: 2px solid cyan; border-radius: 8px; padding: 6px; background-color: #0c1212;")
        ai_layout = QHBoxLayout()
        
        self.ai_label = QLabel("AI SIGNAL RECOGNITION: IDLE")
        self.ai_label.setStyleSheet("font-size: 14pt; color: yellow; border: none;")
        self.snr_label = QLabel("AI CHANNEL QUALITY: STANDBY")
        self.snr_label.setStyleSheet("font-size: 14pt; color: lime; border: none;")
        
        ai_layout.addWidget(self.ai_label)
        ai_layout.addWidget(self.snr_label)
        ai_panel.setLayout(ai_layout)
        self.main_layout.addWidget(ai_panel)

    def build_input_display_box(self):
        self.disp_panel = QFrame()
        self.disp_panel.setStyleSheet("""
            border: 2px solid #FF8C00; 
            border-radius: 8px; 
            padding: 8px; 
            background-color: #140d05;
        """)
        disp_layout = QHBoxLayout()

        self.lbl_input_shape = QLabel("PROFILE: SINE")
        self.lbl_input_freq = QLabel("FREQ: 0 Hz")
        self.lbl_input_vpp = QLabel("Vpp: 0.00V")
        self.lbl_input_vrms = QLabel("Vrms: 0.00V")
        self.lbl_input_p_w = QLabel("POWER: 0.000W")
        self.lbl_input_p_dbm = QLabel("POWER: 0.00 dBm")

        label_list = [
            self.lbl_input_shape, 
            self.lbl_input_freq, 
            self.lbl_input_vpp, 
            self.lbl_input_vrms, 
            self.lbl_input_p_w, 
            self.lbl_input_p_dbm
        ]
        
        for lbl in label_list:
            lbl.setStyleSheet("font-family: Consolas; font-size: 12pt; color: #FFA500; border: none; font-weight: bold;")
            disp_layout.addWidget(lbl)

        self.disp_panel.setLayout(disp_layout)
        self.main_layout.addWidget(self.disp_panel)

    def build_display_tabs(self):
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.scope_tab = QWidget()
        scope_layout = QVBoxLayout()
        self.plot1 = pg.PlotWidget()
        self.plot2 = pg.PlotWidget()
        self.plot3 = pg.PlotWidget()
        
        self.setup_plot_style(self.plot1, "INPUT MESSAGE SIGNAL")
        self.setup_plot_style(self.plot2, "RF TRANSFERS - MODULATED PASSBAND SIGNAL")
        self.setup_plot_style(self.plot3, "RECEIVER STAGE - CAPTURED SIGNAL CORRUPTED BY CHANNEL")
        
        self.curve1 = self.plot1.plot(pen=pg.mkPen((255, 255, 0), width=2))
        self.curve2 = self.plot2.plot(pen=pg.mkPen((0, 255, 255), width=2))
        self.curve3 = self.plot3.plot(pen=pg.mkPen((255, 50, 50), width=2))
        
        scope_layout.addWidget(self.plot1)
        scope_layout.addWidget(self.plot2)
        scope_layout.addWidget(self.plot3)
        self.scope_tab.setLayout(scope_layout)
        self.tabs.addTab(self.scope_tab, "Oscilloscope")

        self.fft_tab = QWidget()
        fft_layout = QVBoxLayout()
        self.fft_plot = pg.PlotWidget()
        self.setup_plot_style(self.fft_plot, "SPECTRUM ENGINE OVERLAY (Baseband Input vs Passband Channel)")
        self.fft_plot.addLegend()
        self.fft_msg_curve = self.fft_plot.plot(pen=pg.mkPen((255, 255, 0), width=2), name="Baseband Msg")
        self.fft_tx_curve = self.fft_plot.plot(pen=pg.mkPen((0, 255, 255), width=2), name="Modulated Passband")
        fft_layout.addWidget(self.fft_plot)
        self.fft_tab.setLayout(fft_layout)
        self.tabs.addTab(self.fft_tab, "Spectrum Analyzer")

        self.waterfall_tab = QWidget()
        waterfall_layout = QVBoxLayout()
        self.waterfall = pg.ImageView()
        waterfall_layout.addWidget(self.waterfall)
        self.waterfall_tab.setLayout(waterfall_layout)
        self.tabs.addTab(self.waterfall_tab, "Waterfall Map")
        self.waterfall_data = np.zeros((300, N // 2))

        self.audio_tab = QWidget()
        audio_layout = QVBoxLayout()
        self.audio_plot = pg.PlotWidget()
        self.setup_plot_style(self.audio_plot, "RECONSTRUCTED OUTPUT AUDIBLE SIGNAL PROFILE")
        self.audio_curve = self.audio_plot.plot(pen=pg.mkPen((255, 100, 255), width=2))
        audio_layout.addWidget(self.audio_plot)
        self.audio_tab.setLayout(audio_layout)
        self.tabs.addTab(self.audio_tab, "Demodulated Profile")

    def build_hardware_hud(self):
        self.hud_panel = QFrame()
        self.hud_panel.setStyleSheet("border: 1px solid #113333; background-color: #0b1111; padding: 5px;")
        hud_layout = QHBoxLayout()
        
        self.hud_pow_in = QLabel("INPUT POWER: 0.00W")
        self.hud_pow_tx = QLabel("TX RF POWER: 0.00W")
        self.hud_pow_rec = QLabel("DEMOD OUTPUT POWER: 0.00W")
        
        hud_list = [self.hud_pow_in, self.hud_pow_tx, self.hud_pow_rec]
        for hud in hud_list:
            hud.setStyleSheet("font-family: Consolas; font-size: 11pt; color: #00FFCC; border: none;")
            hud_layout.addWidget(hud)
            
        self.hud_panel.setLayout(hud_layout)
        self.main_layout.addWidget(self.hud_panel)

    def setup_plot_style(self, plot, title):
        plot.setBackground((5, 5, 5))
        plot.setTitle(title, color='w', size='11pt')
        plot.showGrid(x=True, y=True, alpha=0.3)

    def connect_system_events(self):
        self.clear_btn.clicked.connect(self.clear_signal)
        self.wave_box.currentIndexChanged.connect(self.on_wave_box_changed)
        self.plot1.scene().sigMouseMoved.connect(self.mouse_draw)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_signal)
        self.timer.start(50)

    def on_wave_box_changed(self):
        if self.wave_box.currentText() != "Mouse Draw":
            self.clear_signal()

    def mouse_draw(self, pos):
        global message
        if self.wave_box.currentText() != "Mouse Draw":
            return
        mousePoint = self.plot1.plotItem.vb.mapSceneToView(pos)
        xp, yp = mousePoint.x(), mousePoint.y()
        idx = int(xp * N / t[-1]) if t[-1] != 0 else 0
        if 0 <= idx < N:
            width = 32
            for i in range(-width, width):
                if 0 <= idx + i < N:
                    message[idx + i] = np.clip(yp * np.exp(-i * i / 120), -1, 1)

    def clear_signal(self):
        global message
        message[:] = 0

    def run_ai_classification(self, sig, noise):
        features = [[np.mean(sig), np.std(sig), np.max(sig)]]
        pred = clf.predict(features)[0]
        
        if np.max(np.abs(sig)) > 0.01:
            ai_output_string = pred
        else:
            ai_output_string = "SILENT / ZERO"
            
        self.ai_label.setText(f"AI SIGNAL RECOGNITION: {ai_output_string}")
        
        if noise < 0.1:   quality = "EXCELLENT (HIGH SNR)"
        elif noise < 0.3: quality = "GOOD"
        elif noise < 0.6: quality = "DEGRADED (POOR)"
        else:             quality = "CRITICAL FAILURE (NOISY / FADING)"
            
        self.snr_label.setText(f"AI CHANNEL QUALITY: {quality}")

    # ==============================================================================
    # RUN MAIN LOOP ITERATION (WITH ANIMATION PHASE DRIVER)
    # ==============================================================================
    def process_signal(self):
        global message, phase_offset
        w_type = self.wave_box.currentText()
        mf = self.msg_freq_slider.value()
        fc = self.freq_slider.value()
        ka = self.amp_slider.value() / 100
        noise = self.noise_slider.value() / 100
        mode = self.mode_box.currentText()

        # Step A: Increment the animation runtime phase factor
        # Higher multiplier = faster slithering speed across the viewport
        phase_offset -= 0.25 

        # Step B: Baseband generation paths loaded with phase data
        if w_type != "Mouse Draw":
            msg = generate_base_waveform(w_type, mf, 1.0, phase_offset)
            current_freq_str = f"{mf} Hz"
        else:
            # For custom drawn items, we circularly roll the array entries to make it crawl
            msg = np.roll(message, int(phase_offset * 2) % N)
            if np.max(np.abs(msg)) > 0:
                msg = msg / np.max(np.abs(msg))
            current_freq_str = "VARIABLE (DRAWN)"

        # Step C: Telemetry analysis metrics collection
        p_in_w, p_in_db = calculate_power(msg)
        v_pp = float(np.max(msg) - np.min(msg))
        v_rms = float(np.sqrt(np.mean(msg ** 2)))

        self.lbl_input_shape.setText(f"PROFILE: {w_type.upper()}")
        self.lbl_input_freq.setText(f"FREQ: {current_freq_str}")
        self.lbl_input_vpp.setText(f"Vpp: {v_pp:.2f} V")
        self.lbl_input_vrms.setText(f"Vrms: {v_rms:.2f} V")
        self.lbl_input_p_w.setText(f"POWER: {p_in_w:.4f} W")
        self.lbl_input_p_dbm.setText(f"POWER: {p_in_db:.1f} dBm")

        # Step D: Run animated modulators
        if mode == "AM":       
            tx = am_modulation(msg, fc, ka, phase_offset)
        elif mode == "DSB-SC": 
            tx = dsb_sc_modulation(msg, fc, phase_offset)
        elif mode == "SSB-SC": 
            tx = ssb_sc_modulation(msg, fc, phase_offset)
        elif mode == "FM":     
            tx = fm_modulation(msg, fc, phase_offset)

        p_tx_w, p_tx_db = calculate_power(tx)

        # Step E: Propagation via fading multi-path channel
        fading_loss = 1.0 + 0.25 * np.sin(2 * np.pi * 3 * t)
        echo_reflection = np.roll(tx, 40) * 0.25
        awgn_noise = np.random.normal(0, noise, N)
        
        rx = tx * fading_loss + echo_reflection + awgn_noise
        rx = np.clip(rx, -2.5, 2.5)

        # Step F: Receiver demodulation execution
        if mode == "AM":   
            rec = envelope_detector(rx)
        elif mode == "FM": 
            rec = fm_demodulation(rx)
        else:              
            rec = coherent_detector(rx, fc, phase_offset)

        p_rec_w, p_rec_db = calculate_power(rec)
        stream.write(rec.astype(np.float32))

        # Step G: Spectrum Transformations (FFT stays structurally stable)
        freq_axis = np.fft.fftfreq(N, 1 / FS)[:N // 2]
        fft_msg_db = 20 * np.log10(np.abs(np.fft.fft(msg))[:N // 2] + 1e-6)
        fft_tx_db = 20 * np.log10(np.abs(np.fft.fft(tx))[:N // 2] + 1e-6)

        # Waterfall updates
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0] = fft_tx_db
        self.waterfall.setImage(self.waterfall_data, autoLevels=False)

        self.run_ai_classification(msg, noise)

        # Step H: UI Render updates
        self.plot1.setTitle(f"INPUT MSG SIGNAL ({w_type.upper()}) | Power: {p_in_w:.4f} W ({p_in_db:.1f} dBm)")
        self.plot2.setTitle(f"TX MODULATED SIGNAL ({mode}) | Power: {p_tx_w:.4f} W ({p_tx_db:.1f} dBm)")
        self.plot3.setTitle(f"CHANNEL RECEIVED SIGNAL | Power: {float(np.mean(rx**2)):.4f} W")
        self.audio_plot.setTitle(f"DEMODULATED OUTPUT WAVEFORM | Power: {p_rec_w:.4f} W ({p_rec_db:.1f} dBm)")

        self.hud_pow_in.setText(f"INPUT SIGNAL POWER: {p_in_w:.5f} W | {p_in_db:.2f} dBm")
        self.hud_pow_tx.setText(f"RF OUTPUT TX POWER: {p_tx_w:.5f} W | {p_tx_db:.2f} dBm")
        self.hud_pow_rec.setText(f"RECOVERED RX POWER: {p_rec_w:.5f} W | {p_rec_db:.2f} dBm")

        self.curve1.setData(t, msg)
        self.curve2.setData(t, tx)
        self.curve3.setData(t, rx)
        self.audio_curve.setData(t, rec)
        self.fft_msg_curve.setData(freq_axis, fft_msg_db)
        self.fft_tx_curve.setData(freq_axis, fft_tx_db)

# ==============================================================================
# 5. INITIALIZATION RUNTIME LAYER
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
    QWidget { 
        background-color: #050909; 
        color: #00FFCC; 
        font-size: 11pt; 
        font-family: Consolas, Monospace; 
    }
    QLabel { color: #00FFCC; font-weight: bold; }
    QPushButton { background-color: #0f1717; border: 2px solid #00FFCC; border-radius: 6px; padding: 5px; color: #00FFCC; }
    QPushButton:hover { background-color: #00FFCC; color: #050909; }
    QComboBox { background-color: #0f1717; border: 1px solid #00FFCC; color: #00FFCC; padding: 3px; }
    QSlider::groove:horizontal { background: #121a1a; height: 6px; }
    QSlider::handle:horizontal { background: #00FFCC; width: 14px; margin: -4px 0; border-radius: 7px; }
    QTabWidget::pane { border: 1px solid #00FFCC; }
    QTabBar::tab { background: #0f1717; padding: 8px 15px; border-right: 1px solid #121a1a; }
    QTabBar::tab:selected { background: #00FFCC; color: #050909; font-weight: bold; }
    """)
    window = AISDR()
    window.show()
    sys.exit(app.exec_())
