
# =========================================================
# AI POWERED SDR / COMMUNICATION LAB
# =========================================================
# FEATURES
# =========================================================
# ✔ AM / DSB-SC / SSB-SC / FM
# ✔ Real-Time FFT
# ✔ Waterfall Spectrum
# ✔ Oscilloscope
# ✔ Audio Output
# ✔ Mouse Drawing
# ✔ Keyboard Waveform Generator
# ✔ AI Signal Classification
# ✔ Noise Simulation
# ✔ Professional Cyberpunk GUI
# =========================================================

# INSTALL:
# pip install numpy scipy pyqtgraph PyQt5 sounddevice scikit-learn

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

# =========================================================
# PARAMETERS
# =========================================================

fs = 44100
N = 4096

t = np.arange(N) / fs

message = np.zeros(N)

# =========================================================
# AUDIO OUTPUT
# =========================================================

stream = sd.OutputStream(
    samplerate=fs,
    channels=1,
    blocksize=1024
)

stream.start()

# =========================================================
# AI TRAINING
# =========================================================

X_train = []
y_train = []

for _ in range(50):

    sine = np.sin(2*np.pi*5*t)

    square = np.sign(sine)

    noise = np.random.normal(0,1,N)

    X_train.append([
        np.mean(sine),
        np.std(sine),
        np.max(sine)
    ])
    y_train.append("SINE")

    X_train.append([
        np.mean(square),
        np.std(square),
        np.max(square)
    ])
    y_train.append("SQUARE")

    X_train.append([
        np.mean(noise),
        np.std(noise),
        np.max(noise)
    ])
    y_train.append("NOISE")

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

# =========================================================
# DSP UTILITIES
# =========================================================

def normalize(sig):

    mx = np.max(np.abs(sig))

    if mx == 0:
        return sig

    return sig / mx


def lowpass(sig, cutoff=4000):

    nyq = fs / 2

    b, a = butter(
        5,
        cutoff / nyq
    )

    return filtfilt(b, a, sig)

# =========================================================
# MODULATION
# =========================================================

def am_modulation(msg, fc, ka):

    carrier = np.cos(2*np.pi*fc*t)

    return (1 + ka*msg) * carrier


def dsb_sc_modulation(msg, fc):

    carrier = np.cos(2*np.pi*fc*t)

    return msg * carrier


def ssb_sc_modulation(msg, fc):

    analytic = hilbert(msg)

    h = np.imag(analytic)

    return (
        msg*np.cos(2*np.pi*fc*t)
        - h*np.sin(2*np.pi*fc*t)
    )


def fm_modulation(msg, fc, kf=500):

    integral = np.cumsum(msg) / fs

    return np.cos(
        2*np.pi*fc*t +
        2*np.pi*kf*integral
    )

# =========================================================
# DEMODULATION
# =========================================================

def envelope_detector(sig):

    rectified = np.abs(sig)

    rec = lowpass(rectified)

    rec -= np.mean(rec)

    return normalize(rec)


def coherent_detector(sig, fc):

    local = np.cos(2*np.pi*fc*t)

    mixed = sig * local

    rec = lowpass(mixed)

    rec -= np.mean(rec)

    return normalize(rec)


def fm_demodulation(sig):

    analytic = hilbert(sig)

    phase = np.unwrap(np.angle(analytic))

    demod = np.diff(phase)

    demod = np.append(demod, 0)

    demod -= np.mean(demod)

    return normalize(demod)

# =========================================================
# MAIN WINDOW
# =========================================================

class AISDR(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AI Powered SDR Laboratory"
        )

        self.showMaximized()

        self.main_layout = QVBoxLayout()

        # =================================================
        # HEADER
        # =================================================

        header = QLabel(
            "AI POWERED COMMUNICATION SYSTEM"
        )

        header.setStyleSheet("""
        font-size: 24pt;
        color: cyan;
        padding: 10px;
        """)

        header.setAlignment(Qt.AlignCenter)

        self.main_layout.addWidget(header)

        # =================================================
        # CONTROLS
        # =================================================

        controls = QHBoxLayout()

        # Carrier Frequency
        self.freq_slider = QSlider(Qt.Horizontal)

        self.freq_slider.setMinimum(100)

        self.freq_slider.setMaximum(5000)

        self.freq_slider.setValue(1000)

        controls.addWidget(QLabel("Carrier"))

        controls.addWidget(self.freq_slider)

        # Modulation Index
        self.amp_slider = QSlider(Qt.Horizontal)

        self.amp_slider.setMinimum(1)

        self.amp_slider.setMaximum(200)

        self.amp_slider.setValue(80)

        controls.addWidget(QLabel("Mod Index"))

        controls.addWidget(self.amp_slider)

        # Noise
        self.noise_slider = QSlider(Qt.Horizontal)

        self.noise_slider.setMinimum(0)

        self.noise_slider.setMaximum(100)

        self.noise_slider.setValue(2)

        controls.addWidget(QLabel("Noise"))

        controls.addWidget(self.noise_slider)

        # Mode Selection
        self.mode_box = QComboBox()

        self.mode_box.addItems([
            "AM",
            "DSB-SC",
            "SSB-SC",
            "FM"
        ])

        controls.addWidget(self.mode_box)

        # Clear Button
        self.clear_btn = QPushButton("CLEAR")

        controls.addWidget(self.clear_btn)

        self.main_layout.addLayout(controls)

        # =================================================
        # AI PANEL
        # =================================================

        ai_panel = QFrame()

        ai_panel.setStyleSheet("""
        border: 2px solid cyan;
        border-radius: 10px;
        padding: 10px;
        """)

        ai_layout = QHBoxLayout()

        self.ai_label = QLabel(
            "AI SIGNAL TYPE: UNKNOWN"
        )

        self.ai_label.setStyleSheet("""
        font-size: 16pt;
        color: yellow;
        """)

        ai_layout.addWidget(self.ai_label)

        self.snr_label = QLabel(
            "AI CHANNEL QUALITY: GOOD"
        )

        self.snr_label.setStyleSheet("""
        font-size: 16pt;
        color: lime;
        """)

        ai_layout.addWidget(self.snr_label)

        ai_panel.setLayout(ai_layout)

        self.main_layout.addWidget(ai_panel)

        # =================================================
        # TABS
        # =================================================

        self.tabs = QTabWidget()

        self.main_layout.addWidget(self.tabs)

        # =================================================
        # OSCILLOSCOPE TAB
        # =================================================

        self.scope_tab = QWidget()

        scope_layout = QVBoxLayout()

        self.plot1 = pg.PlotWidget()
        self.plot2 = pg.PlotWidget()
        self.plot3 = pg.PlotWidget()

        self.setup_plot(
            self.plot1,
            "MESSAGE SIGNAL"
        )

        self.setup_plot(
            self.plot2,
            "MODULATED SIGNAL"
        )

        self.setup_plot(
            self.plot3,
            "RECEIVED SIGNAL"
        )

        self.curve1 = self.plot1.plot(
            pen=pg.mkPen((255,255,0), width=2)
        )

        self.curve2 = self.plot2.plot(
            pen=pg.mkPen((0,255,255), width=2)
        )

        self.curve3 = self.plot3.plot(
            pen=pg.mkPen((255,0,0), width=2)
        )

        scope_layout.addWidget(self.plot1)
        scope_layout.addWidget(self.plot2)
        scope_layout.addWidget(self.plot3)

        self.scope_tab.setLayout(scope_layout)

        self.tabs.addTab(
            self.scope_tab,
            "Oscilloscope"
        )

        # =================================================
        # FFT TAB
        # =================================================

        self.fft_tab = QWidget()

        fft_layout = QVBoxLayout()

        self.fft_plot = pg.PlotWidget()

        self.setup_plot(
            self.fft_plot,
            "FFT ANALYZER"
        )

        self.fft_curve = self.fft_plot.plot(
            pen=pg.mkPen((0,255,0), width=2)
        )

        fft_layout.addWidget(self.fft_plot)

        self.fft_tab.setLayout(fft_layout)

        self.tabs.addTab(
            self.fft_tab,
            "FFT"
        )

        # =================================================
        # WATERFALL TAB
        # =================================================

        self.waterfall_tab = QWidget()

        waterfall_layout = QVBoxLayout()

        self.waterfall = pg.ImageView()

        waterfall_layout.addWidget(
            self.waterfall
        )

        self.waterfall_tab.setLayout(
            waterfall_layout
        )

        self.tabs.addTab(
            self.waterfall_tab,
            "Waterfall"
        )

        self.waterfall_data = np.zeros(
            (300, N//2)
        )

        # =================================================
        # AUDIO TAB
        # =================================================

        self.audio_tab = QWidget()

        audio_layout = QVBoxLayout()

        self.audio_plot = pg.PlotWidget()

        self.setup_plot(
            self.audio_plot,
            "RECOVERED AUDIO"
        )

        self.audio_curve = self.audio_plot.plot(
            pen=pg.mkPen((255,100,255), width=2)
        )

        audio_layout.addWidget(self.audio_plot)

        self.audio_tab.setLayout(audio_layout)

        self.tabs.addTab(
            self.audio_tab,
            "Audio"
        )

        self.setLayout(self.main_layout)

        # =================================================
        # EVENTS
        # =================================================

        self.freq_slider.valueChanged.connect(
            self.process_signal
        )

        self.amp_slider.valueChanged.connect(
            self.process_signal
        )

        self.noise_slider.valueChanged.connect(
            self.process_signal
        )

        self.mode_box.currentIndexChanged.connect(
            self.process_signal
        )

        self.clear_btn.clicked.connect(
            self.clear_signal
        )

        # Mouse drawing
        self.plot1.scene().sigMouseMoved.connect(
            self.mouse_draw
        )

        # Timer
        self.timer = QTimer()

        self.timer.timeout.connect(
            self.process_signal
        )

        self.timer.start(50)

    # =====================================================
    # PLOT STYLE
    # =====================================================

    def setup_plot(self, plot, title):

        plot.setBackground((0,0,0))

        plot.setTitle(
            title,
            color='w',
            size='14pt'
        )

        plot.showGrid(
            x=True,
            y=True
        )

        plot.setMouseEnabled(
            x=True,
            y=True
        )

    # =====================================================
    # MOUSE DRAWING
    # =====================================================

    def mouse_draw(self, pos):

        global message

        mousePoint = self.plot1.plotItem.vb.mapSceneToView(pos)

        xp = mousePoint.x()

        yp = mousePoint.y()

        idx = int(xp * N / t[-1])

        if 0 <= idx < N:

            width = 30

            for i in range(-width, width):

                if 0 <= idx+i < N:

                    smooth = (
                        yp
                        * np.exp(-i*i/100)
                    )

                    message[idx+i] = np.clip(
                        smooth,
                        -1,
                        1
                    )

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_signal(self):

        global message

        message[:] = 0

    # =====================================================
    # AI ANALYSIS
    # =====================================================

    def ai_analysis(self, sig, noise):

        features = [[
            np.mean(sig),
            np.std(sig),
            np.max(sig)
        ]]

        pred = clf.predict(features)[0]

        self.ai_label.setText(
            f"AI SIGNAL TYPE: {pred}"
        )

        if noise < 0.1:

            quality = "EXCELLENT"

        elif noise < 0.3:

            quality = "GOOD"

        elif noise < 0.6:

            quality = "POOR"

        else:

            quality = "VERY NOISY"

        self.snr_label.setText(
            f"AI CHANNEL QUALITY: {quality}"
        )

    # =====================================================
    # PROCESS SIGNAL
    # =====================================================

    def process_signal(self):

        global message

        fc = self.freq_slider.value()

        ka = self.amp_slider.value()/100

        noise = self.noise_slider.value()/100

        mode = self.mode_box.currentText()

        msg = normalize(message)

        # =================================================
        # MODULATION
        # =================================================

        if mode == "AM":

            tx = am_modulation(
                msg,
                fc,
                ka
            )

        elif mode == "DSB-SC":

            tx = dsb_sc_modulation(
                msg,
                fc
            )

        elif mode == "SSB-SC":

            tx = ssb_sc_modulation(
                msg,
                fc
            )

        elif mode == "FM":

            tx = fm_modulation(
                msg,
                fc
            )

        # =================================================
        # CHANNEL
        # =================================================

        fading = (
            1
            + 0.2*np.sin(
                2*np.pi*2*t
            )
        )

        rx = tx * fading

        rx += np.roll(rx, 50) * 0.3

        rx += np.random.normal(
            0,
            noise,
            N
        )

        rx = np.clip(rx, -2, 2)

        # =================================================
        # DEMODULATION
        # =================================================

        if mode == "AM":

            rec = envelope_detector(rx)

        elif mode == "FM":

            rec = fm_demodulation(rx)

        else:

            rec = coherent_detector(
                rx,
                fc
            )

        # =================================================
        # AUDIO OUTPUT
        # =================================================

        audio = normalize(rec)

        stream.write(
            audio.astype(np.float32)
        )

        # =================================================
        # FFT
        # =================================================

        fft_data = np.abs(
            np.fft.fft(rx)
        )

        fft_half = 20*np.log10(
            fft_data[:N//2] + 1e-6
        )

        # =================================================
        # WATERFALL
        # =================================================

        self.waterfall_data = np.roll(
            self.waterfall_data,
            1,
            axis=0
        )

        self.waterfall_data[0] = fft_half

        self.waterfall.setImage(
            self.waterfall_data,
            autoLevels=False
        )

        # =================================================
        # AI ANALYSIS
        # =================================================

        self.ai_analysis(
            msg,
            noise
        )

        # =================================================
        # UPDATE PLOTS
        # =================================================

        self.curve1.setData(
            t,
            msg
        )

        self.curve2.setData(
            t,
            tx
        )

        self.curve3.setData(
            t,
            rx
        )

        self.audio_curve.setData(
            t,
            rec
        )

        self.fft_curve.setData(
            fft_half
        )

    # =====================================================
    # KEYBOARD SHORTCUTS
    # =====================================================

    def keyPressEvent(self, event):

        global message

        key = event.key()

        # SINE
        if key == Qt.Key_S:

            message[:] = np.sin(
                2*np.pi*5*t
            )

        # SQUARE
        elif key == Qt.Key_Q:

            message[:] = np.sign(
                np.sin(2*np.pi*5*t)
            )

        # TRIANGLE
        elif key == Qt.Key_T:

            message[:] = (
                2*np.abs(
                    2*(
                        t*5
                        - np.floor(t*5 + 0.5)
                    )
                ) - 1
            )

        # RANDOM NOISE
        elif key == Qt.Key_N:

            message[:] = np.random.normal(
                0,
                0.5,
                N
            )

        # CLEAR
        elif key == Qt.Key_C:

            message[:] = 0

# =========================================================
# APPLICATION
# =========================================================

app = QApplication(sys.argv)

app.setStyle("Fusion")

# =========================================================
# THEME
# =========================================================

app.setStyleSheet("""

QWidget {
    background-color: #080808;
    color: #00FFCC;
    font-size: 12pt;
    font-family: Consolas;
}

QPushButton {
    background-color: #111111;
    border: 2px solid #00FFCC;
    border-radius: 12px;
    padding: 8px;
}

QPushButton:hover {
    background-color: #00FFCC;
    color: black;
}

QComboBox {
    background-color: #111111;
    border: 2px solid #00FFCC;
    padding: 5px;
}

QSlider::groove:horizontal {
    background: #222222;
    height: 8px;
}

QSlider::handle:horizontal {
    background: #00FFCC;
    width: 20px;
    margin: -5px 0;
    border-radius: 10px;
}

QTabWidget::pane {
    border: 2px solid #00FFCC;
}

QTabBar::tab {
    background: #111111;
    padding: 10px;
}

QTabBar::tab:selected {
    background: #00FFCC;
    color: black;
}

""")

window = AISDR()

window.show()

sys.exit(app.exec_())

