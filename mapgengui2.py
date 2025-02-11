import sys
import random
import math
import os
import numpy as np
import multiprocessing
import opensimplex
from PIL import Image
from typing import Tuple, Dict
import logging

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QProgressBar,
    QMenu,
    QAction,
    QFileDialog,
    QMessageBox,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QScrollArea,
    QMenuBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

try:
    import qdarkstyle
except ImportError:
    qdarkstyle = None

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# COLOR DEFINITIONS
# --------------------------
Color = Tuple[int, int, int]
COLOR_MAP: Dict[str, Color] = {
    "dirt": (0, 0, 0),
    "plant": (0, 224, 0),
    "grass": (0, 255, 0),
    "water": (102, 0, 255),
    "metal_bg": (102, 102, 102),
    "water_metal_bg": (102, 102, 255),
    "metal_bg_2": (153, 102, 255),
    "NoFly": (255, 0, 153),
    "Ice": (102, 204, 255),
    "Lit Cave": (153, 0, 0),
    "Cave Air": (153, 51, 0),
    "fluxite": (175, 0, 224),
    "hard ice": (204, 255, 255),
    "bedrock": (170, 170, 170),
    "collector_start": (255, 0, 0),
    "red sand": (255, 85, 0),
    "lava": (255, 102, 0),
    "air": (255, 255, 255),
    "door": (0, 102, 0),
    "air_stone_bg": (51, 51, 51),
}
BORDER_WIDTH = 10

# --------------------------
# OVERWORLD TERRAIN
# --------------------------
def generate_overworld_terrain(array: np.ndarray) -> None:
    import opensimplex
    rock_base, width = array.shape[0], array.shape[1]
    max_height = 50
    for ix in range(width):
        val = opensimplex.noise2(ix * 0.01, 0)
        start_height = rock_base - int(val * max_height) - 50
        if start_height < 0:
            start_height = 0
        elif start_height >= rock_base:
            start_height = rock_base - 1

        grass_top = start_height
        grass_bot = min(start_height + 20, rock_base)
        array[grass_top:grass_bot, ix] = COLOR_MAP["grass"]
        array[grass_bot:, ix] = COLOR_MAP["dirt"]

# --------------------------
# NOISE GENERATION
# --------------------------
def _noise_chunk_worker(chunk_start, chunk_end, shape, freq, seed):
    import opensimplex
    opensimplex.seed(seed)
    h, w = shape
    partial_height = chunk_end - chunk_start
    out = np.zeros((partial_height, w), dtype=np.float32)
    for local_row in range(partial_height):
        iy = chunk_start + local_row
        row_vals = [opensimplex.noise2(ix * freq, iy * freq) for ix in range(w)]
        out[local_row, :] = row_vals
    return (chunk_start, chunk_end, out)

def generate_noise_array_parallel(height, width, freq, seed):
    if seed is None:
        seed = random.randint(0, 99999999)
    noise_array = np.zeros((height, width), dtype=np.float32)
    num_proc = max(1, min(multiprocessing.cpu_count(), 8))
    chunk_size = max(1, math.ceil(height / num_proc))

    tasks = []
    row_start = 0
    while row_start < height:
        row_end = min(row_start + chunk_size, height)
        tasks.append((row_start, row_end, (height, width), freq, seed))
        row_start = row_end

    if not tasks:
        return noise_array

    with multiprocessing.Pool(num_proc) as pool:
        results = pool.starmap(_noise_chunk_worker, tasks)

    for (chunk_start, chunk_end, partial_data) in results:
        noise_array[chunk_start:chunk_end, :] = partial_data
    return noise_array

class MapParamsDialog(QDialog):
    def __init__(self, parent,
                 current_size: int,
                 current_rockbase: int,
                 water_enabled: bool,
                 water_thresh: float,
                 water_freq: float,
                 fluxite_enabled: bool,
                 fluxite_thresh: float,
                 fluxite_freq: float,
                 plant_enabled: bool,
                 plant_thresh: float,
                 plant_freq: float,
                 caves_enabled: bool,
                 cave_thresh: float,
                 cave_freq: float,
                 current_seed: str):
        super().__init__(parent)
        self.setWindowTitle("Map Parameters")

        self._size = current_size
        self._rock_base = current_rockbase
        self._water_enabled = water_enabled
        self._water_threshold = water_thresh
        self._water_freq = water_freq
        self._fluxite_enabled = fluxite_enabled
        self._fluxite_threshold = fluxite_thresh
        self._fluxite_freq = fluxite_freq
        self._plant_enabled = plant_enabled
        self._plant_threshold = plant_thresh
        self._plant_freq = plant_freq
        self._caves_enabled = caves_enabled
        self._cave_threshold = cave_thresh
        self._cave_freq = cave_freq
        self._seed_string = current_seed

        layout = QFormLayout(self)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(100, 4000)
        self.size_spin.setValue(self._size)
        layout.addRow("Map Size:", self.size_spin)

        self.rockbase_spin = QSpinBox()
        self.rockbase_spin.setRange(0, 3999)
        self.rockbase_spin.setValue(self._rock_base)
        layout.addRow("Rock Base:", self.rockbase_spin)

        self.water_check = QCheckBox("Enable Water")
        self.water_check.setChecked(self._water_enabled)
        layout.addRow(self.water_check)
        self.water_threshold_spin = QDoubleSpinBox()
        self.water_threshold_spin.setRange(0.0, 1.0)
        self.water_threshold_spin.setDecimals(4)
        self.water_threshold_spin.setSingleStep(0.0001)
        self.water_threshold_spin.setValue(self._water_threshold)
        layout.addRow("Water Threshold:", self.water_threshold_spin)
        self.water_freq_spin = QDoubleSpinBox()
        self.water_freq_spin.setRange(0.0, 1.0)
        self.water_freq_spin.setDecimals(4)
        self.water_freq_spin.setSingleStep(0.0001)
        self.water_freq_spin.setValue(self._water_freq)
        layout.addRow("Water Freq:", self.water_freq_spin)

        self.fluxite_check = QCheckBox("Enable Fluxite")
        self.fluxite_check.setChecked(self._fluxite_enabled)
        layout.addRow(self.fluxite_check)
        self.fluxite_threshold_spin = QDoubleSpinBox()
        self.fluxite_threshold_spin.setRange(0, 1)
        self.fluxite_threshold_spin.setDecimals(4)
        self.fluxite_threshold_spin.setSingleStep(0.0001)
        self.fluxite_threshold_spin.setValue(self._fluxite_threshold)
        layout.addRow("Fluxite Threshold:", self.fluxite_threshold_spin)
        self.fluxite_freq_spin = QDoubleSpinBox()
        self.fluxite_freq_spin.setRange(0, 1)
        self.fluxite_freq_spin.setDecimals(4)
        self.fluxite_freq_spin.setSingleStep(0.0001)
        self.fluxite_freq_spin.setValue(self._fluxite_freq)
        layout.addRow("Fluxite Freq:", self.fluxite_freq_spin)

        self.plant_check = QCheckBox("Enable Plants")
        self.plant_check.setChecked(self._plant_enabled)
        layout.addRow(self.plant_check)
        self.plant_threshold_spin = QDoubleSpinBox()
        self.plant_threshold_spin.setRange(0, 1)
        self.plant_threshold_spin.setDecimals(4)
        self.plant_threshold_spin.setSingleStep(0.0001)
        self.plant_threshold_spin.setValue(self._plant_threshold)
        layout.addRow("Plant Threshold:", self.plant_threshold_spin)
        self.plant_freq_spin = QDoubleSpinBox()
        self.plant_freq_spin.setRange(0, 1)
        self.plant_freq_spin.setDecimals(4)
        self.plant_freq_spin.setSingleStep(0.0001)
        self.plant_freq_spin.setValue(self._plant_freq)
        layout.addRow("Plant Freq:", self.plant_freq_spin)

        self.caves_check = QCheckBox("Enable Caves")
        self.caves_check.setChecked(self._caves_enabled)
        layout.addRow(self.caves_check)
        self.cave_threshold_spin = QDoubleSpinBox()
        self.cave_threshold_spin.setRange(0, 1)
        self.cave_threshold_spin.setDecimals(4)
        self.cave_threshold_spin.setSingleStep(0.0001)
        self.cave_threshold_spin.setValue(self._cave_threshold)
        layout.addRow("Cave Threshold:", self.cave_threshold_spin)
        self.cave_freq_spin = QDoubleSpinBox()
        self.cave_freq_spin.setRange(0, 1)
        self.cave_freq_spin.setDecimals(4)
        self.cave_freq_spin.setSingleStep(0.0001)
        self.cave_freq_spin.setValue(self._cave_freq)
        layout.addRow("Cave Freq:", self.cave_freq_spin)

        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("Leave blank for random seed")
        self.seed_edit.setText(self._seed_string)
        layout.addRow("Seed:", self.seed_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def on_accept(self):
        self._size = self.size_spin.value()
        self._rock_base = self.rockbase_spin.value()
        if self._rock_base >= self._size:
            self._rock_base = self._size - 1

        self._water_enabled = self.water_check.isChecked()
        self._water_threshold = self.water_threshold_spin.value()
        self._water_freq = self.water_freq_spin.value()

        self._fluxite_enabled = self.fluxite_check.isChecked()
        self._fluxite_threshold = self.fluxite_threshold_spin.value()
        self._fluxite_freq = self.fluxite_freq_spin.value()

        self._plant_enabled = self.plant_check.isChecked()
        self._plant_threshold = self.plant_threshold_spin.value()
        self._plant_freq = self.plant_freq_spin.value()

        self._caves_enabled = self.caves_check.isChecked()
        self._cave_threshold = self.cave_threshold_spin.value()
        self._cave_freq = self.cave_freq_spin.value()

        text_seed = self.seed_edit.text().strip()
        if not text_seed:
            text_seed = ""
        self._seed_string = text_seed

        self.accept()

    @property
    def size(self): return self._size
    @property
    def rock_base(self): return self._rock_base
    @property
    def water_enabled(self): return self._water_enabled
    @property
    def water_threshold(self): return self._water_threshold
    @property
    def water_freq(self): return self._water_freq
    @property
    def fluxite_enabled(self): return self._fluxite_enabled
    @property
    def fluxite_threshold(self): return self._fluxite_threshold
    @property
    def fluxite_freq(self): return self._fluxite_freq
    @property
    def plant_enabled(self): return self._plant_enabled
    @property
    def plant_threshold(self): return self._plant_threshold
    @property
    def plant_freq(self): return self._plant_freq
    @property
    def caves_enabled(self): return self._caves_enabled
    @property
    def cave_threshold(self): return self._cave_threshold
    @property
    def cave_freq(self): return self._cave_freq
    @property
    def seed_string(self): return self._seed_string


class GenerationWorker(QThread):
    progressMsg = pyqtSignal(str)
    progressVal = pyqtSignal(int)
    resultReady = pyqtSignal(object)

    def __init__(self,
                 size, rock_base,
                 water_enabled, water_threshold, water_freq,
                 fluxite_enabled, fluxite_threshold, fluxite_freq,
                 plant_enabled, plant_threshold, plant_freq,
                 caves_enabled, cave_threshold, cave_freq,
                 seed_str):
        super().__init__()
        self.size = size
        self.rock_base = rock_base
        self.water_enabled = water_enabled
        self.water_threshold = water_threshold
        self.water_freq = water_freq
        self.fluxite_enabled = fluxite_enabled
        self.fluxite_threshold = fluxite_threshold
        self.fluxite_freq = fluxite_freq
        self.plant_enabled = plant_enabled
        self.plant_threshold = plant_threshold
        self.plant_freq = plant_freq
        self.caves_enabled = caves_enabled
        self.cave_threshold = cave_threshold
        self.cave_freq = cave_freq

        if seed_str:
            try:
                self.noise_seed = int(seed_str)
            except ValueError:
                self.noise_seed = random.randint(0, 99999999)
        else:
            self.noise_seed = random.randint(0, 99999999)

        w = self.size
        h = self.size
        border_steps = 3 * BORDER_WIDTH
        underground_h = max(0, h - rock_base)

        steps = rock_base
        if water_enabled: steps += underground_h
        if fluxite_enabled: steps += underground_h
        if plant_enabled: steps += underground_h
        steps += 1
        steps += h
        if caves_enabled: steps += h
        steps += border_steps
        self.maxGlobalSteps = steps
        self.currentGlobalStep = 0

    def report(self, stageName: str, localIndex: int, localMax: int):
        if localMax <= 1:
            localPct = 100
        else:
            localPct = int((localIndex / (localMax - 1)) * 100)
        if not hasattr(self, '_lastLocalPct'):
            self._lastLocalPct = {}
        if stageName not in self._lastLocalPct:
            self._lastLocalPct[stageName] = -1

        if localPct != self._lastLocalPct[stageName] or localIndex == localMax - 1:
            self.progressMsg.emit(f"{stageName}: {localPct}%")
            self._lastLocalPct[stageName] = localPct

        self.currentGlobalStep += 1
        if self.currentGlobalStep > self.maxGlobalSteps:
            self.currentGlobalStep = self.maxGlobalSteps
        globalPct = int((self.currentGlobalStep / self.maxGlobalSteps) * 100)
        self.progressVal.emit(globalPct)

    def run(self):
        opensimplex.seed(self.noise_seed)
        self.progressMsg.emit(f"Using Seed = {self.noise_seed}")

        w = self.size
        h = self.size

        overworld = np.full((self.rock_base, w, 3), COLOR_MAP["air"], dtype=np.uint8)
        generate_overworld_terrain(overworld)
        for row_i in range(self.rock_base):
            self.report("Overworld Terrain", row_i, self.rock_base)

        underground_h = h - self.rock_base
        underground = np.full((underground_h, w, 3), COLOR_MAP["dirt"], dtype=np.uint8)

        if self.water_enabled and underground_h > 0:
            self.apply_resource_parallel(underground, "Water (Underground)",
                                         freq=self.water_freq,
                                         threshold=self.water_threshold,
                                         material=COLOR_MAP["water"],
                                         global_offset=self.rock_base)
        if self.fluxite_enabled and underground_h > 0:
            self.apply_resource_parallel(underground, "Fluxite",
                                         freq=self.fluxite_freq,
                                         threshold=self.fluxite_threshold,
                                         material=COLOR_MAP["fluxite"],
                                         global_offset=self.rock_base)
        if self.plant_enabled and underground_h > 0:
            self.apply_resource_parallel(underground, "Plants",
                                         freq=self.plant_freq,
                                         threshold=self.plant_threshold,
                                         material=COLOR_MAP["plant"],
                                         global_offset=self.rock_base)

        combined = np.vstack((overworld, underground))
        self.report("Combine Overworld+Underground", 1, 1)

        self.draw_mountain(1, 20, 0, combined, "Mountains (Left)")
        self.draw_mountain(2, w - 20, 0, combined, "Mountains (Right)")

        if self.caves_enabled:
            self.apply_caves_parallel(combined, "Caves")

        self.world_border(combined, "World Border")

        self.resultReady.emit(combined)

    def apply_resource_parallel(self, arr, stageName, freq, threshold, material, global_offset: int):
        h, w, _ = arr.shape
        self.progressMsg.emit(f"{stageName}: generating noise array (parallel)")
        noise_arr = generate_noise_array_parallel(h, w, freq, self.noise_seed)
        self.progressMsg.emit(f"{stageName}: applying threshold overlay")

        for local_row in range(h):
            global_row = local_row + global_offset
            if global_row < self.rock_base:
                continue
            row_mask = noise_arr[local_row, :] > threshold
            arr[local_row, row_mask] = material
            self.report(stageName, local_row, h)

    def apply_caves_parallel(self, arr, stageName):
        h, w, _ = arr.shape
        if h <= self.rock_base:
            return
        self.progressMsg.emit(f"{stageName}: generating noise array (parallel)")
        noise_arr = generate_noise_array_parallel(h, w, self.cave_freq, self.noise_seed)
        self.progressMsg.emit(f"{stageName}: applying threshold overlay")

        for row_i in range(self.rock_base):
            self.report(stageName, row_i, h)
        mask = (noise_arr > self.cave_threshold)
        for row_i in range(self.rock_base, h):
            self.report(stageName, row_i, h)
            row_mask = mask[row_i, :]
            arr[row_i, row_mask] = COLOR_MAP["Cave Air"]

    def draw_mountain(self, side, start_x, start_y, arr, stageName):
        h, w, _ = arr.shape
        current_x = start_x
        bottom_threshold = int(h * 0.3)
        for y in range(start_y, h):
            self.report(stageName, y, h)
            prob = (y - start_y) / (h - start_y)
            if y >= bottom_threshold:
                prob *= 0.6
            move = 0
            if side == 1 and random.random() < prob:
                move = 1
            elif side == 2 and random.random() < prob:
                move = -1
            current_x = min(max(current_x + move, 0), w - 1)
            arr[y, current_x] = COLOR_MAP["bedrock"]
            if side == 1:
                arr[y, :current_x+1] = COLOR_MAP["bedrock"]
            else:
                arr[y, current_x:] = COLOR_MAP["bedrock"]

    def world_border(self, arr, stageName):
        h, w, _ = arr.shape
        for ix in range(BORDER_WIDTH):
            self.report(stageName, ix, BORDER_WIDTH)
            arr[:, ix] = COLOR_MAP["bedrock"]
            arr[:, -ix-1] = COLOR_MAP["bedrock"]
        for iy in range(BORDER_WIDTH):
            local_index = BORDER_WIDTH + iy
            self.report(stageName, local_index, 2 * BORDER_WIDTH)
            arr[-iy-1, :] = COLOR_MAP["bedrock"]


class ColorSwatch(QFrame):
    from PyQt5.QtCore import pyqtSignal
    clicked = pyqtSignal(str)
    def __init__(self, color_name: str, rgb: Tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.color_name = color_name
        self.rgb = rgb
        self.setFixedSize(30, 30)
        tip = f"{color_name} ({rgb[0]},{rgb[1]},{rgb[2]})"
        self.setToolTip(tip)
        css_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        self.setStyleSheet(f"QFrame{{background-color:{css_color}; border:1px solid #000;}}")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.color_name)


class MapEditorDialog(QDialog):
    def __init__(self, parent,
                 map_array: np.ndarray,
                 fog_array: np.ndarray = None,
                 rock_base: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Map Editor")

        self.map_array = map_array
        self.height_map, self.width_map, _ = map_array.shape
        if fog_array is None:
            self.fog_array = np.zeros((self.height_map, self.width_map, 4), dtype=np.uint8)
        else:
            self.fog_array = fog_array
        self.rock_base = rock_base

        self.is_fog_mode = False
        self.zoom_factor = 1.0
        self.is_painting = False
        self.selected_color = None

        main_layout = QVBoxLayout(self)

        self.menu_bar = QMenuBar()
        main_layout.addWidget(self.menu_bar)

        layers_menu = self.menu_bar.addMenu("Layers")
        self.fog_action = QAction("Show Fog Layer", self, checkable=True)
        self.fog_action.triggered.connect(self.on_fog_toggled)
        layers_menu.addAction(self.fog_action)

        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        self.palette_widget = QWidget()
        self.palette_layout = QGridLayout(self.palette_widget)
        self.palette_widget.setLayout(self.palette_layout)
        top_layout.addWidget(self.palette_widget)

        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 50)
        self.brush_size_spin.setValue(5)
        top_layout.addWidget(self.brush_size_spin)

        self.scroll_area = QScrollArea()
        main_layout.addWidget(self.scroll_area, stretch=1)

        self.map_label = QLabel()
        self.scroll_area.setWidget(self.map_label)
        self.scroll_area.setWidgetResizable(True)

        self.build_map_palette()

        self.update_pixmap()
        self.map_label.installEventFilter(self)

    def on_fog_toggled(self, checked: bool):
        self.is_fog_mode = checked
        if self.is_fog_mode:
            self.build_fog_palette()
        else:
            self.build_map_palette()
        self.update_pixmap()

    def build_map_palette(self):
        for i in reversed(range(self.palette_layout.count())):
            w = self.palette_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        row = 0
        col = 0
        for cname, rgb in COLOR_MAP.items():
            swatch = ColorSwatch(cname, rgb)
            swatch.clicked.connect(self.on_swatch_clicked)
            self.palette_layout.addWidget(swatch, row, col)
            col += 1
            if col > 5:
                col = 0
                row += 1

    def build_fog_palette(self):
        for i in reversed(range(self.palette_layout.count())):
            w = self.palette_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.fog_colors = {
            "Transparent": (0, 0, 0, 0),
            "LightGray": (128, 128, 128, 255),
            "Gray": (80, 80, 80, 255),
            "DarkGray": (32, 32, 32, 255),
            "Black": (0, 0, 0, 255)
        }
        row = 0
        col = 0
        for cname, rgba in self.fog_colors.items():
            rgb = (rgba[0], rgba[1], rgba[2])
            swatch = ColorSwatch(cname, rgb)
            swatch.clicked.connect(self.on_swatch_clicked)
            self.palette_layout.addWidget(swatch, row, col)
            col += 1
            if col > 5:
                col = 0
                row += 1

    def on_swatch_clicked(self, color_name):
        self.selected_color = color_name
        logging.debug(f"Selected color: {color_name}")

    def update_pixmap(self):
        h, w, _ = self.map_array.shape
        if self.is_fog_mode:
            map_rgba = np.dstack([self.map_array, np.full((h, w), 255, dtype=np.uint8)])
            map_img = Image.fromarray(map_rgba, "RGBA")
            fog_img = Image.fromarray(self.fog_array, "RGBA")
            composed = Image.alpha_composite(map_img, fog_img)
            arr_composed = np.array(composed)
            arr_rgb = arr_composed[:, :, :3].astype(np.uint8)
            data_bytes = arr_rgb.tobytes()
            qimg = QImage(data_bytes, w, h, 3*w, QImage.Format_RGB888)
        else:
            arr_rgb = self.map_array.astype(np.uint8)
            data_bytes = arr_rgb.tobytes()
            qimg = QImage(data_bytes, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled_w = int(w * self.zoom_factor)
        scaled_h = int(h * self.zoom_factor)
        scaled_pix = pix.scaled(scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.map_label.setPixmap(scaled_pix)
        self.map_label.setFixedSize(scaled_w, scaled_h)

    def eventFilter(self, watched, event):
        if watched == self.map_label:
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.is_painting = True
                    self.paint_at(event.pos())
                    return True
            elif event.type() == event.MouseMove:
                if self.is_painting:
                    self.paint_at(event.pos())
                    return True
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.is_painting = False
                    return True
            elif event.type() == event.Wheel:
                if event.modifiers() & Qt.ShiftModifier:
                    delta = event.angleDelta().y()
                    if delta > 0:
                        self.zoom_factor *= 1.1
                    else:
                        self.zoom_factor /= 1.1
                    if self.zoom_factor < 0.1:
                        self.zoom_factor = 0.1
                    if self.zoom_factor > 8.0:
                        self.zoom_factor = 8.0
                    self.update_pixmap()
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def paint_at(self, pos):
        if not self.selected_color:
            return
        pix = self.map_label.pixmap()
        if not pix:
            return

        disp_w = pix.width()
        disp_h = pix.height()
        if disp_w <= 0 or disp_h <= 0:
            return

        scale_x = disp_w / self.width_map
        scale_y = disp_h / self.height_map
        x_img = int(pos.x() / scale_x)
        y_img = int(pos.y() / scale_y)
        if x_img < 0 or x_img >= self.width_map or y_img < 0 or y_img >= self.height_map:
            return

        brush_size = self.brush_size_spin.value()
        r_sq = brush_size * brush_size
        y_min = max(0, y_img - brush_size)
        y_max = min(self.height_map - 1, y_img + brush_size)
        x_min = max(0, x_img - brush_size)
        x_max = min(self.width_map - 1, x_img + brush_size)

        if self.is_fog_mode:
            fog_colors = {
                "Transparent": (0, 0, 0, 0),
                "LightGray": (128, 128, 128, 255),
                "Gray": (80, 80, 80, 255),
                "DarkGray": (32, 32, 32, 255),
                "Black": (0, 0, 0, 255)
            }
            rgba = fog_colors.get(self.selected_color, (0, 0, 0, 0))
            for yy in range(y_min, y_max + 1):
                dy = yy - y_img
                for xx in range(x_min, x_max + 1):
                    dx = xx - x_img
                    if dx * dx + dy * dy <= r_sq:
                        self.fog_array[yy, xx] = rgba
        else:
            color_rgb = COLOR_MAP.get(self.selected_color, (255, 255, 255))
            for yy in range(y_min, y_max + 1):
                dy = yy - y_img
                for xx in range(x_min, x_max + 1):
                    dx = xx - x_img
                    if dx * dx + dy * dy <= r_sq:
                        self.map_array[yy, xx] = color_rgb

        self.update_pixmap()


class MapGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Generator")
        self.resize(1000, 800)

        self.last_map_array = None
        self.fog_array = None

        self.current_size = 1280
        self.current_rock_base = 675
        self.water_enabled = True
        self.water_threshold = 0.005
        self.water_freq = 0.005
        self.fluxite_enabled = True
        self.fluxite_threshold = 0.7
        self.fluxite_freq = 0.0133
        self.plant_enabled = True
        self.plant_threshold = 0.6
        self.plant_freq = 0.0167
        self.caves_enabled = True
        self.cave_threshold = 0.3
        self.cave_freq = 0.01
        self.current_seed_str = ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=0)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box, stretch=1)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate_clicked)
        left_layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        right_frame = QFrame()
        main_layout.addWidget(right_frame, stretch=1)

        frame_layout = QVBoxLayout(right_frame)
        self.preview_label = QLabel("Generated map will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.preview_label)

        self.worker = None

        self.create_menubar()

    def create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.on_save_as)
        file_menu.addAction(save_as_action)

        options_menu = menubar.addMenu("Options")

        params_action = QAction("Map Parameters...", self)
        params_action.triggered.connect(self.on_map_params)
        options_menu.addAction(params_action)

        edit_action = QAction("Manual Edit", self)
        edit_action.triggered.connect(self.on_manual_edit)
        options_menu.addAction(edit_action)

        self.dark_mode_action = QAction("Dark Mode", self, checkable=True)
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        options_menu.addAction(self.dark_mode_action)

    def on_map_params(self):
        dlg = MapParamsDialog(
            parent=self,
            current_size=self.current_size,
            current_rockbase=self.current_rock_base,
            water_enabled=self.water_enabled,
            water_thresh=self.water_threshold,
            water_freq=self.water_freq,
            fluxite_enabled=self.fluxite_enabled,
            fluxite_thresh=self.fluxite_threshold,
            fluxite_freq=self.fluxite_freq,
            plant_enabled=self.plant_enabled,
            plant_thresh=self.plant_threshold,
            plant_freq=self.plant_freq,
            caves_enabled=self.caves_enabled,
            cave_thresh=self.cave_threshold,
            cave_freq=self.cave_freq,
            current_seed=self.current_seed_str
        )
        if dlg.exec_() == QDialog.Accepted:
            self.current_size = dlg.size
            self.current_rock_base = dlg.rock_base
            self.water_enabled = dlg.water_enabled
            self.water_threshold = dlg.water_threshold
            self.water_freq = dlg.water_freq
            self.fluxite_enabled = dlg.fluxite_enabled
            self.fluxite_threshold = dlg.fluxite_threshold
            self.fluxite_freq = dlg.fluxite_freq
            self.plant_enabled = dlg.plant_enabled
            self.plant_threshold = dlg.plant_threshold
            self.plant_freq = dlg.plant_freq
            self.caves_enabled = dlg.caves_enabled
            self.cave_threshold = dlg.cave_threshold
            self.cave_freq = dlg.cave_freq
            self.current_seed_str = dlg.seed_string

            self.log_box.appendPlainText(
                f"Parameters updated:\n"
                f"  Size={self.current_size}, RockBase={self.current_rock_base}\n"
                f"  Water={self.water_enabled}, WaterThreshold={self.water_threshold}, WaterFreq={self.water_freq}\n"
                f"  Fluxite={self.fluxite_enabled}, FluxiteThreshold={self.fluxite_threshold}, FluxiteFreq={self.fluxite_freq}\n"
                f"  Plant={self.plant_enabled}, PlantThreshold={self.plant_threshold}, PlantFreq={self.plant_freq}\n"
                f"  Caves={self.caves_enabled}, CaveThreshold={self.cave_threshold}, CaveFreq={self.cave_freq}\n"
                f"  Seed={self.current_seed_str or 'Random'}"
            )
        else:
            self.log_box.appendPlainText("Map Parameters dialog canceled.")

    def on_generate_clicked(self):
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.generate_button.setEnabled(False)

        size = self.current_size
        rock_base = self.current_rock_base
        if rock_base >= size:
            rock_base = size - 1
            self.current_rock_base = rock_base

        self.worker = GenerationWorker(
            size, rock_base,
            self.water_enabled, self.water_threshold, self.water_freq,
            self.fluxite_enabled, self.fluxite_threshold, self.fluxite_freq,
            self.plant_enabled, self.plant_threshold, self.plant_freq,
            self.caves_enabled, self.cave_threshold, self.cave_freq,
            self.current_seed_str
        )
        self.worker.progressMsg.connect(self.on_progress_message)
        self.worker.progressVal.connect(self.on_progress_value)
        self.worker.resultReady.connect(self.on_generation_finished)
        self.worker.start()

    def on_progress_message(self, msg: str):
        self.log_box.appendPlainText(msg)

    def on_progress_value(self, val: int):
        self.progress_bar.setValue(val)

    def on_generation_finished(self, map_array):
        self.log_box.appendPlainText("Map generation complete!")
        self.progress_bar.setValue(100)
        self.generate_button.setEnabled(True)

        if map_array is not None:
            self.last_map_array = map_array
            h, w, _ = map_array.shape
            self.fog_array = np.zeros((h, w, 4), dtype=np.uint8)
            qimg = QImage(map_array.tobytes(), w, h, 3*w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(
                pix.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        else:
            self.log_box.appendPlainText("No map array generated (None).")

    def on_manual_edit(self):
        if self.last_map_array is None:
            QMessageBox.warning(self, "No Map", "No map generated yet!")
            return
        dlg = MapEditorDialog(self, self.last_map_array, self.fog_array, self.current_rock_base)
        dlg.exec_()
        if self.last_map_array is not None:
            h, w, _ = self.last_map_array.shape
            qimg = QImage(self.last_map_array.tobytes(), w, h, 3*w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(
                pix.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def on_save_as(self):
        if self.last_map_array is None:
            QMessageBox.warning(self, "No Map", "No map to save!")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Map As", "", "PNG Files (*.png);;All Files (*)"
        )
        if file_name:
            try:
                h, w, _ = self.last_map_array.shape
                img = Image.fromarray(self.last_map_array)
                img.save(file_name)
                QMessageBox.information(self, "Saved", f"Map saved to: {file_name}")
                if self.fog_array is not None:
                    fog_path = os.path.join(os.path.dirname(file_name), "fog_playtest.png")
                    fog_img = Image.fromarray(self.fog_array, "RGBA")
                    fog_img.save(fog_path)
                    QMessageBox.information(self, "Fog Saved", f"Fog layer saved to: {fog_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")

    def toggle_dark_mode(self):
        app = QApplication.instance()
        if self.dark_mode_action.isChecked():
            if qdarkstyle is not None:
                app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            else:
                app.setStyleSheet("""
                    QMainWindow,QWidget{
                        background-color:#2b2b2b;
                        color:#ffffff;
                    }
                    QMenuBar,QMenu{
                        background-color:#2b2b2b;
                        color:#ffffff;
                    }
                    QMenu::item:selected{
                        background-color:#444444;
                    }
                    QFrame{
                        background-color:#333333;
                    }
                    QPlainTextEdit{
                        background-color:#3b3b3b;
                        color:#ffffff;
                        border:none;
                    }
                    QPushButton{
                        background-color:#444444;
                        color:#ffffff;
                    }
                    QLabel{
                        color:#ffffff;
                    }
                    QProgressBar{
                        background-color:#3b3b3b;
                        color:#ffffff;
                    }
                """)
        else:
            app.setStyleSheet("")

    def create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.on_save_as)
        file_menu.addAction(save_as_action)

        options_menu = menubar.addMenu("Options")

        params_action = QAction("Map Parameters...", self)
        params_action.triggered.connect(self.on_map_params)
        options_menu.addAction(params_action)

        edit_action = QAction("Manual Edit", self)
        edit_action.triggered.connect(self.on_manual_edit)
        options_menu.addAction(edit_action)

        self.dark_mode_action = QAction("Dark Mode", self, checkable=True)
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        options_menu.addAction(self.dark_mode_action)


def main():
    app = QApplication(sys.argv)
    window = MapGeneratorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
