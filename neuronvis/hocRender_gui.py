"""
hocRender_gui : a single-window GUI front end for hocRender.

The left pane is a pyqtgraph ParameterTree holding every hocRender command line
option plus a file chooser; the right pane holds the rendered morphology.
A triangle mesh (.obj, e.g. from SyGlass) can be shown alone or on top of the
morphology (pyqtgraph and vispy renderers). Imaris filament tracings (.ims) are
converted to SWC (see imaris_reader) and rendered like an swc file.

Every morphology is checked with topology_check. In the pyqtgraph view,
pieces not connected to the main tree are drawn in red with red arrows
pointing at their ends nearest the tree, gaps at connections get orange
arrows, and hovering over the cell shows the section and segment under the
mouse (the part nearest the viewer; see section_picker).

Usage:
    uv run python -m neuronvis.hocRender_gui [optional morphology or .obj mesh file]
Parameter values and window geometry are restored from the previous session.

The command line tool (hocRender) is unchanged and still available.

Renderers:
    pyqtgraph : embedded GLViewWidget (HocViewer)
    mpl       : embedded matplotlib 3D axes (slow for large cells)
    vispy     : embedded vispy SceneCanvas

"Reload source" (at the bottom of the parameter list) uses pyqtgraph's reload
machinery to re-import any neuronvis source files that changed since they were
loaded, then re-renders the current file. Changes to this module itself need a
restart. It is disabled in frozen (PyInstaller) builds, which have no source
files to reload.
"""

import argparse
import datetime
import gc
import html
import json
import os
import sys
import tempfile
import time
import traceback
from math import radians, tan
from pathlib import Path
from typing import Union

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.reload
from OpenGL import GL
from pyqtgraph import opengl as opengl
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pylibrary.tools.cprint import cprint as CP
from PyQt6 import sip
from scipy.spatial import cKDTree

import neuron
import neuronvis
import neuronvis.renderer_colormaps as rc
from neuronvis import (
    hoc_reader,
    hoc_viewer,
    imaris_reader,
    mesh_tools,
    obj_reader,
    section_picker,
    swc_sectypes,
    topology_check,
)

# Classes and tables from other neuronvis modules are looked up through their
# module at render time (hoc_viewer.HocViewer, rc.section_colors, ...) rather
# than imported by name, so that "Reload source" picks up the new versions.

renderers = ["pyqtgraph", "mpl", "vispy"]
styles = {  # styles each renderer can draw
    "pyqtgraph": ["cylinders", "graph", "surface"],
    "mpl": ["cylinders", "graph"],
    "vispy": ["cylinders"],
}
modes = ["sec-type", "mechanism", "vm"]
morphology_filter = "Morphology (*.swc *.hoc *.hocx *.p *.ims);;All files (*)"
points_filter = "Points (*.csv);;All files (*)"
mesh_filter = "Mesh (*.obj);;All files (*)"
image_filter = "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif)"
disconnected_color = (1.0, 0.0, 0.0, 1.0)
fragment_color = (0.6, 0.15, 0.15, 1.0)  # darker, so that the red arrows stand out
gap_color = (1.0, 0.6, 0.0, 1.0)
highlight_color = (1.0, 1.0, 0.0, 1.0)
warning_color = "#ff5050"  # info panel and status bar text for a morphology/mesh size mismatch
size_mismatch = 5.0  # warn when the morphology and the mesh differ in size by more than this factor
smoothing_iterations = {"Taubin": 20, "GAMer": 6}  # put in Mesh > Iterations when a method is chosen
# drawn over everything, so that markers on hidden parts are still seen
on_top = {
    GL.GL_DEPTH_TEST: False,
    GL.GL_BLEND: True,
    "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
}
settings_org = "neuronvis"
settings_app = "hocRenderGUI"


def make_embedded_viewer(*args, **kwds) -> object:
    """
    HocViewer calls self.show() in its constructor, which would briefly open it
    as a separate top-level window. Suppress that until it has been placed in
    the main window's layout.
    The subclass is built from the current hoc_viewer.HocViewer each time:
    HocViewer.__init__ calls super(HocViewer, self), which after a reload names
    the new class, so a subclass of the old class would fail.
    """

    class EmbeddedHocViewer(hoc_viewer.HocViewer):
        def show(self) -> None:
            if self.parent() is None:
                return
            super().show()

    return EmbeddedHocViewer(*args, **kwds)


def clear_neuron_sections() -> None:
    """
    NEURON keeps all sections in one global namespace; remove them so that a
    new file does not render on top of the previous one.
    """
    neuron.h("forall delete_section()")


def force_load_hoc(filename: Path, verbose: bool = False) -> None:
    """
    h.load_file() will not load a file a second time, so a re-render of the
    same .hoc/.hocx file would come up empty after clear_neuron_sections().
    Force the load here; HocReader's own load_file() call then finds the file
    already loaded and goes on to its centering step.
    """
    if not verbose:
        neuron.h.hoc_stdout(os.devnull)
    try:
        success = neuron.h.load_file(1, str(filename))
    finally:
        if not verbose:
            neuron.h.hoc_stdout()
    if success == 0:
        raise NameError(f"NEURON failed to load: {str(filename):s}")


class HocRenderWindow(QtWidgets.QMainWindow):
    def __init__(self, filename: Union[str, Path, None] = None) -> None:
        super().__init__()
        self.setWindowTitle("hocRender")
        self.resize(1400, 900)
        self.viewer = None  # widget currently in the right pane
        self.viewer_kind = None  # renderer that made it
        self.vispy_graphic = None  # vispy_Cylinders keeps its own canvas + timer
        self.vispy_canvas = None
        self.vispy_view = None
        self.mpl_figure = None
        self.hoc = None
        self.bounds = None  # (min xyz, max xyz) of the rendered cell
        self.topology_report = None  # topology_check.TopologyReport of the last morphology
        self.topology_warned = None  # (file, errors) last shown in a dialog
        self.swc_nodes = None  # SWC nodes of the morphology file, in rendered coordinates
        self.conversion_notes = []  # repairs made converting the morphology (Imaris joins)
        self.file_first_points = {}  # first 3D point of each section as read, before define_shape()
        self.layers = {}  # "morphology", "mesh", "points", "grid": drawn items, for show/hide
        self.extents = {}  # same keys: points the camera is fitted to
        self.transform_lines = []  # every transform applied to what is shown, for the user
        self.measurements = {}  # areas and positions of the reconstruction and the mesh
        self.radius_confirmed = None  # (file, x/y/z scales, radius scale) the user chose
        self.imaris_provenance = []
        self.info_label = None
        self.short = {"morphology": [], "mesh": []}  # one-line transform summaries for the info panel
        # (key, (area, boundary edges, non-manifold edges)); key: (path, mtime, smoothing method, iterations)
        self.mesh_stats_cache = None
        self.section_shift = {}  # how far define_shape() moved each section
        self.picker = None  # section_picker.SegmentPicker for the pyqtgraph view
        self.pick_sections = None  # section name of each picker edge
        self.hover_items = None  # highlight line and point
        self.hover_label = None
        self.hover_pos = None
        self.hover_timer = QtCore.QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self.hover_pick)
        self.mesh_cache = None  # (path, mtime, vertices, faces) of the last mesh read
        self.mesh_merge = None  # (path, mtime, info, lines): the seam merge done when it was read
        self.smoothed = None  # the last smoothing: dict(source, method, iterations, vertices, faces, ...)
        self.smooth_process = None  # QProcess running GAMer
        self.updating_filament_choices = False

        self.params = Parameter.create(name="params", type="group", children=self.param_spec())
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.params, showTop=False)
        self.tree.setMinimumWidth(320)

        self.viewer_host = QtWidgets.QWidget()
        self.viewer_layout = QtWidgets.QVBoxLayout(self.viewer_host)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QtWidgets.QLabel("Choose a morphology or mesh file to render.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.viewer_layout.addWidget(self.placeholder)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.viewer_host)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([360, 1040])
        self.setCentralWidget(self.splitter)
        self.statusBar().showMessage("Ready")
        # progress of GAMer smoothing, at the right of the status bar while it runs and when done
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimumWidth(320)
        self.progress_bar.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

        self.restore_settings()
        if getattr(sys, "frozen", False):
            self.params.child("Development", "Reload source").setOpts(enabled=False)
        self.update_style_choices()
        self.connect_params()

        if filename is not None:
            input_param = "Mesh file" if Path(filename).suffix == ".obj" else "Morphology file"
            self.params.child("Input", input_param).setValue(str(Path(filename).resolve()))
            # wait for the event loop so that vispy/matplotlib see a running application
            QtCore.QTimer.singleShot(0, self.render)

    def param_spec(self) -> list:
        return [
            dict(
                name="Input",
                type="group",
                children=[
                    dict(
                        name="Morphology file",
                        type="file",
                        value="",
                        nameFilter=morphology_filter,
                        fileMode="ExistingFile",
                        tip=".swc, .hoc, .hocx, Imaris .ims (filaments), or a .p simulation result",
                    ),
                    dict(
                        name="Imaris filaments",
                        type="checklist",
                        limits=[],
                        value=[],
                        exclusive=False,
                        visible=False,
                        tip="Filament objects in the .ims file to convert and show",
                    ),
                    dict(
                        name="Points file",
                        type="file",
                        value="",
                        nameFilter=points_filter,
                        fileMode="ExistingFile",
                        tip="csv with x, y, z, color columns (--points)",
                    ),
                    dict(name="Clear points", type="action"),
                    dict(
                        name="Mesh file",
                        type="file",
                        value="",
                        nameFilter=mesh_filter,
                        fileMode="ExistingFile",
                        tip="Triangle mesh (.obj, e.g. from SyGlass), in the same units as the morphology",
                    ),
                    dict(name="Clear mesh", type="action"),
                    dict(
                        name="Render on file select",
                        type="bool",
                        value=True,
                        tip="Render as soon as a morphology or mesh file is chosen",
                    ),
                ],
            ),
            dict(
                name="Display",
                type="group",
                children=[
                    dict(name="Renderer", type="list", limits=renderers, value="pyqtgraph"),
                    dict(name="Style", type="list", limits=styles["pyqtgraph"], value="cylinders"),
                    dict(name="Mode", type="list", limits=modes, value="sec-type"),
                    dict(
                        name="Mechanism",
                        type="str",
                        value="None",
                        tip="Mechanism density to display in 'mechanism' mode",
                    ),
                    dict(
                        name="Section map",
                        type="list",
                        limits=list(swc_sectypes.sectypes.keys()),
                        value="sbem3",
                    ),
                    dict(name="Alpha", type="float", value=0.45, limits=(0.0, 1.0), step=0.05),
                ],
            ),
            dict(
                name="Mesh",
                type="group",
                children=[
                    dict(name="Mesh color", type="color", value="#6fa8dc"),
                    dict(
                        name="Mesh alpha",
                        type="float",
                        value=0.35,
                        limits=(0.0, 1.0),
                        step=0.05,
                        tip="Opacity; below 1 the morphology shows through the mesh",
                    ),
                    dict(
                        name="Smoothing",
                        type="list",
                        limits=["none", "Taubin", "GAMer"],
                        value="none",
                        tip="Taubin: removes the voxel staircase, keeps the volume (fast). "
                        "GAMer: GAMer2 smooth(), improves triangle shape (needs PyGAMer; minutes). "
                        "Applied with 'Smooth mesh'; the file is not changed.",
                    ),
                    dict(
                        name="Iterations",
                        type="int",
                        value=20,
                        limits=(1, 500),
                        tip="Taubin: 20-50 removes the staircase (more starts to change the volume); GAMer: 6 is its default",
                    ),
                    dict(name="Smooth mesh", type="action"),
                    dict(name="Save smoothed mesh...", type="action"),
                ],
            ),
            dict(
                name="Show",
                type="group",
                children=[
                    dict(name="Morphology", type="bool", value=True),
                    dict(name="Mesh", type="bool", value=True),
                    dict(name="Points", type="bool", value=True),
                    dict(name="Grid and axes", type="bool", value=True),
                    dict(
                        name="Info panel",
                        type="bool",
                        value=True,
                        tip="Transforms applied, areas and positions, over the view",
                    ),
                ],
            ),
            dict(
                name="Inspect",
                type="group",
                children=[
                    dict(
                        name="Hover info",
                        type="bool",
                        value=True,
                        tip="Show the section and segment under the mouse (pyqtgraph renderer)",
                    ),
                    dict(
                        name="Mark disconnections",
                        type="bool",
                        value=True,
                        tip="Red: pieces not connected to the main tree, with arrows at their nearest ends",
                    ),
                    dict(
                        name="Mark gaps",
                        type="bool",
                        value=True,
                        tip="Orange arrows: children whose first point is away from where they attach",
                    ),
                ],
            ),
            dict(
                name="Geometry",
                type="group",
                children=[
                    dict(name="Center", type="bool", value=False, tip="Move first point to (0,0,0)"),
                    dict(name="Scale X", type="float", value=1.0, step=0.001, decimals=6),
                    dict(name="Scale Y", type="float", value=1.0, step=0.001, decimals=6),
                    dict(name="Scale Z", type="float", value=1.0, step=0.001, decimals=6),
                    dict(name="Scale radius", type="float", value=1.0, step=0.001, decimals=6),
                ],
            ),
            dict(
                name="Diagnostics",
                type="group",
                children=[
                    dict(name="Verify", type="bool", value=False, tip="Print hoc generated from swc"),
                    dict(name="Verbose", type="bool", value=False),
                ],
            ),
            dict(
                name="Actions",
                type="group",
                children=[
                    dict(name="Render", type="action"),
                    dict(name="Reset view", type="action", tip="Fit the camera to the cell"),
                    dict(name="Save image...", type="action"),
                    dict(
                        name="Report...",
                        type="action",
                        tip="Sources, every transform applied, areas, and the connectivity check",
                    ),
                    dict(
                        name="Export SWC...",
                        type="action",
                        tip="Save the selected Imaris filaments as an SWC file",
                    ),
                ],
            ),
            dict(
                name="Development",
                type="group",
                children=[
                    dict(
                        name="Reload source",
                        type="action",
                        tip="Reload changed neuronvis source files and re-render",
                    ),
                ],
            ),
        ]

    def connect_params(self) -> None:
        self.params.child("Input", "Morphology file").sigValueChanged.connect(self.morphology_selected)
        self.params.child("Input", "Imaris filaments").sigValueChanged.connect(self.filaments_changed)
        self.params.child("Input", "Clear points").sigActivated.connect(self.clear_points)
        self.params.child("Input", "Mesh file").sigValueChanged.connect(self.file_selected)
        self.params.child("Input", "Clear mesh").sigActivated.connect(self.clear_mesh)
        self.params.child("Display", "Renderer").sigValueChanged.connect(self.update_style_choices)
        self.params.child("Mesh", "Smoothing").sigValueChanged.connect(self.smoothing_changed)
        self.params.child("Mesh", "Smooth mesh").sigActivated.connect(self.smooth_mesh)
        self.params.child("Mesh", "Save smoothed mesh...").sigActivated.connect(self.save_smoothed_mesh)
        self.params.child("Actions", "Render").sigActivated.connect(self.render)
        self.params.child("Actions", "Reset view").sigActivated.connect(self.reset_view)
        self.params.child("Actions", "Save image...").sigActivated.connect(self.save_image)
        self.params.child("Actions", "Export SWC...").sigActivated.connect(self.export_swc)
        self.params.child("Actions", "Report...").sigActivated.connect(self.show_topology_report)
        for name in ["Morphology", "Mesh", "Points", "Grid and axes", "Info panel"]:
            self.params.child("Show", name).sigValueChanged.connect(self.apply_visibility)
        self.params.child("Inspect", "Mark disconnections").sigValueChanged.connect(self.rerender)
        self.params.child("Inspect", "Mark gaps").sigValueChanged.connect(self.rerender)
        self.params.child("Inspect", "Hover info").sigValueChanged.connect(self.hide_hover)
        self.params.child("Development", "Reload source").sigActivated.connect(self.reload_source)

    # ------------------------------------------------------------------
    # parameter handling
    # ------------------------------------------------------------------

    def value(self, group: str, name: str):
        return self.params.child(group, name).value()

    def file_selected(self, param: Parameter) -> None:
        if self.value("Input", "Render on file select") and param.value():
            self.render()

    def morphology_selected(self, param: Parameter) -> None:
        self.update_filament_choices()
        self.file_selected(param)

    def filaments_changed(self, param: Parameter) -> None:
        if not self.updating_filament_choices:
            self.file_selected(param)

    def update_filament_choices(self, preferred: Union[list, None] = None) -> None:
        """
        Show the filament objects of the chosen .ims file (hidden for other
        files); keep the *preferred* ones selected, or the typed ones by default.
        """
        param = self.params.child("Input", "Imaris filaments")
        filename = self.value("Input", "Morphology file")
        names = []
        if filename and Path(filename).suffix == ".ims" and Path(filename).is_file():
            try:
                names = imaris_reader.list_filament_objects(filename)
            except Exception as exc:
                self.statusBar().showMessage(f"Could not read filaments: {exc!s}")
        selected = [n for n in (preferred or []) if n in names] or imaris_reader.default_objects(names)
        self.updating_filament_choices = True
        try:
            param.setOpts(visible=len(names) > 0)
            param.setLimits(names)
            param.setValue(selected)
            # pyqtgraph 0.14: ChecklistParameter.setValue updates its child
            # parameters but not their checkboxes; re-announce the child values
            for child in param.children():
                child.sigValueChanged.emit(child, child.value())
        finally:
            self.updating_filament_choices = False

    def rerender(self) -> None:
        if self.viewer is not None:
            self.render()

    def clear_points(self) -> None:
        self.params.child("Input", "Points file").setValue("")

    def clear_mesh(self) -> None:
        self.params.child("Input", "Mesh file").setValue("")

    def update_style_choices(self) -> None:
        style = self.params.child("Display", "Style")
        allowed = styles[self.value("Display", "Renderer")]
        current = style.value()
        style.setLimits(allowed)
        style.setValue(current if current in allowed else allowed[0])

    def param_values(self) -> dict:
        """values of all non-action parameters, keyed by "group/name" """
        values = {}
        for group in self.params.children():
            for param in group.children():
                if param.type() == "action":
                    continue
                value = param.value()
                if isinstance(value, QtGui.QColor):  # not JSON serializable
                    value = value.name(QtGui.QColor.NameFormat.HexRgb)
                values[f"{group.name():s}/{param.name():s}"] = value
        return values

    def restore_settings(self) -> None:
        """restore the parameter values and window geometry from the last session"""
        settings = QtCore.QSettings(settings_org, settings_app)
        values = settings.value("params", None)
        if values is not None:
            try:
                values = json.loads(values)
            except Exception:
                values = {}
                CP("y", "hocRenderGUI: could not read saved settings, using defaults")
            filaments = values.pop("Input/Imaris filaments", None)
            for key, value in values.items():
                try:
                    self.params.child(*key.split("/")).setValue(value)
                except Exception:
                    CP("y", f"hocRenderGUI: could not restore setting {key:s}")
            # the filament choices depend on the restored morphology file
            self.update_filament_choices(preferred=filaments)
        geometry = settings.value("geometry", None)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def save_settings(self) -> None:
        settings = QtCore.QSettings(settings_org, settings_app)
        settings.setValue("params", json.dumps(self.param_values()))
        settings.setValue("geometry", self.saveGeometry())
        settings.sync()  # main() ends with os._exit, so write now

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_settings()
        self.clear_viewer()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        morphology_file = self.value("Input", "Morphology file")
        mesh_file = self.value("Input", "Mesh file")
        if not morphology_file and not mesh_file:
            self.statusBar().showMessage("No morphology or mesh file selected")
            return
        names = " + ".join(Path(f).name for f in (morphology_file, mesh_file) if f)
        self.statusBar().setStyleSheet("")  # a size-mismatch warning from the last render is red
        self.statusBar().showMessage(f"Rendering {names:s} ...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            self.render_files(
                Path(morphology_file) if morphology_file else None,
                Path(mesh_file) if mesh_file else None,
            )
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"Render failed: {exc!s}")
            QtWidgets.QMessageBox.critical(self, "Render failed", traceback.format_exc())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def render_files(self, morphology_file: Union[Path, None], mesh_file: Union[Path, None]) -> None:
        notes = []
        renderer = self.value("Display", "Renderer")
        style = self.value("Display", "Style")
        mode = self.value("Display", "Mode")
        if mode == "vm":
            notes.append("vm animation is not implemented; showing section types")
            mode = "sec-type"
        mechanism = self.value("Display", "Mechanism")
        if mechanism in ["", "None"]:
            mechanism = None
        alpha = self.value("Display", "Alpha")

        self.clear_viewer()
        self.hoc = None
        self.transform_lines = []
        self.measurements = {}
        self.short = {"morphology": [], "mesh": []}
        offset = np.zeros(3)  # centering shift applied to the morphology
        if morphology_file is not None:
            self.load_morphology(morphology_file, notes)
            offset = np.array([self.hoc.centerpos[k] for k in "xyz"])
        shifted = f"translated by ({-offset[0]:.4f}, {-offset[1]:.4f}, {-offset[2]:.4f}) um, with the morphology"
        points = self.read_points(offset)
        if points is not None:
            self.transform_lines += [f"Points: {self.value('Input', 'Points file')!s}"]
            self.transform_lines += [f"  {shifted:s}" if np.any(offset != 0) else "  as in the file"]
        mesh = None
        if mesh_file is not None:
            vertices, faces = self.read_mesh(mesh_file)
            self.transform_lines += [f"Mesh: {mesh_file.resolve()!s}", "  as in the file (no scaling)"]
            self.transform_lines += [f"  {line:s}" for line in self.mesh_merge[3]]
            self.short["mesh"].append("as in the file (no scaling)")
            if self.mesh_merge[2]["merged"] > 0:
                self.short["mesh"].append(f"seams merged ({self.mesh_merge[2]['merged']:d} duplicated vertices)")
            smoothing = self.current_smoothing(mesh_file)
            if smoothing is not None:
                vertices, faces = smoothing["vertices"], smoothing["faces"]
                self.transform_lines.append(f"  smoothed: {smoothing['description']:s}")
                self.short["mesh"].append(f"smoothed: {smoothing['method']:s} x{smoothing['iterations']:d}")
            elif self.smoothed is not None and self.smoothed["source"][0] == mesh_file:
                notes.append("mesh not smoothed with the current settings: press Smooth mesh")
            mesh = (vertices - offset.astype(np.float32), faces)
            if np.any(offset != 0):
                self.transform_lines.append(f"  {shifted:s}")
                self.short["mesh"].append("shifted with the morphology (Center)")

        match renderer:
            case "pyqtgraph":
                self.render_pyqtgraph(style, mode, mechanism, alpha, points, mesh, notes)
            case "mpl":
                if self.hoc is None:
                    raise ValueError("The mpl renderer needs a morphology file; use pyqtgraph or vispy for a mesh alone.")
                if mesh is not None:
                    notes.append("mesh not drawn with mpl (too many triangles)")
                self.render_mpl(style, points)
            case "vispy":
                title = morphology_file.name if morphology_file is not None else mesh_file.name
                self.render_vispy(mechanism, title, mesh)

        self.measure(mesh)
        self.show_info()
        self.apply_visibility()
        names = " + ".join(f.name for f in (morphology_file, mesh_file) if f is not None)
        self.setWindowTitle(f"hocRender - {names:s}")
        msg = f"Rendered {names:s} ({renderer:s}"
        msg += f", {style:s}, {mode:s})" if self.hoc is not None else ")"
        if len(notes) > 0:
            msg += "  [" + "; ".join(notes) + "]"
        warning = self.measurements.get("size_warning")
        if warning is not None:
            msg = warning + "  |  " + msg
        self.statusBar().setStyleSheet(f"QStatusBar {{ color: {warning_color:s}; }}" if warning is not None else "")
        self.statusBar().showMessage(msg)

    def load_morphology(self, filename: Path, notes: list) -> None:
        """read a .swc, .hoc, .hocx, .ims or .p file into NEURON and self.hoc"""
        self.conversion_notes = []
        hoc_file = filename
        section_map = self.value("Display", "Section map")
        if filename.suffix == ".p":
            from neuronvis.sim_result import SimulationResult

            sim_data = SimulationResult(str(filename))
            hoc_file = Path(sim_data.hoc_file)
        elif filename.suffix == ".ims":
            hoc_file = self.convert_imaris(filename, notes)
            section_map = "swc"  # imaris_reader writes standard SWC types
        elif filename.suffix not in [".hoc", ".hocx", ".swc"]:
            raise ValueError(
                "Morphology file must be a hoc, hocx, swc, ims or p file (choose .obj meshes with 'Mesh file')."
            )
        verbose = self.value("Diagnostics", "Verbose")
        self.topology_report = None
        self.swc_nodes = None
        node_report = None
        if hoc_file.suffix == ".swc":
            node_report = topology_check.check_swc(hoc_file, source=filename.name)
        clear_neuron_sections()
        if hoc_file.suffix in [".hoc", ".hocx"]:
            force_load_hoc(hoc_file, verbose=verbose)
        xyz_scale = [self.value("Geometry", f"Scale {a:s}") for a in "XYZ"]
        r_scale = self.value("Geometry", "Scale radius")
        radius_note = "set in Geometry"
        if r_scale == 1.0 and any(v != 1.0 for v in xyz_scale) and hoc_file.suffix == ".swc":
            # no assumption: x, y, z are scaled, so ask whether the radii are in the same units
            key = (str(hoc_file), tuple(xyz_scale))
            if self.radius_confirmed is not None and self.radius_confirmed[:2] == key:
                r_scale = self.radius_confirmed[2]
            else:
                r_scale = self.ask_radius_scale(xyz_scale)
                if r_scale is None:
                    raise ValueError("Cancelled: choose a radius scale (Geometry > Scale radius)")
                self.radius_confirmed = (*key, r_scale)
            radius_note = "chosen when asked"
            # Claude fixed 2026-11-18: show the value in use in the field, not 1
            self.params.child("Geometry", "Scale radius").setValue(r_scale)
        self.transform_lines.append(f"Morphology: {filename.resolve()!s}")
        if filename.suffix == ".ims":
            self.transform_lines += [f"  {line:s}" for line in self.imaris_provenance]
            self.transform_lines.append(f"  converted SWC: {hoc_file!s}")
        self.hoc = hoc_reader.HocReader(
            str(hoc_file),
            section_map=section_map,
            center=self.value("Geometry", "Center"),
            scale={
                "x": xyz_scale[0],
                "y": xyz_scale[1],
                "z": xyz_scale[2],
                "r": r_scale,
            },
            verify=self.value("Diagnostics", "Verify"),
            verbose=verbose,
        )
        if sum(1 for _ in neuron.h.allsec()) == 0:
            # NEURON's own error message is hidden while HocReader loads the file
            raise ValueError(
                f"No sections were created from {hoc_file.name:s} with section map "
                f"'{section_map:s}' (invalid HOC?). "
                "Turn on Verbose to see NEURON's messages."
            )
        if hoc_file.suffix == ".swc":
            self.read_swc_nodes(hoc_file, r_scale)
            self.transform_lines.append(f"  radius scale {r_scale:g} ({radius_note:s})")
            self.transform_lines += [f"  {line:s}" for line in self.hoc.provenance]
        else:
            self.transform_lines += [f"  file header: {line:s}" for line in hoc_header(hoc_file)]
        self.transform_lines += [f"  {line:s}" for line in self.hoc.transforms]
        if hoc_file.suffix == ".swc" and any(v != 1.0 for v in xyz_scale + [r_scale]):
            self.short["morphology"].append(
                f"scaled x {xyz_scale[0]:g}, y {xyz_scale[1]:g}, z {xyz_scale[2]:g}, radius {r_scale:g} ({radius_note:s})"
            )
        else:
            self.short["morphology"].append("not scaled")
        if any(v != 0 for v in self.hoc.centerpos.values()):
            c = self.hoc.centerpos
            self.short["morphology"].append(f"centred: shifted by ({-c['x']:.2f}, {-c['y']:.2f}, {-c['z']:.2f}) um")
        if filename.suffix == ".ims" and len(self.conversion_notes) > 0:
            self.short["morphology"].append(f"Imaris: {len(self.conversion_notes):d} piece(s) joined")
        self.file_first_points = {
            sec.name(): np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)]) for sec in neuron.h.allsec() if sec.n3d() > 0
        }
        self.section_shift = {}
        self.check_topology(filename, node_report, notes)

    def swc_to_rendered(self, xyz: np.ndarray) -> np.ndarray:
        """SWC file coordinates to rendered ones: HocReader scales, then centers"""
        scale = np.array([self.value("Geometry", f"Scale {a:s}") for a in "XYZ"])
        center = np.array([self.hoc.centerpos[a] for a in "xyz"])
        return np.asarray(xyz) * scale - center

    def read_swc_nodes(self, swc_file: Path, r_scale: float) -> None:
        """keep the SWC nodes, to report the nodes under the mouse; r_scale: the radius scale used"""
        data = np.loadtxt(swc_file, comments="#", ndmin=2)
        xyz = self.swc_to_rendered(data[:, 2:5])
        # NEURON keeps 3D points in single precision
        tolerance = 1e-3 + 1e-6 * float(np.abs(xyz).max(initial=0.0))
        self.swc_nodes = {"data": data, "tree": cKDTree(xyz), "tolerance": tolerance, "r_scale": r_scale}

    def check_topology(
        self, filename: Path, node_report: Union["topology_check.TopologyReport", None], notes: list
    ) -> None:
        """
        Confirm that the loaded morphology is one connected tree (see
        topology_check): the SWC nodes, if read from an SWC file, and the NEURON
        sections. Problems are reported, but the cell is still drawn so that
        they can be found.
        """
        # HocReader checks (and prints) the sections; combine with the node check
        sections = self.hoc.topology_report
        report = topology_check.TopologyReport(source=filename.name, summary=dict(sections.summary))
        report.errors = list(sections.errors)
        report.warnings = self.conversion_notes + list(sections.warnings)
        report.markers = list(sections.markers)
        if node_report is not None:
            report.errors = node_report.errors + report.errors
            report.warnings = node_report.warnings + report.warnings
            report.summary = {"nodes": node_report.summary["nodes"], **report.summary}
            # node-level positions are in the file's coordinates
            for marker in node_report.markers:
                marker.end = self.swc_to_rendered(marker.end)
                marker.toward = self.swc_to_rendered(marker.toward)
            report.markers = node_report.markers + report.markers
            report.fragments = [self.swc_to_rendered(f) for f in node_report.fragments]
        self.topology_report = report
        if not report.ok:
            notes.append(f"topology: {len(report.errors):d} error(s), see Report")
            # once per file and problem, not on every re-render
            warned = (str(filename), tuple(report.errors))
            if warned != self.topology_warned:
                self.topology_warned = warned
                QtWidgets.QMessageBox.warning(self, "Topology errors", str(report))
        elif len(report.warnings) > 0:
            notes.append(f"topology OK, {len(report.warnings):d} warning(s)")
        else:
            notes.append("topology OK")

    def show_topology_report(self) -> None:
        """sources and every transform applied, measurements, and the topology check"""
        if len(self.transform_lines) == 0 and self.topology_report is None:
            self.statusBar().showMessage("Nothing rendered yet")
            return
        text = "SOURCES AND TRANSFORMS\n" + "\n".join(self.transform_lines)
        text += "\n\nMEASUREMENTS\n" + "\n".join(self.measurement_lines())
        if self.topology_report is not None:
            text += "\n\nTOPOLOGY\n" + str(self.topology_report)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Report")
        dialog.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        view = QtWidgets.QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(view)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def ask_radius_scale(self, xyz_scale: list) -> Union[float, None]:
        """
        X, Y, Z are scaled but the radius scale was left at 1: ask, rather than
        assume, whether the radii are in the same units. None if cancelled.
        """
        # mean = float(np.mean(xyz_scale))
        # Claude fixed 2026-11-18: syGlass radii are in X/Y pixels (GC102 SWC vs its mesh: node depth in the
        # mesh ~0.012 x file radius, the X/Y pixel size); the mean of X/Y/Z made processes ~2.5x too thick
        xy = float(np.mean(xyz_scale[:2]))
        xy_text = "the X/Y pixel size" if xyz_scale[0] == xyz_scale[1] else "the mean of X and Y"
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Radius scale")
        box.setText(
            f"X, Y and Z are scaled by {xyz_scale[0]:g}, {xyz_scale[1]:g}, {xyz_scale[2]:g}, "
            "but Scale radius is 1.\n\nAre the radii in the file in the same (pixel) units as the coordinates?"
            "\n\nThe answer is written into Geometry > Scale radius."
        )
        # same = box.addButton(f"Yes: scale radii by {mean:.4g} (mean of X/Y/Z)", QtWidgets.QMessageBox.ButtonRole.YesRole)
        same = box.addButton(f"Yes: scale radii by {xy:.4g} ({xy_text:s})", QtWidgets.QMessageBox.ButtonRole.YesRole)
        as_is = box.addButton("No: radii are already in um (scale 1)", QtWidgets.QMessageBox.ButtonRole.NoRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is same:
            # return mean
            return xy
        if box.clickedButton() is as_is:
            return 1.0
        return None

    # ------------------------------------------------------------------
    # measurements, info panel, show/hide
    # ------------------------------------------------------------------

    def measure(self, mesh: Union[tuple, None]) -> None:
        """membrane area of the reconstruction, surface area of the mesh, and where each is"""
        m = {}
        if self.hoc is not None:
            groups = self.hoc.retrieve_section_group()
            secs = list(neuron.h.allsec())
            m["area"] = sum(seg.area() for sec in secs for seg in sec)
            m["soma_area"] = sum(
                seg.area() for sec in secs if "soma" in str(groups.get(sec.name(), "")).lower() for seg in sec
            )
            m["length"] = sum(sec.L for sec in secs)
            pos = self.hoc.get_geometry()[0]["pos"]
            m["morph_box"] = (pos.min(axis=0), pos.max(axis=0))
        if mesh is not None:
            vertices, faces = mesh
            m["mesh_area"], m["mesh_open"], m["mesh_nonmanifold"] = self.mesh_stats(vertices, faces)
            if m["mesh_open"] == 0:  # the volume means something only for a closed surface
                m["mesh_volume"] = mesh_tools.enclosed_volume(vertices, faces)
            smoothing = self.current_smoothing(self.mesh_cache[0])
            if smoothing is not None:
                m["mesh_smoothing"] = smoothing
            m["mesh_box"] = (vertices.min(axis=0), vertices.max(axis=0))
        if "morph_box" in m and "mesh_box" in m:
            # a pixel-unit morphology with a um mesh (or the reverse) differs by ~10-100x: say so, never rescale
            size_morph = float(np.linalg.norm(m["morph_box"][1] - m["morph_box"][0]))
            size_mesh = float(np.linalg.norm(m["mesh_box"][1] - m["mesh_box"][0]))
            ratio = size_morph / size_mesh if size_mesh > 0 else np.inf
            if ratio > size_mismatch or ratio < 1.0 / size_mismatch:
                m["size_warning"] = (
                    f"SIZE MISMATCH: the reconstruction is {size_morph:.4g} units across, the mesh {size_mesh:.4g} "
                    f"({max(ratio, 1.0 / ratio):.0f}x {'larger' if ratio > 1 else 'smaller'}): "
                    "are both in um? Check Geometry > Scale X/Y/Z and Scale radius"
                )
        self.measurements = m

    def mesh_stats(self, vertices: np.ndarray, faces: np.ndarray) -> tuple:
        """surface area, and the numbers of open (boundary) and non-manifold edges, cached per file (and smoothing)"""
        # key = None if self.mesh_cache is None else self.mesh_cache[:2]
        # Claude fixed 2026-11-18: a smoothed mesh has its own area
        smoothing = None if self.mesh_cache is None else self.current_smoothing(self.mesh_cache[0])
        key = None
        if self.mesh_cache is not None:
            key = (*self.mesh_cache[:2], *((smoothing["method"], smoothing["iterations"]) if smoothing else ("none", 0)))
        # if self.mesh_stats_cache is not None and self.mesh_stats_cache[:2] == key:
        #     return self.mesh_stats_cache[2:]
        if key is not None and self.mesh_stats_cache is not None and self.mesh_stats_cache[0] == key:
            return self.mesh_stats_cache[1]
        # v = vertices.astype(np.float64)
        # a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
        # area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
        # edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1).astype(np.int64)
        # _, counts = np.unique(edges[:, 0] * (len(vertices) + 1) + edges[:, 1], return_counts=True)
        # stats = (area, int(np.sum(counts == 1)), int(np.sum(counts > 2)))
        # self.mesh_stats_cache = (*key, *stats) if key is not None else None
        stats = (mesh_tools.surface_area(vertices, faces), *mesh_tools.edge_counts(faces, len(vertices)))
        self.mesh_stats_cache = (key, stats) if key is not None else None
        return stats

    def measurement_lines(self) -> list:
        m = self.measurements
        lines = []
        if "size_warning" in m:
            lines.append(m["size_warning"])
        if "area" in m:
            lines.append(
                f"Reconstruction (NEURON sections): membrane area {m['area']:.1f} um2 "
                f"(soma {m['soma_area']:.1f} um2), total length {m['length']:.1f} um"
            )
            lines.append("  membrane area is the side surface of the sections (no end caps), as NEURON computes it")
        if "mesh_area" in m:
            closed = "closed surface" if m["mesh_open"] == 0 else f"open surface: {m['mesh_open']:d} boundary edges"
            if m["mesh_nonmanifold"] > 0:
                closed += f", {m['mesh_nonmanifold']:d} non-manifold edges"
            # lines.append(f"Mesh: surface area {m['mesh_area']:.1f} um2 ({closed:s})")
            volume = f", volume {m['mesh_volume']:.1f} um3" if "mesh_volume" in m else ""
            lines.append(f"Mesh: surface area {m['mesh_area']:.1f} um2{volume:s} ({closed:s})")
            if "mesh_smoothing" in m:
                s = m["mesh_smoothing"]
                lines.append(
                    f"  smoothed ({s['method']:s} x{s['iterations']:d}); before: area {s['area_before']:.1f} um2, "
                    f"volume {s['volume_before']:.1f} um3; area changed {100 * (m['mesh_area'] / s['area_before'] - 1):+.1f}%"
                )
        if "area" in m and "mesh_area" in m and m["area"] > 0:
            lines.append(f"Mesh / reconstruction area: {m['mesh_area'] / m['area']:.3f}")
            if "mesh_smoothing" in m:
                lines.append(f"  before smoothing: {m['mesh_smoothing']['area_before'] / m['area']:.3f}")
        if "morph_box" in m and "mesh_box" in m:
            (lo1, hi1), (lo2, hi2) = m["morph_box"], m["mesh_box"]
            d = (lo2 + hi2) / 2 - (lo1 + hi1) / 2
            inter = np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None).prod()
            smaller = min(np.prod(hi1 - lo1), np.prod(hi2 - lo2))
            overlap = inter / smaller if smaller > 0 else 0.0
            lines.append(
                f"Bounding-box centres differ by {np.linalg.norm(d):.1f} um "
                f"(mesh - reconstruction: {d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}); boxes overlap {100 * overlap:.0f}% "
                "of the smaller one"
            )
        return lines

    def show_info(self) -> None:
        """transforms and measurements over the view, so that they are always in sight"""
        if self.viewer is None:
            return
        lines = []
        if self.hoc is not None:
            lines.append(f"Morphology: {Path(self.value('Input', 'Morphology file')).name:s}")
            lines += [f"  {line:s}" for line in self.short["morphology"]]
        if len(self.short["mesh"]) > 0:
            lines.append(f"Mesh: {Path(self.value('Input', 'Mesh file')).name:s}")
            lines += [f"  {line:s}" for line in self.short["mesh"]]
        lines += self.measurement_lines()
        lines.append("All details: Actions > Report...")
        # self.info_label = QtWidgets.QLabel("\n".join(lines), self.viewer)
        # rich text, so that a size mismatch can be shown in red
        warning = self.measurements.get("size_warning")
        text = "<br>".join(
            f'<span style="color: {warning_color:s};">{html.escape(line):s}</span>'
            if line == warning
            else html.escape(line).replace("  ", "&nbsp;&nbsp;")
            for line in lines
        )
        self.info_label = QtWidgets.QLabel(text, self.viewer)
        self.info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.info_label.setStyleSheet(
            "QLabel { background-color: rgba(20, 20, 20, 190); color: #e0e0e0; padding: 4px; "
            "font-family: Menlo, Consolas, monospace; font-size: 10px; }"
        )
        self.info_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.info_label.adjustSize()
        self.info_label.move(6, 6)
        self.info_label.show()
        self.info_label.raise_()
        CP("c", "\n".join(self.transform_lines + self.measurement_lines()))

    def apply_visibility(self) -> None:
        """show or hide morphology, mesh, points and grid without re-rendering"""
        names = {"morphology": "Morphology", "mesh": "Mesh", "points": "Points", "grid": "Grid and axes"}
        for kind, items in self.layers.items():
            visible = self.value("Show", names[kind])
            for item in items:
                if self.viewer_kind == "vispy":
                    item.visible = visible
                else:
                    item.setVisible(visible)
        if self.viewer_kind == "vispy" and self.vispy_canvas is not None:
            self.vispy_canvas.update()
        if self.info_label is not None:
            self.info_label.setVisible(self.value("Show", "Info panel"))
        if not self.value("Show", "Morphology"):
            self.hide_hover()

    def convert_imaris(self, filename: Path, notes: list) -> Path:
        """
        Convert the selected filament objects of an .ims file to an SWC file in
        the temporary directory and return its path. Pieces not connected to the
        soma are kept as separate trees: swc_to_hoc builds only the soma's tree,
        and the topology check reports the others so they can be marked.
        """
        names = self.value("Input", "Imaris filaments")
        if len(self.params.child("Input", "Imaris filaments").opts["limits"]) == 0:
            # re-read so that the cause (e.g. h5py not installed) is reported
            if len(imaris_reader.list_filament_objects(filename)) == 0:
                raise ValueError(f"{filename.name:s} has no filament tracings (image data only?)")
            self.update_filament_choices()
            names = self.value("Input", "Imaris filaments")
        if len(names) == 0:
            raise ValueError(f"No filament objects selected in {filename.name:s}")
        swc_dir = Path(tempfile.gettempdir(), "neuronvis_ims")
        swc_dir.mkdir(exist_ok=True)
        swc_file, info, _ = imaris_reader.ims_to_swc(
            filename, Path(swc_dir, filename.stem + ".swc"), names, connected_only=False
        )
        self.imaris_provenance = info["provenance"]
        self.conversion_notes = [f"repaired: {j:s}" for j in info["joins"]]
        if len(info["joins"]) > 0:
            notes.append(f"{len(info['joins']):d} piece(s) joined, see Report")
        return swc_file

    def read_points(self, offset: np.ndarray) -> Union[pd.DataFrame, None]:
        points_file = self.value("Input", "Points file")
        if not points_file:
            return None
        points = pd.read_csv(points_file)
        points["x"] -= offset[0]
        points["y"] -= offset[1]
        points["z"] -= offset[2]
        return points

    def read_mesh(self, filename: Path) -> tuple:
        """
        read an .obj mesh, reusing the last one read if the file has not changed;
        vertices duplicated at block seams are merged (the file is not changed)
        """
        mtime = filename.stat().st_mtime
        if self.mesh_cache is None or self.mesh_cache[:2] != (filename, mtime):
            vertices, faces = obj_reader.read_obj(filename)
            vertices, faces, info, lines = mesh_tools.merge_seams(vertices, faces)
            self.mesh_cache = (filename, mtime, vertices, faces)
            self.mesh_merge = (filename, mtime, info, lines)
        return self.mesh_cache[2], self.mesh_cache[3]

    def current_smoothing(self, filename: Path) -> Union[dict, None]:
        """the last smoothing, if it was of this file and with the Mesh settings now shown"""
        s = self.smoothed
        if s is None or s["source"] != (filename, filename.stat().st_mtime):
            return None
        if (s["method"], s["iterations"]) != (self.value("Mesh", "Smoothing"), self.value("Mesh", "Iterations")):
            return None
        return s

    def smoothing_changed(self, param: object, method: str) -> None:
        """each method's usual number of iterations, shown in the field (change it there)"""
        if method in smoothing_iterations:
            self.params.child("Mesh", "Iterations").setValue(smoothing_iterations[method])

    def smooth_mesh(self) -> None:
        """smooth the mesh with the method in Mesh > Smoothing; the result is drawn and measured"""
        mesh_file = self.value("Input", "Mesh file")
        method = self.value("Mesh", "Smoothing")
        iterations = self.value("Mesh", "Iterations")
        if not mesh_file:
            self.statusBar().showMessage("Choose a mesh file first (Input > Mesh file)")
            return
        if method == "none":
            self.statusBar().showMessage("Choose a smoothing method first (Mesh > Smoothing)")
            return
        if self.smooth_process is not None:
            self.statusBar().showMessage("GAMer is still running")
            return
        filename = Path(mesh_file)
        vertices, faces = self.read_mesh(filename)
        source = (filename, filename.stat().st_mtime)
        if method == "Taubin":
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                smoothed = mesh_tools.taubin_smooth(vertices, faces, iterations)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            description = (
                f"Taubin lambda {mesh_tools.taubin_lambda:g}, mu {mesh_tools.taubin_mu:g}, {iterations:d} iterations "
                "(umbrella operator; keeps the volume; triangles unchanged)"
            )
            self.finish_smoothing(source, method, iterations, smoothed, faces, description)
            return
        ok, message = mesh_tools.gamer_available()
        if getattr(sys, "frozen", False):
            ok, message = False, "GAMer is not included in the packaged app: run hocRenderGUI from source with PyGAMer."
        if not ok:
            QtWidgets.QMessageBox.information(self, "GAMer smoothing", message)
            return
        # GAMer takes minutes on a large mesh and holds the interpreter while it runs: use a separate process
        work = Path(tempfile.gettempdir(), "neuronvis_gamer")
        work.mkdir(exist_ok=True)
        mesh_tools.write_obj(Path(work, "input.obj"), vertices, faces, [f"seams merged from {filename.resolve()!s}"])
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        log = []
        partial = [""]  # output can arrive in pieces that split a line
        started = time.time()
        # steps: starting the process and reading the mesh, GAMer's setup, the iterations, drawing the result
        steps = iterations + 3
        self.show_progress(0, steps, "GAMer: starting")

        def output() -> None:
            text = partial[0] + bytes(process.readAllStandardOutput()).decode(errors="replace")
            lines = text.split("\n")
            partial[0] = lines.pop()  # incomplete last line, kept for the next call
            for line in (line.strip() for line in lines):
                if not line:
                    continue
                if line.startswith("progress "):  # "progress done total text" from mesh_tools.main
                    _, done, total, *words = line.split(" ")
                    self.show_progress(1 + int(done), steps, f"GAMer: {' '.join(words):s}")
                else:
                    log.append(line)
                    self.statusBar().showMessage(f"GAMer: {line:s}")

        def finished(code: int, status: object) -> None:
            self.smooth_process = None
            self.params.child("Mesh", "Smooth mesh").setOpts(enabled=True)
            CP("c", "\n".join(log))
            result = Path(work, "output.obj")
            if code != 0 or not result.is_file():
                self.show_progress(steps, steps, f"GAMer failed after {time.time() - started:.0f} s", failed=True)
                QtWidgets.QMessageBox.critical(self, "GAMer smoothing failed", "\n".join(log[-20:]))
                return
            self.show_progress(steps - 1, steps, "GAMer: drawing the smoothed mesh")
            v, f = obj_reader.read_obj(result)
            description = f"GAMer2 smooth(max_iter={iterations:d}, preserve_ridges=False), all vertices selected"
            self.finish_smoothing(source, method, iterations, v, f, description)
            self.show_progress(steps, steps, f"GAMer done: {iterations:d} iterations in {time.time() - started:.0f} s")
            self.statusBar().showMessage(
                f"GAMer smoothing done ({time.time() - started:.0f} s): the mesh shown and measured is the smoothed one"
            )

        process.readyReadStandardOutput.connect(output)
        process.finished.connect(finished)
        self.smooth_process = process
        self.params.child("Mesh", "Smooth mesh").setOpts(enabled=False)
        self.statusBar().showMessage(f"GAMer: smoothing ({iterations:d} iterations; minutes for a large mesh) ...")
        process.start(
            sys.executable,
            ["-m", "neuronvis.mesh_tools", "gamer", str(Path(work, "input.obj")), str(Path(work, "output.obj")),
             "--iterations", str(iterations)],
        )

    def show_progress(self, done: int, total: int, text: str, failed: bool = False) -> None:
        """
        the status-bar progress bar: filling while GAMer runs; full and green
        ("done") or red ("failed") when it ends, until the next smoothing
        """
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"{text:s}  %p%" if done < total else text)
        if done < total:
            color = ""
        else:
            color = "QProgressBar::chunk { background-color: %s; }" % ("#c0392b" if failed else "#2e8b57")
        self.progress_bar.setStyleSheet(color)
        self.progress_bar.show()

    def finish_smoothing(
        self, source: tuple, method: str, iterations: int, vertices: np.ndarray, faces: np.ndarray, description: str
    ) -> None:
        before_v, before_f = self.mesh_cache[2], self.mesh_cache[3]
        self.smoothed = {
            "source": source,
            "method": method,
            "iterations": iterations,
            "vertices": np.asarray(vertices, dtype=np.float32),
            "faces": np.asarray(faces, dtype=np.int32),
            "description": description,
            "area_before": mesh_tools.surface_area(before_v, before_f),
            "volume_before": mesh_tools.enclosed_volume(before_v, before_f),
            "merge_lines": list(self.mesh_merge[3]),
        }
        self.render()

    def save_smoothed_mesh(self) -> None:
        """write the smoothed mesh (in the source file's coordinates) with a header saying how it was made"""
        mesh_file = self.value("Input", "Mesh file")
        s = self.current_smoothing(Path(mesh_file)) if mesh_file else None
        if s is None:
            QtWidgets.QMessageBox.information(
                self, "Save smoothed mesh", "No smoothed mesh for this file and these Mesh settings: press Smooth mesh first."
            )
            return
        source = Path(mesh_file)
        default = source.with_name(f"{source.stem:s}_{s['method'].lower():s}{s['iterations']:d}.obj")
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save smoothed mesh", str(default), "OBJ meshes (*.obj)")
        if not name:
            return
        area = mesh_tools.surface_area(s["vertices"], s["faces"])
        volume = mesh_tools.enclosed_volume(s["vertices"], s["faces"])
        header = [
            "neuronvis smoothed mesh",
            f"written: {datetime.datetime.now().isoformat(timespec='seconds'):s}",
            f"source: {source.resolve()!s}",
            *[f"source header: {line:s}" for line in mesh_tools.obj_header(source)],
            "coordinates: as in the source file (not scaled, not centred)",
            *s["merge_lines"],
            f"smoothing: {s['description']:s}",
            f"area: {s['area_before']:.2f} -> {area:.2f} um2; volume {s['volume_before']:.2f} -> {volume:.2f} um3",
        ]
        mesh_tools.write_obj(name, s["vertices"], s["faces"], header)
        self.statusBar().showMessage(f"Saved {name:s}")

    def mesh_rgba(self) -> tuple:
        color = self.params.child("Mesh", "Mesh color").value()
        return (color.redF(), color.greenF(), color.blueF(), self.value("Mesh", "Mesh alpha"))

    def set_viewer(self, widget: QtWidgets.QWidget, kind: str) -> None:
        self.placeholder.hide()
        self.viewer_layout.addWidget(widget)
        widget.show()
        self.viewer = widget
        self.viewer_kind = kind

    def clear_viewer(self) -> None:
        self.bounds = None
        if self.vispy_graphic is not None:
            self.vispy_graphic.timer.stop()
            self.vispy_graphic.close()  # the unused vispy.app.Canvas it also inherits from
            self.vispy_graphic = None
        if self.vispy_canvas is not None:
            self.vispy_canvas.close()
            self.vispy_canvas = None
            self.vispy_view = None
        if self.mpl_figure is not None:
            self.mpl_figure.clear()
            self.mpl_figure = None
        self.picker = None
        self.pick_sections = None
        self.hover_items = None
        self.hover_label = None
        self.info_label = None
        self.layers = {}
        self.extents = {}
        self.hover_timer.stop()
        if self.viewer is not None:
            self.viewer_layout.removeWidget(self.viewer)
            self.viewer.setParent(None)
            self.viewer.deleteLater()
            self.viewer = None
            self.viewer_kind = None
        self.placeholder.show()

    def render_pyqtgraph(
        self,
        style: str,
        mode: str,
        mechanism: Union[str, None],
        alpha: float,
        points: Union[pd.DataFrame, None],
        mesh: Union[tuple, None],
        notes: list,
    ) -> None:
        extents = []  # point clouds whose bounds the camera is fitted to
        if self.hoc is None:  # mesh only: HocViewer needs a HocReader
            view = opengl.GLViewWidget()
            axis = opengl.GLAxisItem()
            axis.setSize(10, 10, 10)
            view.addItem(axis)
        else:
            view = make_embedded_viewer(self.hoc, camerapos=[200.0, 0.0, 0.0], renderer="pyqtgraph")
        self.set_viewer(view, "pyqtgraph")
        self.layers = {"grid": list(view.items), "morphology": [], "mesh": [], "points": []}
        self.extents = {}

        if self.hoc is not None:
            match style:
                case "cylinders":
                    g = view.draw_cylinders()
                    g.setShader("balloon")
                case "graph":
                    g = view.draw_graph()
                case "surface":
                    g = view.draw_surface()
            if style == "graph":  # HocGraph's data is held by its child line items
                extents.append(np.concatenate([line.pos for line in g.lines], axis=0))
            else:
                extents.append(g.opts["meshdata"].vertexes().reshape(-1, 3))
            self.layers["morphology"].append(g)
            self.extents["morphology"] = extents[-1]
            # HocGraph/HocSurface.set_section_colors expect an array, not the group
            # dict; hocRender does not color them either.
            if style == "cylinders":
                if mode == "sec-type":
                    g.set_group_colors(rc.section_colors, alpha=alpha)
                elif mode == "mechanism" and mechanism is not None:
                    g.set_group_colors(rc.section_colors, mechanism=mechanism)
            else:
                notes.append(f"{style:s} is drawn uncolored")
            self.record_shifts(notes)
            self.setup_hover(view)

        if points is not None:
            colors = np.array([pg.mkColor(c).getRgbF() for c in points["color"].values])
            scatter = opengl.GLScatterPlotItem(
                pos=points[["x", "y", "z"]].to_numpy(),
                size=1,
                color=colors,
                pxMode=False,
            )
            view.addItem(scatter)
            self.layers["points"].append(scatter)
            self.extents["points"] = points[["x", "y", "z"]].to_numpy()

        if mesh is not None:
            vertices, faces = mesh
            rgba = self.mesh_rgba()
            # flat shading: pyqtgraph's per-vertex normals are slow for ~1e6 vertices
            mesh_item = opengl.GLMeshItem(
                meshdata=opengl.MeshData(vertexes=vertices, faces=faces),
                smooth=False,
                shader="shaded",
                color=rgba,
                glOptions="translucent" if rgba[3] < 1.0 else "opaque",
            )
            view.addItem(mesh_item)  # added last so the morphology shows through it
            extents.append(vertices)
            self.layers["mesh"].append(mesh_item)
            self.extents["mesh"] = vertices

        if self.topology_report is not None and len(self.topology_report.fragments) > 0:
            extents.append(np.concatenate(self.topology_report.fragments).reshape(-1, 3))
        pos = np.concatenate(extents, axis=0)
        self.bounds = (pos.min(axis=0), pos.max(axis=0))
        if self.hoc is not None:
            self.draw_markers(view)
        self.reset_view()

    # ------------------------------------------------------------------
    # disconnection markers and hover information (pyqtgraph view)
    # ------------------------------------------------------------------

    def draw_markers(self, view: object) -> None:
        """
        Dark red: pieces not connected to the main tree; red arrows point at
        their ends nearest it; orange arrows: gaps between a child and the point
        where it attaches. Arrows are drawn on top of everything.
        """
        report = self.topology_report
        if report is None:
            return
        show = {
            "disconnected": self.value("Inspect", "Mark disconnections"),
            "gap": self.value("Inspect", "Mark gaps"),
        }
        if show["disconnected"] and len(report.fragments) > 0:
            pieces = opengl.GLLinePlotItem(
                pos=np.concatenate(report.fragments).reshape(-1, 3),
                mode="lines",
                color=fragment_color,
                width=2,
                antialias=True,
            )
            view.addItem(pieces)
            self.layers["morphology"].append(pieces)  # Claude fixed 2026-11-18: hide with the morphology
        lo, hi = self.bounds
        center = (lo + hi) / 2.0
        length = max(0.12 * float(np.max(hi - lo)), 8.0)
        for kind, color in [("disconnected", disconnected_color), ("gap", gap_color)]:
            markers = [m for m in report.markers if m.kind == kind]
            if not show[kind] or len(markers) == 0:
                continue
            lines = []
            for m in markers:
                lines += arrow_lines(np.asarray(m.end, float), np.asarray(m.end, float) - center, length)
                lines += [m.end, m.toward]  # the gap itself
            arrows = opengl.GLLinePlotItem(
                pos=np.array(lines, dtype=float), mode="lines", color=color, width=4, antialias=True
            )
            arrows.setGLOptions(on_top)
            view.addItem(arrows)
            ends = opengl.GLScatterPlotItem(
                pos=np.array([m.end for m in markers], dtype=float), color=color, size=14, pxMode=True
            )
            ends.setGLOptions(on_top)
            view.addItem(ends)
            self.layers["morphology"] += [arrows, ends]  # Claude fixed 2026-11-18: hide with the morphology

    def record_shifts(self, notes: list) -> None:
        """
        get_geometry() ran define_shape(), which moves child sections to close
        gaps at connections: note how far each moved, and put the gap markers
        where the sections are drawn
        """
        self.section_shift = {}
        for name, before in self.file_first_points.items():
            sec = self.hoc.sections.get(name)
            if sec is None or sec.n3d() == 0:
                continue
            shift = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)]) - before
            if np.linalg.norm(shift) > 1e-3:
                self.section_shift[name] = shift
        if len(self.section_shift) > 0:
            largest = max(np.linalg.norm(v) for v in self.section_shift.values())
            notes.append(f"NEURON moved {len(self.section_shift):d} section(s), up to {largest:.2f} um, to close gaps")
            self.transform_lines.append(
                f"  drawn: NEURON's define_shape() moved {len(self.section_shift):d} section(s), up to {largest:.2f} um, "
                "to close gaps at connections"
            )
            self.short["morphology"].append(f"drawn: NEURON moved {len(self.section_shift):d} section(s), up to {largest:.2f} um")
        if self.topology_report is not None:
            for m in self.topology_report.markers:
                if m.kind == "gap" and m.section in self.hoc.sections:
                    sec = self.hoc.sections[m.section]
                    m.end = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
                    m.toward = m.end

    def setup_hover(self, view: object) -> None:
        """pick the segment under the mouse from the edges HocReader drew"""
        vertexes, edges = self.hoc.get_geometry()
        if len(edges) == 0:
            return
        names = np.array(list(self.hoc.sections.keys()), dtype=object)
        self.picker = section_picker.SegmentPicker(vertexes["pos"], vertexes["dia"], edges)
        # parent-child bridging edges run parent -> child; both kinds end in their own section
        self.pick_sections = names[vertexes["sec_index"][edges[:, 1]]]
        view.setMouseTracking(True)
        view.installEventFilter(self)
        self.hover_label = QtWidgets.QLabel(view)
        self.hover_label.setStyleSheet(
            "QLabel { background-color: rgba(20, 20, 20, 210); color: white; "
            "border: 1px solid #888; padding: 4px; font-family: Menlo, Consolas, monospace; font-size: 11px; }"
        )
        self.hover_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
        self.hover_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hover_label.hide()
        line = opengl.GLLinePlotItem(pos=np.zeros((2, 3)), color=highlight_color, width=4, antialias=True)
        point = opengl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=highlight_color, size=9, pxMode=True)
        for item in (line, point):
            item.setGLOptions(on_top)
            item.setVisible(False)
            view.addItem(item)
        self.hover_items = (line, point)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if self.picker is not None and obj is self.viewer:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                if (
                    event.buttons() == QtCore.Qt.MouseButton.NoButton
                    and self.value("Inspect", "Hover info")
                    and self.value("Show", "Morphology")
                ):
                    self.hover_pos = event.position()
                    self.hover_timer.start(30)  # pick once the mouse pauses
                else:
                    self.hide_hover()
            elif event.type() in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.Wheel):
                self.hide_hover()
        return super().eventFilter(obj, event)

    def hide_hover(self) -> None:
        self.hover_timer.stop()
        if self.hover_label is not None:
            self.hover_label.hide()
        if self.hover_items is not None:
            for item in self.hover_items:
                item.setVisible(False)

    def mouse_ray(self, x: float, y: float) -> tuple:
        """camera position, ray direction through widget point (x, y), and pixel size at unit distance"""
        view = self.viewer
        w, h = view.width(), view.height()
        half = tan(radians(view.opts["fov"]) / 2.0)  # fov is horizontal
        direction = np.array([half * (2.0 * x / w - 1.0), half * (h / w) * (1.0 - 2.0 * y / h), -1.0])
        to_world = np.array(view.viewMatrix().inverted()[0].copyDataTo()).reshape(4, 4)
        return to_world[:3, 3], to_world[:3, :3] @ direction, 2.0 * half / w

    def hover_pick(self) -> None:
        if self.picker is None or self.hover_pos is None:
            return
        x, y = self.hover_pos.x(), self.hover_pos.y()
        origin, direction, pixel_size = self.mouse_ray(x, y)
        picked = self.picker.pick(origin, direction, pixel_size)
        if picked is None:
            self.hide_hover()
            return
        k, t, point, _ = picked
        sec = self.hoc.sections[self.pick_sections[k]]
        line, marker = self.hover_items
        line.setData(pos=np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())]))
        marker.setData(pos=np.array([point]))
        line.setVisible(True)
        marker.setVisible(True)
        self.hover_label.setText(self.segment_info(k, t, point))
        self.hover_label.adjustSize()
        # beside the mouse, kept inside the view
        lx = x + 18 if x + 18 + self.hover_label.width() < self.viewer.width() else x - 18 - self.hover_label.width()
        ly = min(max(y + 18, 0), max(self.viewer.height() - self.hover_label.height(), 0))
        self.hover_label.move(int(max(lx, 0)), int(ly))
        self.hover_label.show()
        self.hover_label.raise_()

    def segment_info(self, k: int, t: float, point: np.ndarray) -> str:
        """text for picker edge *k*: its section, the segment at the mouse, and the SWC nodes"""
        vertexes, edges = self.hoc.get_geometry()
        a, b = edges[k]
        name = self.pick_sections[k]
        sec = self.hoc.sections[name]
        n3d = sec.n3d()
        pts = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n3d)])
        diams = np.array([sec.diam3d(i) for i in range(n3d)])
        i0 = int(np.argmin(np.linalg.norm(pts - vertexes["pos"][a], axis=1)))
        i1 = int(np.argmin(np.linalg.norm(pts - vertexes["pos"][b], axis=1)))
        arc0, arc1 = sec.arc3d(i0), sec.arc3d(i1)
        x = min(max((arc0 + t * (arc1 - arc0)) / sec.L, 0.0), 1.0) if sec.L > 0 else 0.0
        seg = sec(x)
        parent = sec.parentseg()
        n_children = len(sec.children())
        n_connections = n_children + (0 if parent is None else 1)
        role = {0: "not connected", 1: "terminal", 2: "continuing"}.get(n_connections, "branch point")
        if parent is None:
            role = "root" if n_connections > 0 else "not connected"
        mechs = sorted(sec.psection()["density_mechs"].keys())
        piece = float(np.linalg.norm(vertexes["pos"][b] - vertexes["pos"][a]))
        # t runs from 3D point i0 (edge vertex a) to i1 (b)
        here = float(diams[i0] + t * (diams[i1] - diams[i0]))
        differs = abs(seg.diam - here) > 0.25 * max(here, 1e-6)
        lines = [
            f"{name:s}   type: {vertexes['sec_type'][b]!s}",
            f"mechanisms: {', '.join(mechs) if mechs else 'none'}",
            f"L {sec.L:.2f} um, nseg {sec.nseg:d}, {n3d:d} 3D points, diam {diams.min():.2f}-{diams.max():.2f} um",
            f"parent: {'none' if parent is None else f'{parent.sec.name()}({parent.x:.3g})'}, "
            f"children: {n_children:d} ({role:s})",
            "",
            f"at the mouse: x = {x:.3f}, diam {here:.2f} um (between 3D points {i0:d} and {i1:d})",
            f"  3D point {i0:d}: ({pts[i0, 0]:.2f}, {pts[i0, 1]:.2f}, {pts[i0, 2]:.2f})  diam {diams[i0]:.2f}",
            f"  3D point {i1:d}: ({pts[i1, 0]:.2f}, {pts[i1, 1]:.2f}, {pts[i1, 2]:.2f})  diam {diams[i1]:.2f}",
            f"  piece length {piece:.2f} um; pointer at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})",
            # seg.diam is NEURON's value for the whole segment (area-weighted over L/nseg),
            # not the diameter at the mouse
            f"NEURON segment {name:s}({seg.x:.3g}): diam {seg.diam:.2f} um, the average over "
            f"{sec.L / sec.nseg:.2f} um (nseg {sec.nseg:d})"
            + ("; differs from the local diameter, more segments (nseg) would follow it" if differs else ""),
        ]
        shift = self.section_shift.get(name)
        if shift is not None:
            lines.append(f"  drawn {np.linalg.norm(shift):.2f} um from its file position (moved by NEURON's define_shape)")
        if self.swc_nodes is not None:
            for end in (a, b):
                end_section = self.pick_sections[k] if end == b else names_of(vertexes, end, self.hoc)
                file_pos = vertexes["pos"][end] - self.section_shift.get(end_section, 0.0)
                distance, row = self.swc_nodes["tree"].query(file_pos)
                if distance <= self.swc_nodes["tolerance"]:
                    node = self.swc_nodes["data"][row]
                    # lines.append(
                    #     f"  SWC node {int(node[0]):d}: type {int(node[1]):d}, radius {node[5]:.3f}, "
                    #     f"parent {int(node[6]):d}  (as in the file)"
                    # )
                    # Claude fixed 2026-11-18: the file radius is in the file's units (pixels for this cell); show it scaled to um too
                    r_scale = self.swc_nodes["r_scale"]
                    lines.append(
                        f"  SWC node {int(node[0]):d}: type {int(node[1]):d}, parent {int(node[6]):d}, "
                        f"in the file at ({node[2]:.3f}, {node[3]:.3f}, {node[4]:.3f})"
                    )
                    lines.append(
                        f"    radius {node[5]:.4g} (file units) x {r_scale:.4g} = {node[5] * r_scale:.3f} um "
                        f"(diam {2.0 * node[5] * r_scale:.3f} um)"
                    )
        return "\n".join(lines)

    def render_mpl(self, style: str, points: Union[pd.DataFrame, None]) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        from neuronvis import hoc_graphics as HG

        fig = Figure()
        ax = fig.add_subplot(projection="3d")
        canvas = FigureCanvasQTAgg(fig)
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(canvas, holder))
        layout.addWidget(canvas)
        if style == "graph":
            HG.mpl_Graph(self.hoc, fax=[fig, ax])
        else:
            HG.mpl_Cylinders(self.hoc, fax=[fig, ax], colors=rc.section_colors)
        if points is not None:
            ax.scatter(points["x"], points["y"], points["z"], color=points["color"], s=10)
        self.mpl_figure = fig
        self.set_viewer(holder, "mpl")
        canvas.draw_idle()

    def render_vispy(self, mechanism: Union[str, None], title: str, mesh: Union[tuple, None]) -> None:
        import vispy

        try:
            vispy.use(app="pyqt6")
        except RuntimeError:
            pass  # backend already selected
        import vispy.scene

        from neuronvis import hoc_graphics as HG

        # Moving a vispy canvas into another window after it has been shown
        # recreates its GL context and vispy then fails to draw (GL_INVALID_OPERATION).
        # vispy_Cylinders does not take a parent, so create its SceneCanvas
        # directly inside a host widget that is already in the main window.
        host = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        self.set_viewer(host, "vispy")
        scene_canvas = vispy.scene.SceneCanvas

        def embedded_scene_canvas(*args, **kwds):
            kwds.setdefault("parent", host)
            canvas = scene_canvas(*args, **kwds)
            layout.addWidget(canvas.native)
            return canvas

        if self.hoc is not None:
            vispy.scene.SceneCanvas = embedded_scene_canvas
            try:
                # vispy_Cylinders shows its canvas and calls vispy.app.run(); with the
                # Qt event loop already running, run() returns immediately.
                self.vispy_graphic = HG.vispy_Cylinders(
                    self.hoc, mechanism=mechanism, color=rc.section_colors, title=title, headlight=True
                )
            finally:
                vispy.scene.SceneCanvas = scene_canvas
            self.vispy_canvas = self.vispy_graphic.canvas
            self.vispy_view = self.vispy_graphic.view
            extents = [np.concatenate(self.vispy_graphic.tubes["points"], axis=0)]
            self.layers = {
                "morphology": list(self.vispy_graphic.vtubes) + list(self.vispy_graphic.vspheres),
                "grid": [v for v in self.vispy_view.scene.children if type(v).__name__ == "XYZAxis"],
                "mesh": [],
                "points": [],
            }
            self.extents = {"morphology": extents[0]}
        else:  # mesh only
            self.vispy_canvas = embedded_scene_canvas(keys="interactive", bgcolor=[0.75, 0.75, 0.75, 1])
            self.vispy_view = self.vispy_canvas.central_widget.add_view()
            self.vispy_view.camera = "arcball"
            axis = vispy.scene.visuals.XYZAxis(parent=self.vispy_view.scene)
            axis.transform = vispy.visuals.transforms.STTransform(scale=(10, 10, 10))
            self.vispy_canvas.show()
            extents = []
            self.layers = {"morphology": [], "grid": [axis], "mesh": [], "points": []}
            self.extents = {}

        if mesh is not None:
            vertices, faces = mesh
            mesh_visual = vispy.scene.visuals.Mesh(
                vertices=vertices, faces=faces, color=self.mesh_rgba(), shading="smooth"
            )
            # translucent and drawn after the morphology so that it shows through
            mesh_visual.set_gl_state("translucent", depth_test=True, cull_face=False)
            self.vispy_view.add(mesh_visual)
            extents.append(vertices)
            self.layers["mesh"].append(mesh_visual)
            self.extents["mesh"] = vertices
        pos = np.concatenate(extents, axis=0)
        self.bounds = (pos.min(axis=0), pos.max(axis=0))
        self.reset_view()

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        """center the camera on what is shown (morphology, mesh, points) and back off to see all of it"""
        shown = [
            pts
            for kind, pts in self.extents.items()
            if len(pts) > 0 and self.value("Show", {"morphology": "Morphology", "mesh": "Mesh", "points": "Points"}[kind])
        ]
        if len(shown) > 0:
            pos = np.concatenate(shown, axis=0)
            self.bounds = (pos.min(axis=0), pos.max(axis=0))
        if self.viewer_kind == "vispy":
            # fit to the cell and mesh, not the whole scene: the axis marker at
            # the origin can be far from the data
            lo, hi = self.bounds
            self.vispy_view.camera.set_range(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))
            return
        if self.viewer_kind != "pyqtgraph" or self.bounds is None:
            return
        lo, hi = self.bounds
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo))
        self.viewer.setCameraPosition(
            pos=QtGui.QVector3D(*[float(c) for c in center]),
            distance=max(extent * 1.5, 10.0),
            elevation=30.0,
            azimuth=45.0,
        )

    def save_image(self) -> None:
        if self.viewer is None:
            self.statusBar().showMessage("Nothing to save")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", str(Path.cwd()), image_filter
        )
        if not filename:
            return
        match self.viewer_kind:
            case "pyqtgraph":
                self.viewer.grabFramebuffer().save(filename)
            case "mpl":
                self.mpl_figure.savefig(filename, dpi=300)
            case "vispy":
                import vispy.io

                vispy.io.write_png(filename, self.vispy_canvas.render())
        self.statusBar().showMessage(f"Saved image to {filename:s}")

    def export_swc(self) -> None:
        """save the selected filament objects of the current .ims file as SWC"""
        filename = self.value("Input", "Morphology file")
        if not filename or Path(filename).suffix != ".ims":
            self.statusBar().showMessage("Export SWC: choose an Imaris (.ims) morphology file first")
            return
        names = self.value("Input", "Imaris filaments")
        if len(names) == 0:
            self.statusBar().showMessage("Export SWC: no filament objects selected")
            return
        swc_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export SWC", str(Path(filename).with_suffix(".swc")), "SWC (*.swc)"
        )
        if not swc_file:
            return
        # only the tree connected to the soma: an SWC must be one connected tree
        swc_file, info, report = imaris_reader.ims_to_swc(filename, swc_file, names, connected_only=True)
        msg = f"Exported {', '.join(names):s} to {str(swc_file):s}"
        if len(info["joins"]) > 0:
            msg += f"  [{len(info['joins']):d} piece(s) joined]"
        if info["fragments"] > 0:
            msg += (
                f"  [{info['fragments']:d} piece(s), {info['fragment_points']:d} points, "
                "not connected to the soma left out]"
            )
        msg += "  [topology OK]" if report.ok else "  [topology errors, see console]"
        print(report)
        self.statusBar().showMessage(msg)

    def reload_source(self) -> None:
        """
        Reload changed neuronvis modules, following pyqtgraph's approach
        (see pyqtgraph.reload): modules whose .py is newer than their cached
        .pyc are re-imported and existing classes/functions are updated in
        place, so the open window picks up the new code.

        pyqtgraph cannot update some existing objects in place (for example a
        method that gains a zero-argument super()) and reports the module as
        failed, but the module itself has been re-imported. Every render builds
        new objects from the modules, so re-render regardless.

        The in-place update also rewrites the classes of live objects, which
        crashes on live Qt/OpenGL widgets, so the view is torn down first and
        the reload runs once Qt has deleted it. For the same reason this
        module (the main window itself) is not reloaded: restart the GUI to
        pick up changes to it.
        """
        changed = changed_modules(str(Path(neuronvis.__file__).parent))
        gui_changed = __name__ in changed
        changed = [name for name in changed if name != __name__]
        if len(changed) == 0:
            msg = "Reload source: no changed files"
            if gui_changed:
                msg = f"{__name__:s} changed: restart to apply"
            self.statusBar().showMessage(msg)
            return
        self.clear_viewer()
        self.hoc = None
        # let the event loop process the deleteLater() calls before reloading
        QtCore.QTimer.singleShot(0, lambda: self.finish_reload(changed, gui_changed))

    def finish_reload(self, changed: list, gui_changed: bool) -> None:
        gc.collect()
        failed = []
        for name in changed:
            try:
                pyqtgraph.reload.reload(sys.modules[name])
            except Exception:
                failed.append(name)
        msg = f"Reloaded: {', '.join(changed):s}"
        if len(failed) > 0:
            msg += f"  [in-place update incomplete: {', '.join(failed):s}]"
        if gui_changed:
            msg += f"  [{__name__:s} changed: restart to apply]"
        CP("g", msg)
        if self.value("Input", "Morphology file"):
            self.render()
        self.statusBar().showMessage(msg)


def hoc_header(filename: Path, max_lines: int = 40) -> list:
    """the comment lines at the top of a .hoc/.hocx file (where swc_to_hoc records its transforms)"""
    lines = []
    with open(filename, "r") as fh:
        for line in fh:
            text = line.strip()
            if text.startswith("//"):
                lines.append(text[2:].strip())
            elif text:
                break
            if len(lines) >= max_lines:
                break
    return lines or ["(no header comments)"]


def names_of(vertexes: np.ndarray, vertex: int, hoc: object) -> str:
    """name of the section a geometry vertex belongs to"""
    return list(hoc.sections.keys())[int(vertexes["sec_index"][vertex])]


def arrow_lines(tip: np.ndarray, outward: np.ndarray, length: float) -> list:
    """line segment end points of a 3D arrow pointing at *tip* from the *outward* direction"""
    norm = np.linalg.norm(outward)
    direction = outward / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    # two directions across the arrow, for a four-barbed head visible from any side
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    across1 = np.cross(direction, helper)
    across1 /= np.linalg.norm(across1)
    across2 = np.cross(direction, across1)
    lines = [tip, tip + direction * length]  # shaft
    for across in (across1, -across1, across2, -across2):
        lines += [tip, tip + direction * 0.3 * length + across * 0.12 * length]
    return lines


def changed_modules(prefix: str) -> list:
    """
    names of loaded modules under *prefix* whose source is newer than their
    cached bytecode: the same test pyqtgraph.reload.reloadAll uses
    """
    changed = []
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        cached = getattr(module, "__cached__", None)
        if filename is None or cached is None or not filename.startswith(prefix):
            continue
        if not filename.endswith(".py") or not os.path.isfile(cached):
            continue
        if os.stat(filename).st_mtime > os.stat(cached).st_mtime:
            changed.append(name)
    return sorted(changed)


def make_app() -> QtWidgets.QApplication:
    """
    Create the QApplication with OpenGL context sharing turned on. pyqtgraph
    caches compiled shader programs; without a shared context they are invalid
    in a view created after the previous one is gone (re-render after Reload
    source), giving GL_INVALID_VALUE from glUseProgram.
    Must be called before any QApplication exists.
    """
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return pg.mkQApp("hocRender")


def main() -> None:
    parser = argparse.ArgumentParser(description="hocRender GUI")
    parser.add_argument(
        dest="input_file",
        nargs="?",
        default=None,
        help="Optional .swc, .hoc, .hocx or .p file to render at startup",
    )
    args = parser.parse_args()

    app = make_app()
    win = HocRenderWindow(filename=args.input_file)
    win.show()
    if sys.flags.interactive == 0:
        status = app.exec()
        # Delete the GL widgets while Qt is still alive, then skip interpreter
        # teardown: destroying Qt/OpenGL objects there (after a pyqtgraph view
        # followed by vispy, or after Reload source) crashes with a bus error.
        # Settings were saved and synced in closeEvent.
        sip.delete(win)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)


if __name__ == "__main__":
    main()
