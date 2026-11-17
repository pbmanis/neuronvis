"""
hocRender_gui : a single-window GUI front end for hocRender.

The left pane is a pyqtgraph ParameterTree holding every hocRender command line
option plus a file chooser; the right pane holds the rendered morphology.
A triangle mesh (.obj, e.g. from SyGlass) can be shown alone or on top of the
morphology (pyqtgraph and vispy renderers). Imaris filament tracings (.ims) are
converted to SWC (see imaris_reader) and rendered like an swc file.

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
import gc
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Union

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.reload
from pyqtgraph import opengl as opengl
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pylibrary.tools.cprint import cprint as CP
from PyQt6 import sip

import neuron
import neuronvis
import neuronvis.renderer_colormaps as rc
from neuronvis import hoc_reader, hoc_viewer, imaris_reader, obj_reader, swc_sectypes

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
        self.mesh_cache = None  # (path, mtime, vertices, faces) of the last mesh read
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
        self.params.child("Actions", "Render").sigActivated.connect(self.render)
        self.params.child("Actions", "Reset view").sigActivated.connect(self.reset_view)
        self.params.child("Actions", "Save image...").sigActivated.connect(self.save_image)
        self.params.child("Actions", "Export SWC...").sigActivated.connect(self.export_swc)
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
        offset = np.zeros(3)  # centering shift applied to the morphology
        if morphology_file is not None:
            self.load_morphology(morphology_file, notes)
            offset = np.array([self.hoc.centerpos[k] for k in "xyz"])
        points = self.read_points(offset)
        mesh = None
        if mesh_file is not None:
            vertices, faces = self.read_mesh(mesh_file)
            mesh = (vertices - offset.astype(np.float32), faces)

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

        names = " + ".join(f.name for f in (morphology_file, mesh_file) if f is not None)
        self.setWindowTitle(f"hocRender - {names:s}")
        msg = f"Rendered {names:s} ({renderer:s}"
        msg += f", {style:s}, {mode:s})" if self.hoc is not None else ")"
        if len(notes) > 0:
            msg += "  [" + "; ".join(notes) + "]"
        self.statusBar().showMessage(msg)

    def load_morphology(self, filename: Path, notes: list) -> None:
        """read a .swc, .hoc, .hocx, .ims or .p file into NEURON and self.hoc"""
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
        clear_neuron_sections()
        if hoc_file.suffix in [".hoc", ".hocx"]:
            force_load_hoc(hoc_file, verbose=verbose)
        self.hoc = hoc_reader.HocReader(
            str(hoc_file),
            section_map=section_map,
            center=self.value("Geometry", "Center"),
            scale={
                "x": self.value("Geometry", "Scale X"),
                "y": self.value("Geometry", "Scale Y"),
                "z": self.value("Geometry", "Scale Z"),
                "r": self.value("Geometry", "Scale radius"),
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

    def convert_imaris(self, filename: Path, notes: list) -> Path:
        """
        Convert the selected filament objects of an .ims file to an SWC file in
        the temporary directory and return its path. Only the parts connected
        to the soma are kept: swc_to_hoc builds a single tree.
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
        swc_file, n_fragments, n_points = imaris_reader.ims_to_swc(
            filename, Path(swc_dir, filename.stem + ".swc"), names, connected_only=True
        )
        if n_fragments > 0:
            notes.append(f"{n_fragments:d} piece(s) ({n_points:d} points) not connected to the soma not shown")
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
        """read an .obj mesh, reusing the last one read if the file has not changed"""
        mtime = filename.stat().st_mtime
        if self.mesh_cache is None or self.mesh_cache[:2] != (filename, mtime):
            vertices, faces = obj_reader.read_obj(filename)
            self.mesh_cache = (filename, mtime, vertices, faces)
        return self.mesh_cache[2], self.mesh_cache[3]

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
            # HocGraph/HocSurface.set_section_colors expect an array, not the group
            # dict; hocRender does not color them either.
            if style == "cylinders":
                if mode == "sec-type":
                    g.set_group_colors(rc.section_colors, alpha=alpha)
                elif mode == "mechanism" and mechanism is not None:
                    g.set_group_colors(rc.section_colors, mechanism=mechanism)
            else:
                notes.append(f"{style:s} is drawn uncolored")

        if points is not None:
            colors = np.array([pg.mkColor(c).getRgbF() for c in points["color"].values])
            scatter = opengl.GLScatterPlotItem(
                pos=points[["x", "y", "z"]].to_numpy(),
                size=1,
                color=colors,
                pxMode=False,
            )
            view.addItem(scatter)

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

        pos = np.concatenate(extents, axis=0)
        self.bounds = (pos.min(axis=0), pos.max(axis=0))
        self.reset_view()

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
        else:  # mesh only
            self.vispy_canvas = embedded_scene_canvas(keys="interactive", bgcolor=[0.75, 0.75, 0.75, 1])
            self.vispy_view = self.vispy_canvas.central_widget.add_view()
            self.vispy_view.camera = "arcball"
            vispy.scene.visuals.XYZAxis(parent=self.vispy_view.scene).transform = (
                vispy.visuals.transforms.STTransform(scale=(10, 10, 10))
            )
            self.vispy_canvas.show()
            extents = []

        if mesh is not None:
            vertices, faces = mesh
            mesh_visual = vispy.scene.visuals.Mesh(
                vertices=vertices, faces=faces, color=self.mesh_rgba(), shading="smooth"
            )
            # translucent and drawn after the morphology so that it shows through
            mesh_visual.set_gl_state("translucent", depth_test=True, cull_face=False)
            self.vispy_view.add(mesh_visual)
            extents.append(vertices)
        pos = np.concatenate(extents, axis=0)
        self.bounds = (pos.min(axis=0), pos.max(axis=0))
        self.reset_view()

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        """center the camera on the cell and back off to see all of it"""
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
        # keep everything in the exported file; unconnected pieces become separate trees
        swc_file, n_fragments, n_points = imaris_reader.ims_to_swc(filename, swc_file, names)
        msg = f"Exported {', '.join(names):s} to {str(swc_file):s}"
        if n_fragments > 0:
            msg += f"  [{n_fragments:d} unconnected piece(s), {n_points:d} points, as separate trees]"
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
