#!/usr/bin/python

"""
hocRender : provide visual rendering for morphology and other attributes
as stored in a "hoc" file.
Usage:

h.loadfile(filename) # standard hoc load
# potentially, you would decorate the membrane with biophysical mechanisms here:
decorate(h)

pg.mkQApp()
pg.dbg()
render = hr.hocRender(h) # where h is the NEURON hoc object (from neuron import h)
render.draw_model(modes=['blob'])
render.getSectionLists(Colors.keys()) # if the sections are named...
render.paint_sections_by_density(self.modelPars.calyxColors, self.modelPars.mechNames['CaPCalyx'])
render.show()

2/3/2014
Portions of this code were taken from neuronvisio (http://neuronvisio.org), specifically, to parse
the hoc file connection structure (specifically: getSectionInfo, and parts of drawModel).

Example usage:
hocRender VCN_Rostral_P60_Granule_Cell_Node_List_06-202500000.swc --secmap sbem3 -r pyqtgraph -s cylinders
    -p new_counting_points.csv -m sec-type --sx 0.024 --sy 0.024 --sz 0.07

Reads the swc file, renders it as cylinders using pyqtgraph, colors the sections by type using the "SBEM3" section map,
and plots the points from the csv file "new_counting_points.csv" as green spheres.
   The data can be scaled by the factors sx, sy, sz (default 1.0) to match the
   scale of the data in the swc file (but really, the swc file should be scaled to match the data).

   We ASSUME that the swc file is in micrometers, and the points are in micrometers as well. If
   the swc file is in pixes, then the swc data needs to be scaled.

   Centering is applied to both the swc file and the points, but must be applied to the swc/hocx file
   structure after scaling, so that the points are in the same coodinate system.

"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"
# from mayavi import mlab
import numpy as np
import pyqtgraph as pg
from pylibrary.tools import fileselector
from pyqtgraph import QtGui
from pyqtgraph import opengl as opengl

import neuronvis.renderer_colormaps as rc

# import here so we can parse display_modes more quickly
# (and without neuron garbage)
from .hoc_reader import HocReader
from .hoc_viewer import HocViewer

section_colors = rc.section_colors


# define all display_modes here.
display_mode = {
    "sec-type": "Sections colored by type",
    "vm": "Animation of per-section membrane voltage over time.",
    "mechanisms": "Show distribution of selected mechanism",
}

display_style = {
    "graph": "Simple wireframe rendering.",
    "cylinders": "Simple cylinder rendering.",
    "volume": "simple volume rendering",
    "surface": "uncolored surface rendering.",
}

display_renderers = {
    "pyqtgraph": "render with pyqtgraph",
    "mpl": "render using matplotlib ",
    # "mayavi": "Render with mayavi",
    "vispy": "render using vispy",
}


# Handle and render multiple display_modes


class Render(object):
    def __init__(
        self,
        hoc_file: Union[Path, str, None] = None,
        display_style: str = "cylinders",
        center: bool = False,
        display_renderer: str = "pyqtgraph",
        display_mode: str = "sec-type",
        mechanism: Union[str, None] = None,
        fighandle: Union[object, None] = None,
        sim_data: Union[Path, str, None] = None,
        points: Union[Path, str, None] = None,
        scalexyzr: dict = {"x": 1.0, "y": 1.0, "z": 1.0, "r": 1.0},  # scale x, y, z and r
        initial_view: list = [200.0, 0.0, 0.0],
        figsize: list = [1000.0, 1000.0],
        output_file: Union[Path, str, None] = None,
        verify: bool = False,
        fax: Union[object, None] = None,  # matplotlib figure axis
        somaonly: bool = False,
        color: str = "blue",
        alpha: float = 1.0,
        label: Union[str, None] = None,
        section_map: str = "swc",  # mapping for swc files
        state: Union[object, None] = None,
        flags=None,  # passed to mayavi, probably str, list or object.
    ) -> None:

        self.section_colors = section_colors
        if hoc_file == "select":
            FS = fileselector.FileSelector(
                title="Select file",
                dialogtype="file",
                extensions=[".hoc", ".hocx", ".swc"],
                startingdir=".",
                useNative=True,
                standalone=False,
            )
            hoc_file = FS.fileName[0]
            if hoc_file is None:
                exit()
        self.color = color
        self.renderer = display_renderer
        self.center = center
        self.scalexyzr = scalexyzr
        self.display_style = display_style
        self.display_mode = display_mode
        self.points = points
        self.label = label
        self.alpha = alpha
        self.verify = verify
        self.state = state  # vispy object state for display turntable
        hoc = HocReader(
            hoc_file,
            somaonly=somaonly,
            section_map=section_map,
            center=self.center,
            scale=self.scalexyzr,
            verify=verify,
        )
        if self.points is not None:
            self.counting_point_data = pd.read_csv(self.points)
            self.counting_point_data["x"] -= hoc.centerpos["x"]
            self.counting_point_data["y"] -= hoc.centerpos["y"]
            self.counting_point_data["z"] -= hoc.centerpos["z"]
        else:
            self.counting_point_data = None

        title = str(Path(hoc_file).name)
        self.view = HocViewer(
            hoc,
            camerapos=initial_view,
            renderer=self.renderer,
            figsize=figsize,
            fighandle=fighandle,
        )
        # print("section_map: ", section_map)
        # print("display_style: ", display_style)
        # print("renderer: ", self.renderer)
        match display_style:
            case "volume":
                if self.renderer == "pyqtgraph":
                    g = self.view.draw_volume()
                elif self.renderer == "mayavi":
                    g = self.view.draw_volume_mayavi()
                else:
                    raise ValueError("Can only render volume with pyqtgraph and mayavi")

            case "surface":
                g = self.view.draw_surface()
                self.color_map(
                    g,
                    display_mode,
                    section_map=section_map,
                    colors=section_colors,
                    mechanism=mechanism,
                    alpha=self.alpha,
                )

            case "graph":
                if self.renderer == "pyqtgraph":
                    g = self.view.draw_graph()
                    self.color_map(
                        g,
                        display_style,
                        mechanism=mechanism,
                        alpha=self.alpha,
                    )
                    if self.points is not None and self.counting_point_data is not None:
                        g.scatter(
                            self.counting_point_data["x"],
                            self.counting_point_data["y"],
                            self.counting_point_data["z"],
                            color=self.counting_point_data["color"],
                            size=32,
                        )
                elif self.renderer == "mpl":
                    g = self.view.draw_mpl_graph(fax=fax)
                    if self.points is not None and self.counting_point_data is not None:
                        fax[1].scatter(
                            self.counting_point_data["x"],
                            self.counting_point_data["y"],
                            self.counting_point_data["z"],
                            color=self.counting_point_data["color"],
                            s=10,
                        )

                elif self.renderer == "mayavi":
                    g = self.view.draw_mayavi_graph(color=self.color, label=label, flags=flags)
                else:
                    raise ValueError("Can only render graph in pyqtgraph, matplotlib and mayavi ")

            case "cylinders":
                if self.renderer == "pyqtgraph":
                    g = self.view.draw_cylinders()
                    g.setShader("balloon")
                    # g.setGLOptions("additive")
                    self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)
                    if self.points is not None and self.counting_point_data is not None:

                        if "darwin" in sys.platform:
                            print("Darwin detected, setting OpenGL format")
                            fmt = QtGui.QSurfaceFormat()
                            fmt.setRenderableType(fmt.RenderableType.OpenGL)
                            fmt.setProfile(fmt.OpenGLContextProfile.CoreProfile)
                            fmt.setVersion(4, 1)
                            QtGui.QSurfaceFormat.setDefaultFormat(fmt)

                        presyns = np.array(
                            [
                                self.counting_point_data["x"],
                                self.counting_point_data["y"],
                                self.counting_point_data["z"],
                            ]
                        ).T
                        colors = [pg.mkColor(c) for c in self.counting_point_data["color"].values]
                        self.pg_SP = opengl.GLScatterPlotItem(
                            pos=presyns,
                            size=1,
                            color=colors[0],
                            pxMode=False,
                        )

                        self.view.addItem(self.pg_SP)

                elif self.renderer == "mpl":
                    g = self.view.draw_mpl_cylinders(fax=fax, colors=section_colors)
                    # self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)
                    if self.points is not None and self.counting_point_data is not None:
                        fax[1].scatter(
                            self.counting_point_data["x"],
                            self.counting_point_data["y"],
                            self.counting_point_data["z"],
                            color=self.counting_point_data["color"],
                            s=10,
                        )
                elif self.renderer == "vispy":
                    g = self.view.draw_vispy(
                        mechanism=mechanism,
                        color=section_colors,
                        state=self.state,
                        title=title,
                        headlight=True,
                    )

                elif self.renderer == "mayavi":
                    g = self.view.draw_mayavi_cylinders(
                        color=section_colors,
                        label=label,
                        flags=flags,
                        mechanism=mechanism,
                    )
                    self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)
                    g.g.render()
                else:
                    raise ValueError(
                        "Can only render cylinders in pyqtgraph, matplotlib, vispy and mayavi "
                    )

        if display_mode == "vm":

            # Render animation of membrane voltage
            if self.sim_data is None:
                raise Exception("Cannot render Vm: no simulation output specified.")

            # unused variables?
            # surf = self.view.draw_surface()
            # stop = 550
            # index = start
            # loopCount = 0
            # nloop = 1
            # start = 375

        if self.renderer == "pyqtgraph":

            import pyqtgraph as pg

            if output_file is not None:
                print(f"Saving to outputfile: {str(output_file):s}")
                img = pg.makeQImage(self.view.renderToArray(size=figsize))
                img.save(output_file)
            elif sys.flags.interactive == 0:
                pg.Qt.QtGui.QGuiApplication.exec()

        if self.renderer == "mayavi":
            if output_file is not None:
                print(f"Saving mayvi rendering to outputfile: {str(output_file):s}")
                f = mlab.gcf()
                mlab.savefig(output_file, figure=f, magnification=1.0)  # size=(1000, 1000))
            else:
                mlab.show()

        if self.renderer == "mpl":
            import matplotlib.pyplot as mpl

            mpl.show()

    def color_map(
        self,
        g: object,
        display_mode: str,
        mechanism: Union[str, None] = None,
        section_map: str = "swc",
        colors: dict = section_colors,
        alpha: float = 1.0,
    ) -> None:
        assert g is not None

        if display_mode == "sec-type":
            if self.renderer == "pyqtgraph":
                g.set_group_colors(colors, alpha=alpha)
                # self.view.setBackground(0xddddddff)
            elif self.renderer == "mayavi":
                pass
                # g.set_group_colors(colors, alpha=alpha)

        elif display_mode == "mechanism" and (mechanism != "None" or mechanism is not None):
            if self.renderer == "pyqtgraph":
                g.set_group_colors(colors, mechanism=mechanism)
            else:
                raise ValueError("Can only render mechanism density with pyqtgraph")

    def vm_to_color(self, v: np.ndarray) -> np.ndarray:
        """
        Convert an array of Vm to array of representative colors
        """
        color = np.empty((v.shape[0], 4), dtype=float)
        v_min = -80  # mV
        v_range = 100.0  # mV range in scale
        v = (v - v_min) / v_range
        color[:, 0] = v  # R
        color[:, 1] = 1.5 * abs(v - 0.5)  # G
        color[:, 2] = 1.0 - v  # B
        color[:, 3] = 0.1 + 0.8 * v  # alpha
        return color

    def set_index(self, index: int) -> None:
        """
        Set the currently-displayed time index.
        """
        # v = sim_data.data['Vm'][:,index]
        v = self.sim_data.data[:, index]
        color = vm_to_color(v)

        # note that we assume sections are ordered the same in the HocReader
        # as they are in the results data, but really we should use
        # sim_data.section_map to ensure this is True.
        surf.set_section_colors(color)

    def update(self) -> None:
        global index, start, stop, sim_data, surf, loopCount, nloop

        self.set_index(index)

        index += 1
        if index >= stop:
            loopCount += 1
            if loopCount >= nloop:
                timer.stop()
            index = start

    def record(self, file_name: str) -> None:
        """
        Record a video from *start* to *stop* with the current view
        configuration.
        """
        timer.stop()
        self.view.begin_video(file_name)
        try:
            for i in range(start, stop):
                self.set_index(i)
                pg.Qt.QtGui.QApplication.processEvents()
                self.view.save_frame(os.path.join(os.getcwd(), "Video/video_%04d.png" % (i)))
                print("%d / %d" % (i, stop))
        finally:
            self.view.save_video()


# rthis needs tro ve called somewhere...
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(self.update)
# timer.start(10.)
# self.record(os.path.join(os.getcwd(), 'video.avi'))


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(
        description="Hoc Rendering",
        argument_default=argparse.SUPPRESS,
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        dest="input_file",
        action="store",
        default=None,
        help="Select the hoc file to render (no default)",
    )

    parser.add_argument(
        "--renderer",
        "-r",
        dest="display_renderer",
        action="store",
        default="pyqtgraph",
        choices=["pyqtgraph", "vispy", "mpl"],  # vispy but not really implemented yet
        help="Select thedisplay_renderer(default pyqtgraph)",
    )

    parser.add_argument(
        "--secmap",
        type=str,
        default="sbem3",
        dest="section_map",
        choices=["swc", "sbem", "sbem2", "sbem3", "sbem4"],
        help="Choose section mapping",
    )

    parser.add_argument(
        "--style",
        "-s",
        dest="display_style",
        action="store",
        default="cylinders",
        choices=[
            "cylinders",
            "graph",
            "volume",
            "surface",
        ],
        help="Select the display mode (default: cylinders)",
    )

    parser.add_argument(
        "--mode",
        "-m",
        dest="display_mode",
        action="store",
        default="sec-type",
        choices=["vm", "sec-type", "mechanism"],
        help="Select the display mode (default: None)",
    )

    parser.add_argument(
        "--mechanism",
        "-M",
        dest="mechanism",
        action="store",
        default="None",
        help="Select the mechanism density to display (default: None)",
    )

    parser.add_argument(
        "--center",
        "-c",
        dest="center",
        action="store_true",
        default=False,
        help="Force first point to be (0,0,0) (default: False)",
    )

    parser.add_argument(
        "--scale",
        "-S",
        dest="scale",
        type=float,
        default=1.0,
        help="Scale the rendering by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--sx",
        dest="scalex",
        type=float,
        default=1.0,
        help="Scale the X rendering by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--sy",
        dest="scaley",
        type=float,
        default=1.0,
        help="Scale the Y rendering by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--sz",
        dest="scalez",
        type=float,
        default=1.0,
        help="Scale the Z rendering by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--sr",
        dest="scaler",
        type=float,
        default=1.0,
        help="Scale the swc radius rendering by this factor (default: 1.0)",
    )

    parser.add_argument(
        "--alpha",
        "-a",
        dest="alpha",
        type=float,
        default=0.45,
        help="Select the display alpha",
    )

    parser.add_argument(
        "-v",
        "--verify",
        dest="verify",
        action="store_true",
        default=False,
        help="print hoc output from swc for verification",
    )

    parser.add_argument(
        "--points",
        "-p",
        dest="points",
        action="store",
        default=None,
        help="Points from a csv file to plot along with rendering (default: None.)",
    )
    args = vars(parser.parse_args())

    hoc_file = None
    sim_data = None
    # read input file(s)
    if args["input_file"].endswith(".p"):
        print("reading input file")
        from .sim_result import SimulationResult

        sim_data = SimulationResult(args["input_file"])
        print("simdata: ", sim_data)
        hoc_file = sim_data.hoc_file
        print("hoc_file: ", hoc_file)
    elif args["input_file"].endswith(".hoc"):
        hoc_file = args["input_file"]
    elif args["input_file"].endswith(".hocx"):
        hoc_file = args["input_file"]
    elif args["input_file"].endswith(".swc"):
        hoc_file = args["input_file"]
    elif args["input_file"] in ["select", "file"]:
        hoc_file = "select"
    else:
        raise ValueError("Input file must be a hoc, hocx, swc or p file.")

    Render(
        hoc_file=hoc_file,
        display_style=args["display_style"],
        display_renderer=args["display_renderer"],
        center=args["center"],
        scalexyzr={
            "x": args.get("scalex", 1.0),
            "y": args.get("scaley", 1.0),
            "z": args.get("scalez", 1.0),
            "r": args.get("scaler", 1.0),
        },
        display_mode=args["display_mode"],
        mechanism=args["mechanism"],
        alpha=args["alpha"],
        verify=args["verify"],
        sim_data=sim_data,
        section_map=args["section_map"],
        points=args.get("points", None),
    )


if __name__ == "__main__":
    main()
