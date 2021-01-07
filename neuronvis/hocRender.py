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

"""

import os, sys, pickle
from pathlib import Path
import argparse
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Union, Dict, List

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import pyqtgraph as pg
from mayavi import mlab
import numpy as np

from pylibrary.tools import fileselector

# import here so we can parse display_modes more quickly
# (and without neuron garbage)
from .hoc_reader import HocReader
from .hoc_viewer import HocViewer
from . import hoc_graphics

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
    "mayavi": "Render with mayavi",
    # "vispy": "render using vispy",
}


# Handle display_modes
##########################################################
# colors are from XKCD color list. Sorry folks.
#

section_colors = {
    "axon": "green",  # in this dict, we handle multiple labels for the same structure.
    "Axon_Initial_Segment": "cyan",
    "initialsegment": "cyan",
    "initseg": "cyan",
    "ais": "cyan",
    "hillock": "dark cyan",
    "Axon_Hillock": "dark cyan",
    "myelinatedaxon": "white",
    "Myelinated_Axon": "white",
    "unmyelinatedaxon": "light cyan",
    "Unmyelinated_Axon": "light cyan",
    "soma": "blue",
    "somatic": "blue",
    "Soma": "blue",
    "apic": "yellow",
    "apical": "yellow",
    "Distal_Dendrite": "yellow",
    "dend": "magenta",
    "dendrite": "magenta",
    "Proximal_Dendrite": "dandelion", # "wintergreen",
    "basal": "magenta",
    "Dendritic_Swelling": "ochre",
    "Dendritic_Hub": "neon red",
    # calyx specific
    "heminode": "green",
    "stalk": "yellow",
    "branch": "blue",
    "neck": "brown",
    "swelling": "magenta",
    "tip": "powder blue",
    "parentaxon": "orange",
    "synapse": "black",
}


class Render(object):
    def __init__(
        self,
        hoc_file:Union[Path, str, None]=None,
        display_style:str="cylinders",
        display_renderer:str="pyqtgraph",
        display_mode:str="sec-type",
        mechanism:Union[str, None]=None,
        fighandle:Union[object, None]=None,
        sim_data:Union[Path, str, None]=None,
        initial_view:list=[200., 0., 0.],
        figsize:list=[1000., 1000.],
        output_file:Union[Path, str, None]=None,
        fax:Union[object, None]=None,  # matplotlib figure axis
        somaonly:bool=False,
        color:str="blue",
        alpha:float=1.0,
        label:Union[str, None]=None,
        flags=None,  # passed to mayavi, probably str, list or object. 
    ) -> None:


        if hoc_file == 'select':
            FS = fileselector.FileSelector(
                title="Select file",
                dialogtype="file",
                extensions=['.hoc', '.hocx', '.swc'],
                startingdir=".",
                useNative=True,
                standalone=False,
            )
            hoc_file = FS.fileName[0]
            if hoc_file is None:
                exit()
        self.color = color
        hoc = HocReader(hoc_file, somaonly=somaonly)
        self.renderer = display_renderer
        self.display_style = display_style
        self.display_mode = display_mode
        self.view = HocViewer(hoc, 
                        camerapos=initial_view,
                        renderer=self.renderer,
                        figsize=figsize,
                        fighandle=fighandle)
        self.label = label
        self.alpha = alpha
        # print("Section groups:")
        # print(self.view.hr.sec_groups.keys())
        if display_style == "volume":
            if self.renderer == "pyqtgraph":
                g = self.view.draw_volume()
            elif self.renderer == "mayavi":
                g = self.view.draw_volume_mayavi()
            else:
                raise ValueError("Can only render volume with pyqtgraph and mayavi")

        elif display_style == "surface":
            g = self.view.draw_surface()
            self.color_map(g, display_mode, colors=section_colors, mechanism=mechanism, alpha=self.alpha)

        elif display_style == "graph":
            if self.renderer == "pyqtgraph":
                g = self.view.draw_graph()
                self.color_map(
                    g, display, mechanism=mechanism, alpha = self.alpha,
                )
            elif   self.renderer== "mpl":
                g = self.view.draw_mpl_graph(fax=fax)
            elif   self.renderer== "mayavi":
                g = self.view.draw_mayavi_graph(
                    color=self.color, label=label, flags=flags
                )
            else:
                raise ValueError("Can only render graph in pyqtgraph, matplotlib and mayavi ")

        elif display_style == "cylinders":
            if   self.renderer== "pyqtgraph":
                g = self.view.draw_cylinders()
                self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)

            elif   self.renderer=="mpl":
                g = self.view.draw_mpl_cylinders(fax=fax)
                self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)
            
            elif   self.renderer== "mayavi":
                g = self.view.draw_mayavi_cylinders(
                    color=section_colors, label=label, flags=flags,
                    mechanism=mechanism,
                )
                # print("g: ", g)
                self.color_map(g, display_mode, mechanism=mechanism, alpha=self.alpha)
                g.g.render()
            else:
                raise ValueError("Can only render cylinders in pyqtgraph, matplotlib and mayavi ")

        elif   self.renderer == "vispy":
            g = self.view.draw_vispy()

        elif display_mode == "vm":

            # Render animation of membrane voltage
            if self.sim_data is None:
                raise Exception("Cannot render Vm: no simulation output specified.")

            surf = self.view.draw_surface()
            start = 375
            stop = 550
            index = start
            loopCount = 0
            nloop = 1

        if   self.renderer== "pyqtgraph":

            import pyqtgraph as pg
            if output_file is not None:
                print(f"Saving to outputfile: {str(output_file):s}")
                # print(dir(self.view))
                img = pg.makeQImage(self.view.renderToArray(size=figsize))
                img.save(output_file)
            elif sys.flags.interactive == 0:
                pg.Qt.QtGui.QApplication.exec_()
                
        if self.renderer== "mayavi":

            print('outputfile: ', output_file)
            if output_file is not None:
                print(f"Saving mayvi rendering to outputfile: {str(output_file):s}")
                # print(dir(self.view))
                f = mlab.gcf()
                mlab.savefig(output_file, figure=f, magnification=1.0) # size=(1000, 1000))
            else:
                mlab.show()
                            
        if self.renderer== "mpl":
            import matplotlib.pyplot as mpl
            mpl.show()

    def color_map(
        self,
        g: object,
        display_mode: str,
        mechanism: Union[str, None] = None,
        colors:dict=section_colors,
        alpha: float = 1.0,
    ) -> None:
        print('set color map')
        assert g is not None

        if display_mode == "sec-type":
            print('sec type with alpha: ', alpha,   self.renderer)
            if   self.renderer == 'pyqtgraph':
                g.set_group_colors(colors, alpha=alpha)
            elif   self.renderer == 'mayavi':
                print('set sectype colors mayavi')
                # g.set_group_colors(colors, alpha=alpha)
        elif display_mode == "mechanism" and (
            
            mechanism != "None" or mechanism is not None
        ):
            print('Setting color map by mechanism: ', mechanism)
            if   self.renderer == 'pyqtgraph':
                g.set_group_colors(colors, mechanism=mechanism)

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
                self.view.save_frame(
                    os.path.join(os.getcwd(), "Video/video_%04d.png" % (i))
                )
                print("%d / %d" % (i, stop))
        finally:
            self.view.save_video()


# rthis needs tro ve called somewhere...
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(self.update)
# timer.start(10.)
# self.record(os.path.join(os.getcwd(), 'video.avi'))


def main() -> None:

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
        choices=["pyqtgraph", "mayavi", "mpl"], # vispy but not really implemented yet
        help="Select thedisplay_renderer(default pyqtgraph)",
    )

    parser.add_argument(
        "-style",
        "-s",
        dest="display_style",
        action="store",
        default="cylinders",
        choices=["cylinders", "graph", "volume", "surface",],
        help="Select the display mode (default: cylinders)",
    )

    parser.add_argument(
        "--mode",
        "-m",
        dest="display_mode",
        action="store",
        default="None",
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
        "--alpha",
        "-a",
        dest="alpha",
        type=float,
        default=0.45,
        help="Select the display alpha",
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
    elif args["input_file"] in ['select', 'file']:
        hoc_file = "select"
    else:
        error()

    Render(
        hoc_file=hoc_file,
        display_style=args["display_style"],
        display_renderer=args["display_renderer"],
        display_mode=args["display_mode"],
        mechanism=args["mechanism"],
        alpha=args["alpha"],
        sim_data=sim_data,
    )



if __name__ == "__main__":
    main()
