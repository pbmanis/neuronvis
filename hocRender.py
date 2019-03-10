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
from __future__ import print_function
from __future__ import absolute_import


import os, sys, pickle
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5' 
import pyqtgraph as pg
import numpy as np
# import here so we can parse commands more quickly 
# (and without neuron garbage)
from .hoc_reader import HocReader
from .hoc_viewer import HocViewer
from . import hoc_graphics

# define all commands here.
commands = {
    'sec-type': "Sections colored by type",
    'vm': "Animation of per-section membrane voltage over time.",
    'graph': "Simple wireframe rendering.",
    'cylinders': "Simple cylinder rendering.",
    'volume': "simple volume rendering",
    'surface': "uncolored surface rendering.",
    'mpl': "render using matplotlib (cylinders)",
    'mpl_graph': "render using matplotlib (lines)",
    'mayavi': "Render with mayavi",
    'vispy': 'render using vispy'
    }

# Handle command line arguments.
# might introduce real command line parsing later..
##########################################################

def error():
    print("Usage: hocRender input_file command")
    print("  input_file may be either *.hoc defining section properties or")
    print("  *.p containing simulation results.\n")
    print("Available commands:")
    for cmd,desc in commands.items():
        print("  "+cmd+":")
        print("\n".join(["    "+line for line in desc.split("\n")]))
    sys.exit(-1)



# Handle commands
##########################################################

section_colors = {
    'axon': 'g', 
    'hillock': 'r',
    'initialsegment': 'c',
    'myelinatedaxon': 'white',
    'unmyelinatedaxon': 'orange',
    'soma': 'b',
    'somatic': 'b',
    'apic': 'y',
    'apical': 'y',
    'dend': 'm',
    'basal': 'm',
    'initseg': 'c',
    'ais': 'c',
    'heminode': 'g', 
    'stalk':'y', 
    'branch': 'b', 
    'neck': 'brown',
    'swelling': 'magenta', 
    'tip': 'powderblue', 
    'parentaxon': 'orange', 
    'synapse': 'k'}


class Render(object):
    def __init__(self, command='sec-type', renderer='pyqtgraph', fighandle=None, hoc_file=None, 
                       sim_data=None, fax=None, somaonly=False, color='blue', label=None, flags=None):

        self.color = color
        hoc = HocReader(hoc_file, somaonly=somaonly)
        self.renderer = renderer
        self.view = HocViewer(hoc, renderer=renderer, fighandle=fighandle)
        self.label = label
        
        # print("Section groups:")
        # print(self.view.hr.sec_groups.keys())
        if command == 'volume':
            if renderer == 'pyqtgraph':
                vol = self.view.draw_volume()
            if renderer == 'mayavi':
                vol = self.view.draw_volume_mayavi()
        if command == 'surface':
            surf = self.view.draw_surface()
        if command == 'sec-type':
            # Color sections by type.
            surf = self.view.draw_surface()
            surf.set_group_colors(section_colors, alpha=0.35)
        
        elif command == 'graph':
            if renderer == 'pyqtgraph':
                g = self.view.draw_graph()
                g.set_group_colors(section_colors)
            elif renderer == 'mpl':
                g = self.view.draw_mpl_graph(fax=fax)
            elif renderer == 'mayavi':
                g = self.view.draw_mayavi_graph(color=self.color, label=label, flags=flags)
        
        elif command == 'cylinders':
            if renderer == 'pyqtgraph':
                g = self.view.draw_cylinders()
            elif renderer == 'mpl':
                g = self.view.draw_mpl(fax=fax)
            elif renderer == 'mayavi':
                g = self.view.draw_mayavi_cylinders(color=self.color, label=label, flags=flags)

        elif command == 'vispy':
            g = self.view.draw_vispy()
    
        elif command == 'vm':

            # Render animation of membrane voltage
            if self.sim_data is None:
                raise Exception('Cannot render Vm: no simulation output specified.')

            surf = self.view.draw_surface()
            start = 375
            stop = 550
            index = start
            loopCount = 0
            nloop = 1


    def vm_to_color(self, v):
        """
        Convert an array of Vm to array of representative colors
        """
        color = np.empty((v.shape[0], 4), dtype=float)
        v_min = -80 # mV
        v_range = 100. # mV range in scale
        v = (v - v_min) / v_range
        color[:,0] = v     # R
        color[:,1] = 1.5*abs(v-0.5) # G
        color[:,2] = 1.-v # B
        color[:,3] = 0.1+0.8*v # alpha
        return color

    def set_index(self, index):
        """
        Set the currently-displayed time index.
        """
        #v = sim_data.data['Vm'][:,index]
        v = self.sim_data.data[:,index]
        color = vm_to_color(v)
        
        # note that we assume sections are ordered the same in the HocReader 
        # as they are in the results data, but really we should use 
        # sim_data.section_map to ensure this is True.
        surf.set_section_colors(color)
        

    def update(self):
        global index, start, stop, sim_data, surf, loopCount, nloop

        self.set_index(index)

        index += 1
        if index >= stop:
            loopCount += 1
            if loopCount >= nloop:
                timer.stop()
            index = start


    def record(self, file_name):
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
                self.view.save_frame(os.path.join(os.getcwd(), 'Video/video_%04d.png' % (i)))
                print("%d / %d" % (i, stop))
        finally:
            self.view.save_video()


# rthis needs tro ve called somewhere...
    # timer = pg.QtCore.QTimer()
    # timer.timeout.connect(self.update)
    # timer.start(10.)
    # self.record(os.path.join(os.getcwd(), 'video.avi'))


def main():

    
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print('file not found')
        error()
    input_file = sys.argv[1]

    if len(sys.argv) > 2:
        command = sys.argv[2]
        if command not in commands:
            error()
    else:
        command = 'sec-type'


    hoc_file = None
    sim_data =None
    # read input file(s)
    if input_file.endswith('.p'):
        print('reading input file')
        from .sim_result import SimulationResult
        sim_data = SimulationResult(input_file)
        print('simdata: ', sim_data)
        hoc_file = sim_data.hoc_file
        print('hoc_file: ', hoc_file)
    elif input_file.endswith('.hoc'):
        hoc_file = input_file
    else:
        error()
    
    Render(command=command, hoc_file=hoc_file, sim_data=sim_data)

    if sys.flags.interactive == 0:
        import pyqtgraph as pg
        pg.Qt.QtGui.QApplication.exec_()
    
if __name__ == '__main__':
    main()
    
