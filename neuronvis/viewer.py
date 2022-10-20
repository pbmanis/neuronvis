import sys
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import ephys as EP
import pylibrary.tools.tifffile as tifffile
from pylibrary.tools import fileselector
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import Qt
from pyqtgraph.parametertree import Parameter, ParameterTree
from neuronvis import swc_to_hoc  # tools for swc reading

morphpath = "/Users/pbmanis/Desktop/Python/VCN-SBEM-Data/MorphologyFiles"
fn = Path(morphpath, "MouseDCN/NF107Ai322018103020XYFPbiocytinCell2.czi - C=1-e.tif")
swcf = Path(morphpath, "MouseDCN/NF107Ai322018103020XYFPbiocytinCell2_partial.swc")

class Viewer():
    def __init__(self):
        self.testfns = False
        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        # Enable High DPI display with PyQt5
        # self.app.setAttribute(Qt.AA_EnableHighDpiScaling)
#         if hasattr(QStyleFactory, 'AA_UseHighDpiPixmaps'):
#             self.app.setAttribute(Qt.AA_UseHighDpiPixmaps)

        self.win = pg.QtGui.QWidget()
        layout = pg.QtGui.QGridLayout()
        self.win.setLayout(layout)
        self.win.setWindowTitle("viewer")
        self.win.resize(1024, 800)

        self.view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(0, 0, 0))
        self.view.setCentralItem(l)
        layout.addWidget(self.view, 0, 1, 8, 8)
        self.imv = pg.ImageView()
        layout.addWidget(self.imv, 0, 1, 8, 8)  # data plots on right
        self.win.show()
        if not self.testfns:
            FS = fileselector.FileSelector(
                title="Select file",
                dialogtype="file",
                extensions='.tif',
                startingdir=".",
                useNative=True,
                standalone=False,
            )
            self.basepath = Path(
                Path(FS.fileName[0]).parent
            )  # Path('/Users/pbmanis/Desktop/Python/cnmodel_models/cache/bushy/rateintensity/cftone/')
            print(FS.fileName)
            tif = tifffile.TiffFile(str(FS.fileName[0]))
        else:
            tif = tifffile.TiffFile(fn)
        images = tif.asarray()
        # images = np.rot90(images)
        for i, img in enumerate(images):
            images[i] = np.flipud(np.rot90(img, 1))

        self.imv.setImage(images)
        if not self.testfns:
            FS2 = fileselector.FileSelector(
                title="Select SWC file",
                dialogtype="file",
                extensions='.swc',
                startingdir=self.basepath,
                useNative=True,
                standalone=False,
            )
            self.basepath = Path(
                FS2.fileName[0]
            )  # Path('/Users/pbmanis/Desktop/Python/cnmodel_models/cache/bushy/rateintensity/cftone/')
            print(FS2.fileName)
            # read the swc file, then draw it on the image using ROIs
            swc = swc_to_hoc.SWC(Path(FS2.fileName[0]))
        else:
            swc = swc_to_hoc.SWC(Path(swcf))
        # print(swc.data)
        self.draw_rois(swc)
        for r in range(len(self.drawing)):
            self.imv.addItem(self.drawing[r])
            # print(self.drawing[r].lines)
            for l in self.drawing[r].lines:
                handles = l.getHandles()
                # print('# handles: ', len(handles))  # usually 3 for the rectangles
                for h in handles:
                    # h.hide()
                #     print(dir(h))
                    # print(h.types)
                    index = l.indexOfHandle(h)
                    h0 = handles[index]
                    # print(dir(h0))
                    # print(h0.type)
                    # print(h0.typ)
                    if h0.typ == 'sr': # rotation handle
                        h.hide()
                    # print(h.radius)
                    h.radius = 2.0
                # return

    def draw_rois(self, swc):
        hoc = []
        sects = swc.sections
        sec_ids = {}
        self.drawing = []
        nsec = len(swc.sections)
        if self.testfns:
            pix = 0.625
        else:
            pix = 1.0

        for i, sec in enumerate(swc.sections):
            # remember hoc index for this section
            endpt = swc[sec[-1]]["id"]
            sec_id = len(sec_ids)
            sec_ids[endpt] = sec_id
            p = swc[sec[0]]["parent"]
            if p != -1:
                hoc.append(
                    f"connect sections[{sec_id:d}](0), sections[{sec_ids[p]:d}](1)"
                )
            pline = []
            xy = []
            w = 0.
            nseg = float(len(sects[sec_id]))
            for j, seg in enumerate(sects[sec_id]):
                rec = swc[seg]
                if j == 0:
                    xy.append([rec['x']/pix, rec['y']/pix])
                    w += rec['r']/nseg
                else:
                    xy.append([rec['x']/pix, rec['y']/pix])
                    w += rec['r']/nseg
                    
                hoc.append(
                    f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})   // seg={seg:d}"
                )
            # draw the section now:
            if len(xy) == 1:
                xy.append(xy[-1])
            # print(xy)
            print(w)
            if w < pix/2.:
                w = pix/2.
            # print(w)
            c = pg.intColor(i, hues=nsec, alpha=144)
            self.drawing.append(pg.MultiRectROI(xy, width=w, pen=pg.mkPen(c)))




# class MultiRectROI(QtGui.QGraphicsObject):
#     """
#     Chain of trapzoidal ROIs connected by handles.
#
#     This is generally used to mark a curved path through
#     an image similarly to PolyLineROI. It differs in that each segment
#     of the chain is a right trapezoid instead of linear and thus has width.
#
#     ============== =============================================================
#     **Arguments**
#     points         (list of length-2 sequences) The list of points in the path.
#     widths          list of (float) The width of the ROIs orthogonal to the path.
#     \**args        All extra keyword arguments are passed to ROI()
#     ============== =============================================================
#     """
#     sigRegionChangeFinished = QtCore.Signal(object)
#     sigRegionChangeStarted = QtCore.Signal(object)
#     sigRegionChanged = QtCore.Signal(object)
#
#     def __init__(self, points, width, pen=None, **args):
#         QtGui.QGraphicsObject.__init__(self)
#         self.pen = pen
#         self.roiArgs = args
#         self.lines = []
#         if len(points) < 2:
#             raise Exception("Must start with at least 2 points")
#
#         ## create first segment
#         self.addSegment(points[1], connectTo=points[0], scaleHandle=True)
#
#         ## create remaining segments
#         for p in points[2:]:
#             self.addSegment(p)
#
#
#     def paint(self, *args):
#         pass
#
#     def boundingRect(self):
#         return QtCore.QRectF()
#
#     def roiChangedEvent(self):
#         w = self.lines[0].state['size'][1]
#         for l in self.lines[1:]:
#             w0 = l.state['size'][1]
#             if w == w0:
#                 continue
#             l.scale([1.0, w/w0], center=[0.5,0.5])
#         self.sigRegionChanged.emit(self)
#
#     def roiChangeStartedEvent(self):
#         self.sigRegionChangeStarted.emit(self)
#
#     def roiChangeFinishedEvent(self):
#         self.sigRegionChangeFinished.emit(self)
#
#     def getHandlePositions(self):
#         """Return the positions of all handles in local coordinates."""
#         pos = [self.mapFromScene(self.lines[0].getHandles()[0].scenePos())]
#         for l in self.lines:
#             pos.append(self.mapFromScene(l.getHandles()[1].scenePos()))
#         return pos
#
#     def getArrayRegion(self, arr, img=None, axes=(0,1), **kwds):
#         rgns = []
#         for l in self.lines:
#             rgn = l.getArrayRegion(arr, img, axes=axes, **kwds)
#             if rgn is None:
#                 continue
#             rgns.append(rgn)
#             #print l.state['size']
#
#         ## make sure orthogonal axis is the same size
#         ## (sometimes fp errors cause differences)
#         if img.axisOrder == 'row-major':
#             axes = axes[::-1]
#         ms = min([r.shape[axes[1]] for r in rgns])
#         sl = [slice(None)] * rgns[0].ndim
#         sl[axes[1]] = slice(0,ms)
#         rgns = [r[sl] for r in rgns]
#         #print [r.shape for r in rgns], axes
#
#         return np.concatenate(rgns, axis=axes[0])
#
#     def addSegment(self, pos=(0,0), scaleHandle=False, connectTo=None):
#         """
#         Add a new segment to the ROI connecting from the previous endpoint to *pos*.
#         (pos is specified in the parent coordinate system of the MultiRectROI)
#         """
#
#         ## by default, connect to the previous endpoint
#         if connectTo is None:
#             connectTo = self.lines[-1].getHandles()[1]
#
#         ## create new ROI
#         newRoi = ROI((0,0), [1, 5], parent=self, pen=self.pen, **self.roiArgs)
#         self.lines.append(newRoi)
#
#         ## Add first SR handle
#         if isinstance(connectTo, Handle):
#             self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5], item=connectTo)
#             newRoi.movePoint(connectTo, connectTo.scenePos(), coords='scene')
#         else:
#             h = self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5])
#             newRoi.movePoint(h, connectTo, coords='scene')
#
#         ## add second SR handle
#         h = self.lines[-1].addScaleRotateHandle([1, 0.5], [0, 0.5])
#         newRoi.movePoint(h, pos)
#
#         ## optionally add scale handle (this MUST come after the two SR handles)
#         if scaleHandle:
#             newRoi.addScaleHandle([0.5, 1], [0.5, 0.5])
#
#         newRoi.translatable = False
#         newRoi.sigRegionChanged.connect(self.roiChangedEvent)
#         newRoi.sigRegionChangeStarted.connect(self.roiChangeStartedEvent)
#         newRoi.sigRegionChangeFinished.connect(self.roiChangeFinishedEvent)
#         self.sigRegionChanged.emit(self)
#
#
#     def removeSegment(self, index=-1):
#         """Remove a segment from the ROI."""
#         roi = self.lines[index]
#         self.lines.pop(index)
#         self.scene().removeItem(roi)
#         roi.sigRegionChanged.disconnect(self.roiChangedEvent)
#         roi.sigRegionChangeStarted.disconnect(self.roiChangeStartedEvent)
#         roi.sigRegionChangeFinished.disconnect(self.roiChangeFinishedEvent)
#
#         self.sigRegionChanged.emit(self)   

def main():
    V = Viewer()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    main()
