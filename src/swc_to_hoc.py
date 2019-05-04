# -*- coding: utf8 -*-
import numpy as np
from pathlib import Path
import datetime

"""
SWC File format from CNIC:

n T x y z R P
n is an integer label that identifies the current point and increments by one from one line to the next.

T is an integer representing the type of neuronal segment, such as soma, axon, apical dendrite, etc. The standard accepted integer values are given below.

0 = undefined
1 = soma
2 = axon
3 = dendrite
4 = apical dendrite
5 = fork point
6 = end point
7 = custom
x, y, z gives the cartesian coordinates of each node.

R is the radius at that node.

P indicates the parent (the integer label) of the current point or -1 to indicate an origin (soma).

Python 3 version only 3-27-2019 pbm
Handles Singleton "sections" in swc file by inserting the last parent segment information.

"""


class SWC(object):
    """
    Encapsulates a morphology tree as defined by the SWC standard.
    
    Parameters
    ----------
    filename : str or None
        The name of an swc file to load
    types : dict or None
        A dictionary mapping {type_id: type_name} that describes the type IDs
        in the swc data (second column).
    data : ndarray or None
        Optionally, a data array may be provided instead of an swc file. This
        is used internally.
    scales : dict or None
        dict of format: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'r': 1.0} to provide
        appropriate scaling along each of the axes.
    """
    
    def __init__(self, filename=None, types=None, data=None, scales=None):
        self._dtype = [
            ('id', int), 
            ('type', int), 
            ('x', float), 
            ('y', float), 
            ('z', float), 
            ('r', float), 
            ('parent', int)
        ]
        
        self._id_lookup = None
        self._sections = None
        self._children = None
        self.scales = scales
        
        self.sectypes = {
          #  0: 'undefined',
            1: 'soma',
            2: 'axon',
            3: 'dendrite', # 'basal_dendrite',
            4: 'dendrite', # 'apical_dendrite',
            5: 'dendrite', #'custom', # (user-defined preferences)
            #6: 'unspecified_neurites',
            #7: 'glia_processes', # who knows why this is in here… 
            10: 'hillock',
            11: 'unmyelinatedaxon',
            12: 'dendrite', # 'hub',
           # 13: 'proximal_dendrite',
            #14: 'distal_dendrite',
        }
        
        self.converts = {'basal_dendrite': 'dendrite',
                    'apical_dendrite': 'dendrite',
                    'hub': 'dendrite',
                    'proximal_dendrite': 'dendrite',
                    'distal_dendrite': 'dendrite',
                }
        
        if types is not None:
            self.sectypes.update(types)
        
        if data is not None:
            self.data = data
        elif filename is not None:
            self.load(filename.with_suffix('.swc'))
            self.filename = filename
        else:
            raise TypeError("Must initialize with filename or data array.")

        self.sort()
        
    def load(self, filename):
        self.filename = filename
        print(f"Loading: {str(filename):s}")
        self.data = np.loadtxt(filename, dtype=self._dtype)
        if self.scales is not None:
            self.scale(x=scales['x'], y=scales['y'], z=scales['z'], r=scales['r'])
        

    def copy(self):
        return SWC(data=self.data.copy(), types=self.sectypes)

    @property
    def lookup(self):
        """
        Return a dict that maps *id* to *index* in the data array.
        """
        if self._id_lookup is None:
            self._id_lookup = dict([(rec['id'], i) for i, rec in enumerate(self.data)])
            #self._id_lookup = {}
            #for i, rec in enumerate(self.data):
                #self._id_lookup[rec['id']] = i
        return self._id_lookup

    def children(self, ident):
        """
        Return a list of all children of the node *id*.
        """
        if self._children is None:  # build the child dict
            self._children = {}
            for rec in self.data:
                self._children.setdefault(rec['parent'], [])
                self._children[rec['parent']].append(rec['id'])
        # print('children: ', self._children)
        return self._children.get(ident, [])

    def __getitem__(self, id):
        """
        Return record for node *id*.
        """
        return self.data[self.lookup[id]]

    def reparent(self, id):
        """
        Rearrange tree to make *id* the new root parent.
        """
        d = self.data
        
        # bail out if this is already the root
        if self[id]['parent'] == -1:
            return
        
        parent = -1
        while id != -1:
            oldparent = self[id]['parent']
            self[id]['parent'] = parent
            parent = id
            id = oldparent
            
        self._children = None
        self.sort()
        
    @property
    def sections(self):
        """Return lists of IDs grouped by topological section.
        The first item in each list connects to the last item in a previous
        list.
        """
        # print('self.data: ', self.data)
        if self._sections is None:
            sections = []
            sec = []
            
            # find all nodes with nore than 1 child
            branchpts = set()
            endpoints = set(self.data['id'])
            endpoints.add(-1)
            seen = set()
            for r in self.data:
                p = r['parent']
                if p in seen:
                    branchpts.add(p)
                else:
                    seen.add(p)
                    endpoints.remove(p)
            
            # build lists of unbranched node chains
            lasttype = self.data['type'][0]
            for r in self.data:
                sec.append(r['id'])
                if r['id'] in branchpts or r['id'] in endpoints or r['type'] != lasttype:
                    sections.append(sec)
                    sec = []
                    lasttype = r['type']
            
            self._sections = sections
            
        return self._sections
        
    def connect(self, parent_id, swc):
        """
        Combine this tree with another by attaching the root of *swc* as a 
        child of *parent_id*.
        """
        data = swc.data.copy()
        shift = self.data['id'].max() + 1 - data['id'].min()
        data['id'] += shift
        rootmask = data['parent'] == -1
        data['parent'] += shift
        data['parent'][rootmask] = parent_id
        
        self.data = np.concatenate([self.data, data])
        self._children = None
        self.sort()
        
    def set_type(self, typ):
        self.data['type'] = typ
        
    def write_hoc(self, filename, types=None):
        """
        Write data to a HOC file.
        Each node type is written to a separate section list.
        """
        hoc = []
        # Add some header information
        hoc.extend([f"// Translated from SWC format by: swc_to_hoc.py"])
        hoc.append(f"// Source file: {str(self.filename):s}")
        hoc.append(f"// {datetime.datetime.now().strftime('%B %d %Y, %H:%M:%S'):s}")
        if self.scales is None:
            hoc.append(f"// No scaling")
        else:
            hocappend(f"// Scaling: x: {self.scales['x']:f}, y: {self.scales['y']:f}, z: {self.scales['z']:f}, r: {self.scales['r']:f}")
        hoc.append('')
        sectypes = self.sectypes.copy()
        print('sectypes: ', sectypes)
        for t in np.unique(self.data['type']):
            print(t)
            if t not in sectypes:
                sectypes[t] = 'type_%d' % t
        # create section lists
        screated = []
        for t in list(sectypes.values()):
            if t in screated:
                continue
            hoc.extend([f"objref {t:s}\n{t:s} = new SectionList()"])
            screated.append(t)
        hoc.append('')
        # create sections
        sects = self.sections

        hoc.append(f'create sections[{len(sects):d}]')
        sec_ids = {}
        
        for i, sec in enumerate(sects):
            # remember hoc index for this section
            endpt = self[sec[-1]]['id']
            sec_id = len(sec_ids)
            sec_ids[endpt] = sec_id
            
            # add section to list
            hoc.append(f'access sections[{sec_id:d}]')
            typ = self[sec[0]]['type']
            hoc.append(f'{sectypes[typ]:s}.append()')
            
            # connect section to parent
            p = self[sec[0]]['parent']
            if p != -1:
                hoc.append(f'connect sections[{sec_id:d}](0), sections[{sec_ids[p]:d}](1)')

            # set up geometry for this section
            hoc.append('sections[%d] {' % sec_id)
            if len(sec) == 1:
                seg = sects[sec_ids[p]][-1] # get last segement in the parent section
                rec = self[seg]
                if rec['r'] < 0.05:
                    print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                    rec['r'] = 0.05
                hoc.append(f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})  // seg={seg:d} Singleton repair: to section[sec_ids[p]:d]")
            for seg in sects[sec_id]:
                rec = self[seg]
                if rec['r'] < 0.05:
                    print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                    rec['r'] = 0.05
                hoc.append(f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})   // seg={seg:d}")
            hoc.append('}')
            
            hoc.append('')
        
        open(filename, 'w').write('\n'.join(hoc))
        print(f"Wrote {str(filename):s}")
        
    @property
    def root(self):
        """
        ID of the root node of the tree.
        """
        ind = np.argwhere(self.data['parent'] == -1)[0, 0]
        return self.data[ind]['id']

    def sort(self):
        """
        Sort the tree in topological order.
        """
        order = self.branch(self.root)
        lt = self.lookup
        indexes = np.array([lt[i] for i in order], dtype=int)
        self.data = self.data[indexes]
        
        self._id_lookup = None
        self._sections = None
        
    def path(self, node):
        path = [node]
        while True:
            node = self[node]['parent']
            if node < 0:
                return path
            path.append(node)

    def scale(self, x, y, z, r):
        self.data['x'] *= x
        self.data['y'] *= y
        self.data['z'] *= z
        self.data['r'] *= r
        
    def translate(self, x, y, z):
        self.data['x'] += x
        self.data['y'] += y
        self.data['z'] += z
        
    def branch(self, id):
        """
        Return a list of IDs in the branch beginning at *id*.
        """
        branch = [id]
        for ch in self.children(id):
            branch.extend(self.branch(ch))
        return branch
    
    def topology(self):
        """
        Print the tree topology.
        """
        path = []
        indent = ''
        secparents = [self[s[0]]['parent'] for s in self.sections]
        for i, sec in enumerate(self.sections):
            p = secparents[i]
            if p != -1:
                ind = path.index(p)
                path = path[:ind+1]
                indent = indent[:(ind+1) * 3]
            path.append(self[sec[-1]]['id'])

            # look ahead to see whether subsequent sections are children
            if p in secparents[i+1:]:
                this_indent = indent[:-2] + "├─ "
                indent =      indent[:-2] + "│  │  "
            else:
                this_indent = indent[:-2] + "└─ "
                indent =      indent[:-2] + "   │  "
                
            
            typ = self.sectypes[self[sec[0]]['type']]
            if len(sec) > 10:
                secstr = "%s,...%s" % (str(tuple(sec[:3]))[:-1], str(tuple(sec[-3:]))[1:])
            else:
                secstr = str(tuple(sec))
            print("%ssections[%d] type=%s parent=%d %s" % (this_indent, i, typ, p, secstr))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 1:
        exit()
    s = SWC(filename=Path(sys.argv[1]))
    # s.topology()
    s.write_hoc(Path(sys.argv[1]).with_suffix('.hoc'))
