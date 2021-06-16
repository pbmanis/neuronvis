# -*- coding: utf8 -*-
import numpy as np
from pathlib import Path
from typing import Union
import datetime
import argparse
import re


"""
SWC File format from CNIC:

n T x y z R P
n is an integer label that identifies the current point 
and increments by one from one line to the next.

T is an integer representing the type of neuronal segment, 
such as soma, axon, apical dendrite, etc. The standard 
accepted integer values are given below.

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

P indicates the parent (the integer label) of the current
 point or -1 to indicate an origin (soma).

Python 3 version only 3-27-2019 pbm
Handles Singleton "sections" in swc file by inserting 
the last parent segment information.

"""

# standard SWC types:
swc_sectypes = {
    #  0: 'undefined',
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "custom",  # (user-defined preferences)
    6: "unspecified_neurites",
    7: "glia_processes",  # who knows why this is in here…
    10: "hillock",
    11: "unmyelinatedaxon",
    12: "dendrite",  # 'hub',
    # 13: 'proximal_dendrite',
    # 14: 'distal_dendrite',
}

# section types for SBEM data on bushy cells (additional definitions)
sbem_sectypes = {
    # new swc mapping
    0: "Undefined",
    1: "soma",
    2: "Myelinated_Axon",
    3: "Basal_Dendrite",
    4: "Apical_Dendrite",
    5: "Custom",
    6: "Unspecified_Neurites",
    7: "Glia_Processes",
    8: "Blank",
    9: "Blank",
    10: "Axon_Hillock",
    11: "Unmyelinated_Axon",
    12: "Dendritic_Hub",
    13: "Proximal_Dendrite",
    14: "Distal_Dendrite",
    15: "Axon_Initial_Segment",
    16: "Axon_Heminode",
    17: "Axon_Node",
    18: "Dendritic_Swelling",
}

# section types for SBEM data on bushy cells (additional definitions)
# This table is for swcs from Syglassfrom May 2021 (who changed it?)
sbem2_sectypes = {
    #new swc mapping
    0: 'Undefined',
    1: 'Soma',
    2: 'Myelinated_Axon',
    3: 'Basal_Dendrite',
    4: 'Apical_Dendrite',
    5: 'Custom',
    6: 'Unspecified_Neurites',
    7: 'Glia_Processe',
    8: 'Blank',
    9: 'Blank',
    10: 'Axon_Hillock',
    11: 'Dendritic_Swelling',
    12: 'Dendritic_Hub',
    13: 'Proximal_Dendrite',
    14: 'Distal_Dendrite',
    15: 'Axon_Initial_Segment',
    16: 'Axon_Heminode',
    17: 'Axon_Node',
}

# crenaming of cell parts to match cnmodel data tables (temporary)
# renaming = {
#     "basal_dendrite": "dendrite",
#     "Basal_Dendrite": "dendrite",
#     "Apical_Dendrite": "dendrite",
#     "apical_dendrite": "dendrite",
#     "proximal_dendrite": "dendrite",
#     "Proximal_Dendrite": "dendrite",
#     "distal_dendrite": "dendrite",
#     "Distal_Dendrite": "dendrite",
#     "Dendritic_Swelling": "dendrite",
#     "hub": "dendrite",
#     "Dendritic_Hub": "dendrite",
#     "Axon_Hillock": "hillock",
#     "Unmyelinated_Axon": "unmyelinatedaxon",
#     "Axon_Initial_Segment": "initialsegment",
#     "Axon_Heminode": "heminode",
#     "Axon_Node": "node",
# }

# when pruning, we remove any section type that is a 
# part of either dendrite or axon.
idsofpart = {
    'dendrite': [3, 4, 12, 13, 14, 18],
    'distal': [12, 14, 18],
    'axon': [2, 10, 11, 15, 16, 17],
    'soma': [1],
}

partsof = {
    "dendrite": ["dendrite",
                "basal_dendrite"
                "Basal_Dendrite",
                "Apical_Dendrite", 
                "apical_dendrite", 
                "proximal_dendrite",
                "Proximal_Dendrite"
                "distal_dendrite",
                "Distal_Dendrite",
                "Dendritic_Swelling",
                "hub",
                "Dendritic_Hub"],
    "axon": ["Axon_Hillock", 
            "hillock",
            "Unmyelinated_Axon",
            "unmyelinatedaxon",
            "Axon_Initial_Segment",
            "initialsegment",
            "Axon_Heminode", "heminode",
            "Axon_Node", "node",]
}

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
    secmap  str (default: 'swc')
        Which section mapping to use. swc is the standard swc mapping,
        sbem is an extended mapping for VCN serial blockface data.
    data : ndarray or None
        Optionally, a data array may be provided instead of an swc file. This
        is used internally.
    scales : dict or None
        dict of format: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'r': 1.0} to provide
        appropriate scaling along each of the axes.
    """

    def __init__(
        self,
        filename: Union[Path, str, None] = None,
        types: Union[str, None] = None,
        secmap: str = "swc",
        data: Union[np.ndarray, None] = None,
        scales: Union[dict, None] = None,
        verify: bool = False,
        args: object = None
    ) -> None:
        self._dtype = [
            ("id", int),
            
            ("type", int),
            ("x", float),
            ("y", float),
            ("z", float),
            ("r", float),
            ("parent", int),
        ]
        print("swctohoc: ", verify)
        self._id_lookup = None
        self._sections = None
        self._children = None
        self.scales = scales
        self.pruneaxon = False
        self.prunedendrite = False
        self.prunedistal = False
        self.topology = False
        self.verify = verify
        if args is not None:
            self.pruneaxon = args.pruneaxon
            self.prunedendrite = args.prunedendrite
            self.prunedistal = args.prunedistal
            self.topology = args.topology
        if secmap == "swc":
            self.sectypes = swc_sectypes
        elif secmap == "sbem":
            self.sectypes = sbem_sectypes
        elif secmap == "sbem2":
            self.sectypes = sbem2_sectypes
        else:
            raise ValueError("SWC number map type is not recognized: %s" % secmap)

        if types is not None:  # add-on or overwrite types to dictionary
            self.sectypes.update(types)

        if data is not None:
            self.data = data
        elif filename is not None:
            self.load(filename.with_suffix(".swc"))
            self.filename = filename
        else:
            raise TypeError("Must initialize with filename or data array.")

        self.sort()
        self.set_parent_section('soma')

    def load(self, filename: Union[Path, str, None] = None) -> None:
        assert filename is not None
        self.filename = Path(filename).with_suffix('.swc')
        print(f"Loading: {str(self.filename):s}")
        self.data = np.loadtxt(self.filename, dtype=self._dtype)
        if self.scales is not None:
            self.rescale(
                x=self.scales["x"],
                y=self.scales["y"],
                z=self.scales["z"],
                r=self.scales["r"],
            )

    def copy(self) -> object:
        return SWC(data=self.data.copy(), types=self.sectypes)

    def sort(self) -> None:
        """
        Sort the tree in topological order.
        This is the first stop
        """

        order = self.branch(self.root)
        lt = self.lookup
        indexes = np.array([lt[i] for i in order], dtype=int)
        self.data = self.data[indexes]
        self._id_lookup = None
        self._sections = None
        print('sorted')

    def branch(self, id: int) -> list:
        """
        Return a list of IDs in the branch beginning at *id*.
        """
        branch = [id]
        for ch in self.children(id):
            branch.extend(self.branch(ch))
        return branch
    
    def children(self, ident: int) -> list:
        """
        Return a list of all children of the node *id*.
        """
        if self._children is None:  # build the child dict
            self._children = {}
            for rec in self.data:
                self._children.setdefault(rec["parent"], [])
                self._children[rec["parent"]].append(rec["id"])
        # print('children: ', self._children)
        return self._children.get(ident, [])

    @property
    def lookup(self) -> dict:
        """
        Return a dict that maps *id* to *index* in the data array.
        """
        if self._id_lookup is None:
            self._id_lookup = dict([(rec["id"], i) for i, rec in enumerate(self.data)])
            # self._id_lookup = {}
            # for i, rec in enumerate(self.data):
            # self._id_lookup[rec['id']] = i
        return self._id_lookup

    def __getitem__(self, ident: int) -> int:
        """
        Return record for node *ident*.
        """
        return self.data[self.lookup[ident]]

    def set_parent_section(self, secname: str) -> None:
        soma_sec = None
        for r in self.data:
            # print('type: ', r['type'])
            if r['type'] in idsofpart ["soma"]:
                soma_sec = r.copy()
        if soma_sec is not None:
            self.reparent(ident=soma_sec['id']
)
            
    def reparent(self, ident: int) -> None:
        """
        Rearrange tree to make *ident* the new root parent.
        """
        d = self.data

        # bail out if this is already the root
        if self[ident]["parent"] == -1:
            return

        parent = -1
        while ident != -1:
            oldparent = self[ident]["parent"]
            self[ident]["parent"] = parent
            parent = ident
            ident = oldparent
        self._children = None
        self.sort()
        self.sections

    @property
    def sections(self) -> list:
        """Return lists of IDs grouped by topological section.
        The first item in each list connects to the last item in a previous
        list.
        """
        # print('self.data: ', self.data)
        if self._sections is None:
            sections = []
            sec = []

            # find all nodes with nore than 1 child
            branchpts = set()  # no one is a branch
            endpoints = set(self.data["id"])  # everyone is an endpoint
            print("endpoints: ", len(endpoints), len(self.data["id"]))
            endpoints.add(-1)
            seen = set()
            for r in self.data:
                p = r["parent"]
                if p in seen:
                    branchpts.add(p)
                else:
                    seen.add(p)
                    endpoints.remove(p)
                    if r['type'] == 10:
                        print(f"removed {r['id']:d} from endpoint: ")

            # build lists of unbranched node chains
            lasttype = self.data["type"][0]
            lastid = self.data["id"]
                
            for r in self.data:
                if self.prunedendrite and r["type"] in idsofpart['dendrite']:
                    continue
                if self.prunedistal and r["type"] in idsofpart['distal']:
                    continue
                if self.pruneaxon and r["type"] in idsofpart['axon']:
                    continue
                sec.append(r["id"])
                if (
                    r["id"] in branchpts
                    or r["id"] in endpoints
                    or r["type"] != lasttype
                ):
                    if r['type'] == 10:
                        print("id = ", r["id"], " lastid: ", lastid)
                        print("Restarting type 10, because: in endpoint: ", r["id"] in endpoints)
                        print(" or in brancpts: ", r["id"] in branchpts)
                        print(" or not same as last type: ", r["type"], " lastype = ", lasttype)
                        continue
                    sections.append(sec)
                    sec = []
                    lasttype = r["type"]
                    lastid = r["id"]

            self._sections = sections
        return self._sections

    def connect(self, parent_id: int, swc: object) -> None:
        """
        Combine this tree with another by attaching the root of *swc* as a 
        child of *parent_id*.
        """
        data = swc.data.copy()
        shift = self.data["id"].max() + 1 - data["id"].min()
        data["id"] += shift
        rootmask = data["parent"] == -1
        data["parent"] += shift
        data["parent"][rootmask] = parent_id

        self.data = np.concatenate([self.data, data])
        self._children = None
        self.sort()

    def set_type(self, typ: str) -> None:
        self.data["type"] = typ

    def make_hoc(self, verify=False) -> str:
        # if self.topology:
        #     print("Showing topology: no file will be written")
        #     return
        print("MakeHOC in swctohoc", verify)
        hoc = []
        # Add some header information
        hoc.extend([f"// Translated from SWC format by: swc_to_hoc.py"])
        hoc.append(f"// Source file: {str(self.filename):s}")
        hoc.append(f"// {datetime.datetime.now().strftime('%B %d %Y, %H:%M:%S'):s}")
        if self.scales is None:
            hoc.append(f"// No scaling")
        else:
            hoc.append(
                f"// Scaling: x: {self.scales['x']:f}, y: {self.scales['y']:f}, z: {self.scales['z']:f}, r: {self.scales['r']:f}"
            )
        hoc.append("")
        sectypes = self.sectypes.copy()
        # print('sectypes: ', secf)
        for t in np.unique(self.data["type"]):
            # print(t)
            if t not in sectypes:
                sectypes[t] = "type_%d" % t
        # create section lists
        screated = []
        for t in list(sectypes.values()):
            if t in screated:
                continue
            hoc.extend([f"objref {t:s}\n{t:s} = new SectionList()"])
            screated.append(t)
        hoc.append("")
        # create sections
        sects = self.sections

        hoc.append(f"create sections[{len(sects):d}]")
        sec_ids = {}

        for i, sec in enumerate(sects):
            # remember hoc index for this section
            endpt = self[sec[-1]]["id"]
            sec_id = len(sec_ids)
            sec_ids[endpt] = sec_id
            # print(i, sec, endpt, sec_id)
        #            print(sects)
           # add section to list
            hoc.append(f"access sections[{sec_id:d}]")
            typ = self[sec[0]]["type"]
            hoc.append(f"{sectypes[typ]:s}.append()")

           # connect section to parent
            p = self[sec[0]]["parent"]
            if p != -1:
               # print(f"p: {str(p):s}, {sec_id:d}")
               # print(self[sec[0]])
                hoc.append(
                    f"connect sections[{sec_id:d}](0), sections[{sec_ids[p]:d}](1)"
                )

           # set up geometry for this section
            hoc.append("sections[%d] {" % sec_id)
            if len(sec) == 1:
                if p != -1:  # if a parent exists, then make this connections
                    seg = sects[sec_ids[p]][
                        -1
                    ]  # get last segement in the parent section
                    rec = self[seg]
                    if rec["r"] < 0.05:
                        print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                        rec["r"] = 0.05
                    hoc.append(
                        f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})  // seg={seg:d} Singleton repair: to section[{sec_ids[p]:d}]"
                    )
            for seg in sects[sec_id]:
                rec = self[seg]
                if rec["r"] < 0.05:
                    print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                    rec["r"] = 0.05
                hoc.append(
                    f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})   // seg={seg:d}"
                )
            hoc.append("}")

            hoc.append("")
        if verify:
            print(hoc)
        return hoc
                  
        
    def write_hoc(self, filename: Union[Path, str, None] = None, verify:bool=False) -> None:
        """
        Write data to a HOC file.
        Each node type is written to a separate section list.
        """
        hoc = self.make_hoc(verify)
        # print("hoc: ", hoc)
        if filename is not None:
            with open(filename, "w") as fh:
                fh.write("\n".join(hoc))
            print(f"Wrote hoc file: {str(filename):s}")
        # now generate reverse section map for reference
            self.make_segmap(filename)
        return hoc



    @property
    def root(self) -> int:
        """
        ID of the root node of the tree.
        """
        ind = np.argwhere(self.data["parent"] == -1)[0, 0]
        return self.data[ind]["id"]


    def path(self, node) -> list:
        path = [node]
        while True:
            node = self[node]["parent"]
            if node < 0:
                return path
            path.append(node)

    def rescale(self, x: float, y: float, z: float, r: float) -> None:
        self.data["x"] *= x
        self.data["y"] *= y
        self.data["z"] *= z
        self.data["r"] *= r

    def translate(self, x: float, y: float, z: float, r: float) -> None:
        self.data["x"] += x
        self.data["y"] += y
        self.data["z"] += z

    def shorten_secname(self, sec):
        if len(sec) > 10:
            secstr = "%s,...%s" % (
                str(tuple(sec[:3]))[:-1],
                str(tuple(sec[-3:]))[1:],
            )
        else:
            secstr = str(tuple(sec))

    def show_topology(self) -> None:
        """
        Print the tree topology.
        """
        if not self.topology:
            return
        path = []
        indent = ""
        this_indent = ""
        secparents = [self[s[0]]["parent"] for s in self.sections]
        for i, sec in enumerate(self.sections):
            p = secparents[i]
            if p != -1:
                ind = path.index(p)
                path = path[: ind + 1]
                indent = indent[: (ind + 1) * 3]
            path.append(self[sec[-1]]["id"])

            # look ahead to see whether subsequent sections are children
            if p in secparents[i + 1 :]:
                this_indent = indent[:-2] + "├─ "
                indent = indent[:-2] + "│  │  "
            else:
                this_indent = indent[:-2] + "└─ "
                indent = indent[:-2] + "   │  "

            typ = self.sectypes[self[sec[0]]["type"]]
            secstr = self.shorten_secname(sec)

            print(
                "%ssections[%d] type=%s parent=%d %s" % (this_indent, i, typ, p, secstr)
            )

    def make_segmap(self, filename:Path, stronly=False) -> None:
        """
        Create a file that helps map hoc sections back to the original swc segments
        (from hoc_swc_sectionmap.py in vcnmodel)

        This requires a "hocx" file, which has the extended information about which swc
        segment is associated with each hoc pt3dadd call. 
 
        The result is a text file that looks like:
        hocsectionname : 1,3,5,7,9   
        where the numbers are the swc elements.

        Parameters
        ----------
        
        fn : str or Path
            filename of the hoc file to use for input
        """

        re_section = re.compile("\s*(sections\[)([0-9]*)\]\s*{")
        re.compile(
            "\s*(pt3dadd\()([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)"
        )
        re_section = re.compile("\s*(sections\[)([0-9]*)\]\s*{")
        re_access = re.compile("\s*(access)\s*(sections\[)([0-9]*)\]\s*")
        re_append = re.compile("\s*([a-z]*)(.append\(\))")
        re_connect = re.compile(
            "\s*(connect)\s*(sections\[)([0-9]*)\](\([0-9]*\)),\s*(sections\[)([0-9]*)\](\([0-9]*\))"
        )

        re_seg = re.compile("(seg\=)([\d]*)")  # '([d+])$')
        re_endsec = re.compile("^}")
        dout = ""
        in_section = False
        secstr = ""
        print("segmap File: ", filename)
        with open(filename, "r") as fh:
            for cnt, line in enumerate(fh):  # read the input file line by line
                line = line.rstrip().lstrip()
                s = re_section.match(line)
                if s is not None:
                    secno = s.groups()[1]
                    secstr = f"section[{secno:s}]: "
                    in_section = True
                    swcs = []
                    continue
                if in_section:
                    if re_endsec.match(line):
                        in_section = False
                        for swi in swcs:
                            secstr += f"{swi:s}, "
                        # print(secstr)
                        dout += secstr + "\n"
                        secstr = []  # reset
                        continue
                    swcindex = re_seg.search(line)
                    if swcindex is not None:
                        swci = swcindex.groups()[1]
                        swcs.append(swci)
        fout = Path(filename).with_suffix(".segmap")
        fout.write_text(dout)
        print("Wrote hoc->swc segmap to: ", fout)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWC file to HOC file for NEURON",
        argument_default=argparse.SUPPRESS,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        dest="filename",
        action="store",
        default=None,
        help="Select the file to convert (no default)",
    )
    parser.add_argument(
        "-r",
        "--radiiscale",
        dest="radiiscale",
        type=float,
        default=1.0,
        help="Set scale factor for radii",
    )
    parser.add_argument(
        "-v",
        "--verify",
        dest = "verify",
        action="store_true",
        default=False,
        help="print hoc output from swc for verification",
    )
    
    # parser.add_argument(
    #     "--somascale",
    #     type=float,
    #     default=1.0,
    #     dest="somascale",
    #     help="Set scaling for soma sections"
    # )
    #
    # parser.add_argument(
    #     "--dendscale",
    #     type=float,
    #     default=1.0,
    #     dest="dendscale",
    #     help="Set scaling for dendritic sections"
    # )
    #
    parser.add_argument(
        "-s",
        "--secmap",
        type=str,
        default="swc",
        dest="secmap",
        choices=["swc", "sbem", "sbem2"],
        help="Choose section ampping",
    )
    
    parser.add_argument(
        "-t",
        "--topology",
        action="store_true",
        dest="topology",
        default=False,
        help="Show topology (blocks output writing)",
    )
    
    parser.add_argument(
        "--prunedendrite",
        action="store_true",
        dest="prunedendrite",
        default=False,
        help="Prune all dendrite sections from the hoc output",
    )
    parser.add_argument(
        "--prunedistal",
        action="store_true",
        dest="prunedistal",
        default=False,
        help="Prune all dendrite sections beyong proximal from the hoc output",
    )
    parser.add_argument(
        "--pruneaxon",
        action="store_true",
        dest="pruneaxon",
        default=False,
        help="Prune all axon sections from the hoc output",
    )
    
    parser.add_argument(
        "-R",
        action="store_true",
        default=False,
        # dest="remap",
        help="Remap names to basic set used in cnmodel",
    )

    args = parser.parse_args()
    fn = Path(args.filename).with_suffix('.swc')
    scales = {"x": 1.0, "y": 1.0, "z": 1.0, "r": 1.0, "soma": 1.0, "dend":1.0}
    if args.radiiscale != 1.0:
        scales["r"] = args.radiiscale
    # if args.somascale != 1.0:
    #     scales["soma"] = args.somascale
    # if args.dendscale != 1.0:
    #     scales["dend"] = args.dendscale

    noseparatescale = True
    if fn.is_file():
        s = SWC(filename=fn, secmap=args.secmap, scales=scales, verify=args.verify, args=args)
        fname = args.filename
        s.show_topology()
        if noseparatescale or not (args.somascale or args.dendscale):
            s.write_hoc(Path(fname).with_suffix(".hocx"), args.verify)
        else:
            ffn = Path(
                fname.stem,
                '_s_{.3f:args.somascale}_d_{.3f:args.dendscale}').with_suffix(".hocx")
            s.write_hoc(ffn, args.verify)
    else:
        print(f'File "{str(fn):s}" was not found')


if __name__ == "__main__":
    main()
