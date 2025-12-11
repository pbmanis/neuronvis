# -*- coding: utf8 -*-
import numpy as np
from pathlib import Path
from typing import Union
import datetime
import argparse
import re

import neuronvis.swc_sectypes as swc_sectypes
import neuronvis.renderer_colormaps as colormaps
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

Note: Multiple sectypes defined in sec_types.py now - there are several 
mappings. 


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
    section_map  str (default: 'swc')
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
        section_map: str = "swc",
        data: Union[np.ndarray, None] = None,
        scalexyzr: Union[dict, None] = None,
        center: bool = False,
        verify: bool = False,
        args: object = None,
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
        self._id_lookup = None
        self._sections = None
        self._children = None
        self.scalexyzr = scalexyzr
        self.pruneaxon = False
        self.prunedendrite = False
        self.prunedistal = False
        self.topology = False
        self.center = center
        self.verify = verify
        self.centerpos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        if args is not None:
            self.center = args.center
            self.pruneaxon = args.pruneaxon
            self.prunedendrite = args.prunedendrite
            self.prunedistal = args.prunedistal
            self.topology = args.topology
        match section_map:
            case "swc":
                self.sectypes = swc_sectypes.swc_sectypes
                self.idsofpart = swc_sectypes.idsofpart_swc
            case "sbem":
                self.sectypes = swc_sectypes.sbem_sectypes
                self.idsofpart = swc_sectypes.idsofpart_sbem
            case "sbem2":
                self.sectypes = swc_sectypes.sbem2_sectypes
                self.idsofpart = swc_sectypes.idsofpart_sbem2
            case "sbem3":
                self.sectypes = swc_sectypes.sbem3_sectypes
                self.idsofpart = swc_sectypes.idsofpart_sbem3
            case _:
                raise ValueError("SWC number map type is not recognized: %s" % section_map)

        if types is not None:  # add-on or overwrite types to dictionary
            self.sectypes.update(types)

        if data is not None:
            self.data = data
        elif filename is not None:
            self.load(filename.with_suffix(".swc"))
            print("load ok")
            self.filename = filename
        else:
            raise TypeError("Must initialize with filename or data array.")

        print("sorting: ")
        self.sort()
        self.set_parent_section("soma")
        print("sorted: ")
        
    def load(self, filename: Union[Path, str, None] = None) -> None:
        assert filename is not None
        self.filename = Path(filename).with_suffix(".swc")
        if not self.filename.is_file():
            raise FileNotFoundError(f"swc_to_hoc:: SWC file {str(self.filename):s} not found.")
        print(f"swc_to_hoc:: Loading: {str(self.filename):s}")
        self.data = np.loadtxt(self.filename, dtype=self._dtype)

        if self.scalexyzr is not None:
            print("swc_to_hoc: Rescaling swc with: ", self.scalexyzr)
            # hack: for radius if the scale is not set, use the projection of the xyz scales
            rscale = self.scalexyzr.get("r", np.linalg.norm([self.scalexyzr["x"], self.scalexyzr["y"], self.scalexyzr["z"]]))
            self.rescale(
                x=self.scalexyzr["x"],
                y=self.scalexyzr["y"],
                z=self.scalexyzr["z"],
                r=rscale, # self.scalexyzr["r"],
            )
            print("swc_to_hoc: Rescaled swc")
        if self.center:
            print("swc_to_hoc: centering")
            # save the original position
            self.centerpos = {'x': self.data["x"][0],
                              'y': self.data["y"][0],
                              'z': self.data["z"][0]}
            print("swc_to_hoc:: Centering swc on first section in list")
            self.translate(
                x=-self.data["x"][0], # .min(),
                y=-self.data["y"][0], # .min(),
                z=-self.data["z"][0], # .min(),
                r=0.0,
            )

            print("swc_to_hoc:: Center position: ", self.centerpos)

    def copy(self) -> object:
        return SWC(data=self.data.copy(), types=self.sectypes)

    def sort(self) -> None:
        """
        Sort the tree in topological order.
        This is the first step
        """

        order = self.branch(self.root)
        lt = self.lookup
        indexes = np.array([lt[i] for i in order], dtype=int)
        self.data = self.data[indexes]
        self._id_lookup = None
        self._sections = None
 
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
            if r["type"] in self.idsofpart["soma"]:
                soma_sec = r.copy()
        if soma_sec is not None:
            self.reparent(ident=soma_sec["id"])

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
            # print("endpoints: ", len(endpoints), len(self.data["id"]))
            endpoints.add(-1)
            seen = set()
            for r in self.data:
                p = r["parent"]
                if p in seen:
                    branchpts.add(p)
                else:
                    seen.add(p)
                    endpoints.remove(p)
                    if r["type"] == 10:
                        print(f"swc_to_hoc:: removed {r['id']:d} from endpoint: ")

            # build lists of unbranched node chains
            lasttype = self.data["type"][0]
            lastid = self.data["id"]

            for r in self.data:
                if self.prunedendrite and r["type"] in self.idsofpart["dendrite"]:
                    continue
                if self.prunedistal and r["type"] in self.idsofpart["distal"]:
                    continue
                if self.pruneaxon and r["type"] in self.idsofpart["axon"]:
                    continue
                sec.append(r["id"])
                if r["id"] in branchpts or r["id"] in endpoints or r["type"] != lasttype:
                    if r["type"] == 10:
                        print("swc_to_hoc:: Got a hillock: ")
                        print("    id = ", r["id"], " lastid: ", lastid)
                        print("    Restarting type 10, because: in endpoint: ", r["id"] in endpoints)
                        print("        or in brancpts: ", r["id"] in branchpts)
                        print("        or not same as last type: ", r["type"], " lastype = ", lasttype)
                        raise ValueError("swc_to_hoc:: Hillock in section, r type = 10")
                        # continue
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
        hoc = []
        # Add some header information
        hoc.extend([f"// Translated from SWC format by: swc_to_hoc.py"])
        hoc.append(f"// Source file: {str(self.filename):s}")
        hoc.append(f"// {datetime.datetime.now().strftime('%B %d %Y, %H:%M:%S'):s}")
        if self.scalexyzr is None:
            hoc.append(f"// No scaling")
        else:
            hoc.append(
                f"// Scaling: x: {self.scalexyzr['x']:f}, y: {self.scalexyzr['y']:f}, z: {self.scalexyzr['z']:f}, r: {self.scalexyzr['r']:f}"
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
                # print(sec_id, sec_ids, p)
                if p in sec_ids:
                    hoc.append(f"connect sections[{sec_id:d}](0), sections[{sec_ids[p]:d}](1)")

            # set up geometry for this section
            hoc.append("sections[%d] {" % sec_id)
            if len(sec) == 1:
                if p != -1 and p in sec_ids:  # if a parent exists, then make this connections
                    seg = sects[sec_ids[p]][-1]  # get last segement in the parent section
                    rec = self[seg]
                    if rec["r"] < 0.05:
                        # print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                        rec["r"] = 0.05
                    hoc.append(
                        f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})  // seg={seg:d} Singleton repair: to section[{sec_ids[p]:d}]"
                    )
            for seg in sects[sec_id]:
                rec = self[seg]
                if rec["r"] < 0.05:
                    # print(f"MIN DIA ENCOUNTERED: {seg:d}, {rec['r']:f}")
                    rec["r"] = 0.05
                hoc.append(
                    f"  pt3dadd({rec['x']:f}, {rec['y']:f}, {rec['z']:f}, {rec['r']*2:f})   // seg={seg:d}"
                )
            hoc.append("}")

            hoc.append("")
        if verify:
            print(hoc)
        return hoc

    def write_hoc(self, filename: Union[Path, str, None] = None, verify: bool = False) -> None:
        """
        Write data to a HOC file.
        Each node type is written to a separate section list.
        """
        hoc = self.make_hoc(verify)
        # print("hoc: ", hoc)
        if filename is not None:
            with open(filename, "w") as fh:
                fh.write("\n".join(hoc))
            print(f"swc_to_hoc:: Wrote hoc file: {str(filename):s}")
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
        return secstr

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

            print("swc_to_hoc:: %ssections[%d] type=%s parent=%d %s" % (this_indent, i, typ, p, secstr))

    def make_segmap(self, filename: Path, stronly=False) -> None:
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

        re_section = re.compile(r"\s*(sections\[)([0-9]*)\]\s*{")
        re.compile(
            r"\s*(pt3dadd\()([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)\,\s([-+]?[0-9]*\.?[0-9]+)"
        )
        re_section = re.compile(r"\s*(sections\[)([0-9]*)\]\s*{")
        re_access = re.compile(r"\s*(access)\s*(sections\[)([0-9]*)\]\s*")
        re_append = re.compile(r"\s*([a-z]*)(.append\(\))")
        re_connect = re.compile(
            r"\s*(connect)\s*(sections\[)([0-9]*)\](\([0-9]*\)),\s*(sections\[)([0-9]*)\](\([0-9]*\))"
        )

        re_seg = re.compile(r"(seg\=)([\d]*)")  # '([d+])$')
        re_endsec = re.compile(r"^}")
        dout = ""
        in_section = False
        secstr = ""
        print("swc_to_hoc:: Generating segmap File: ", filename)
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
        print(f"    Wrote hoc->swc segmap to: {fout!s}")


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
        dest="verify",
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
        default="sbem3",
        dest="section_map",
        choices=["swc", "sbem", "sbem2", "sbem3"],
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
        "--center",
        "-c",
        dest="center",
        action="store_true",
        default=False,
        help="Force first point to be (0,0,0) (default: False)",
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
    fn = Path(args.filename).with_suffix(".swc")
    scales = {"x": 1.0, "y": 1.0, "z": 1.0, "r": 1.0, "soma": 1.0, "dend": 1.0}
    if args.radiiscale != 1.0:
        scales["r"] = args.radiiscale
    # if args.somascale != 1.0:
    #     scales["soma"] = args.somascale
    # if args.dendscale != 1.0:
    #     scales["dend"] = args.dendscale

    noseparatescale = True
    if fn.is_file():
        s = SWC(
            filename=fn,
            section_map=args.section_map,
            scalexyzr=scales,
            center=args.center,
            verify=args.verify,
            args=args,
        )
        fname = args.filename
        s.show_topology()
        if noseparatescale or not (args.somascale or args.dendscale):
            s.write_hoc(Path(fname).with_suffix(".hocx"), args.verify)
        else:
            ffn = Path(fname.stem, "_s_{.3f:args.somascale}_d_{.3f:args.dendscale}").with_suffix(
                ".hocx"
            )
            s.write_hoc(ffn, args.verify)
    else:
        print(f'File "{str(fn):s}" was not found')


if __name__ == "__main__":
    main()
