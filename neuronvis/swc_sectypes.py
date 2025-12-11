"""
sec_types.py : define section types for various mappings. 

"""
import numpy as np

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
    18: "primarydendrite",
    19: "preclaw",
    20: "dendriticclaw",
}

# grc_swc_sectypes = {
#     # newa swc mapping
#     0: "Undefined",
#     1: "soma",
#     10: "Myelinated_Axon",
#     3: "Basal_Dendrite",
#     15: "Apical_Dendrite",
#     5: "Custom",
#     6: "Unspecified_Neurites",
#     7: "Glia_Processes",
#     # 8: "Blank",
#     # 9: "Blank",
#     8: "Axon_Hillock",
#     11: "Unmyelinated_Axon",
#     # 12: "Dendritic_Hub",
#     12: "Proximal_Dendrite",
#     13: "Distal_Dendrite",
#     14: "Dendrite_Claw",
#     9: "Axon_Initial_Segment",
#     16: "Axon_Heminode",
#     17: "Axon_Node",
#     18: "Dendritic_Swelling",
# }


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
# This table is for swcs from Syglass from May 2021 (who changed it?)
sbem2_sectypes = {
    # new swc mapping
    0: "Undefined",
    1: "Soma",
    2: "Myelinated_Axon",
    3: "Basal_Dendrite",
    4: "Apical_Dendrite",
    5: "Custom",
    6: "Unspecified_Neurites",
    7: "Glia_Processes",
    8: "Dendritic_Claw",
    9: "Blank",
    10: "Axon_Hillock",
    11: "Dendritic_Swelling",
    12: "Dendritic_Hub",
    13: "Proximal_Dendrite",
    14: "Distal_Dendrite",
    15: "Axon_Initial_Segment",
    16: "Axon_Heminode",
    17: "Axon_Node",
}

# List of Nodes (Spirou, 2024)	

sbem3_sectypes = {
0	: "Undefined",
1	: "Soma",
2	: "Soma_spine",
3	: "Soma_undefined_1",
4	: "Soma_undefined_2",
5	: "Soma_undefined_3",
6	: "Axon_hillock",
7	: "Axon_initial_segment",
8	: "Axon_myelinated_segment",
9	: "Axon_node",
10	: "Axon_heminode",
11	: "Axon_spine",
12	: "Axon_undefined_1",
13	: "Axon_undefined_2",
14	: "Axon_undefined_3",
15	: "Terminal",
16	: "Terminal_stalk",
17	: "Terminal_swelling",
18	: "Terminal_neck",
19	: "Terminal_branch",
20	: "Terminal_undefined_1",
21	: "Terminal_undefined_2",
22	: "Terminal_undefined_3",
23	: "Dendrite_basal",
24	: "Dendrite_apical",
25	: "Dendrite_proximal",
26	: "Dendrite_distal",
27	: "Dendrite_spine",
28	: "Dendrite_hub",
29	: "Dendrite_swelling",
30	: "Dendrite_preclaw",
31	: "Dendrite_claw",
32	: "Dendrite_undefined_1",
33	: "Dendrite_undefined_2",
34	: "Dendrite_undefined_3",
35	: "Neurite",
36  : "Cilium",
37	: "Astrocyte_soma",
38	: "Astrocyte_primary_branch",
39	: "Astrocyte_distal_process",
40	: "Astrocyte_vellus_process",
41	: "Astrocyte_blood_brain_barrier",
42	: "Astrocyte_undefined",
}



# renaming of cell parts to match cnmodel data tables (temporary)
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

# Note: Applies only wnen SWC, SBEM or SBEM2 in use.
idsofpart_swc = {
    "dendrite": [3, 4, 12],
    "axon:": [2, 10, 11],
    "soma": [1],
    "distal": [],
}
idsofpart_sbem = {
    "dendrite": [3, 4, 12, 13, 14, 18],
    "distal": [12, 14, 18],
    "axon": [2, 10, 11, 15, 16, 17],
    "soma": [1],
}
idsofpart_sbem2 = {  # for sbem2 map (what a pain! )
    "dendrite": [3, 4, 12, 13, 14, 11, 18],
    "distal": [12, 14, 18],
    "axon": [2, 10, 15, 16, 17],
    "soma": [1],
}

# idsofpart_sbem3 = {  # for sbem3 map (what a pain! )
#     "dendrite": [int(a) for a in range(23, 35)],
#     "distal": [12, 14, 18],
#     "axon": [int(a) for a in range(6, 15)],
#     "soma": [int(a) for a in range(1, 6)],
# }

idsofpart_sbem3 = {  # for sbem2 map (what a pain! )
    "dendrite": [int(x) for x in np.arange(23, 35)],
    "distal": [26],
    "axon": [int(x) for x in np.arange(6, 15)],
    "terminal": [int(x) for x in np.arange(15, 23)],
    "soma": [int(x) for x in np.arange(1,6)],
    "astrocyte": [int(x) for x in np.arange(36, 41)],
}

def get_partsof(sbem3_sectypes):
    parts = {}
    for k, v in sbem3_sectypes.items():
        if v in parts:
            parts[v].append(k)
        else:
            parts[v] = [k]
    return parts

partsof = get_partsof(sbem3_sectypes)
# partsof = {
#     "dendrite": [
#         "dendrite",
#         "basal_dendrite" "Basal_Dendrite",
#         "Apical_Dendrite",
#         "apical_dendrite",
#         "proximal_dendrite",
#         "Proximal_Dendrite" "distal_dendrite",
#         "Distal_Dendrite",
#         "Dendritic_Swelling",
#         "hub",
#         "Dendritic_Hub",
#         "primarydendrite",
#         "preclaw",
#         "Dendritie_Claw",
#     ],
#     "axon": [
#         "Axon_Hillock",
#         "hillock",
#         "Unmyelinated_Axon",
#         "unmyelinatedaxon",
#         "Axon_Initial_Segment",
#         "initialsegment",
#         "Axon_Heminode",
#         "heminode",
#         "Axon_Node",
#         "node",
#     ],
# }


# convienence dictionary
sectypes = {"swc": swc_sectypes, # original definitions
            "sbem": sbem_sectypes, # bushy cell definitions
            "sbem2": sbem2_sectypes, # bushy cell definitions
            "sbem3": sbem3_sectypes, # second block, small-cell cap definitions
            "grc": sbem_sectypes, # granule cell definitions
}
