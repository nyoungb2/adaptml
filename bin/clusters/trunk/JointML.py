#!/usr/bin/env python

#--- Option parsing ---#
"""
JointML: Cluster assignment
         
Usage:
  JointML [options] <tree> <outgroup> <color>
  JointML -h | --help
  JointML --version

Options:
  <tree>       Tree file in newick format.
  <outgroup>   Outgroup name.
  <color>      Color file.
  -H=<H>       Habitat file. [Default: habitat.matrix]
  -m=<m>       mu file. [Default: mu.val]
  -w=<w>       Write directory. [Default: .]
  -c=<x>       C-dist output file. [Default: cdist]
  -h --help    Show this screen.
  --version    Show version.

Description:
  Habitat assignment to nodes on the tree.

  `color` option:
     File specifying visualization colors for both leaves and ancestral 
     nodes on phylogenetic tree.  Leaves sharing identical ecology will 
     have the same radial bar plots; ancestral nodes sharing the same 
     ancestral assignment will also have uniform colors. To specify bar 
     plot components, identify EcologyID character position in column 1 
     (character position begins at 1) and desired character in column 2.  
     To specify habitat colors, put an 'H' in column 1 and identify the 
     habitat number in column 2.  Columns 3, 4, & 5 define R, G, & B 
     integer values from (0,255). Each column should be single-space 
     delimited.
"""


import os
import sys
import pdb
import time
import random

from docopt import docopt
from numpy.linalg import *
from numpy.core import *
from numpy.lib import *
from numpy import *

import multitree
import ML

start_time = time.time()
sys.setrecursionlimit(25000)



def load_migration(inFile):
    # migration 
    with open(inFile,'r') as inFH:
        matrix = eval(inFH.read())
    return matrix

def load_mu(inFile):
    # mu 
    with open(inFile,'r') as inFH:
        mu = inFH.read().strip()
        mu = float(mu)
    return mu

def load_colors(inFile):
    # colors
    color_hash = {}
    with open(inFile, 'r') as inFH:
        for line in inFH:
            parts = line.strip().split(' ')
            ring_number = parts[0]
            ring_state = parts[1]
            hex_vals = [hex(int(part)) for part in parts[2:]]
            hex_code = "#"
            if ring_number not in color_hash:
                color_hash[ring_number] = {}
            for hex_val in hex_vals:
                hex_color = hex_val[2:]
                if len(hex_color) < 2:
                    hex_color = "0" + hex_color
                hex_code += hex_color
            color_hash[ring_number][ring_state] = hex_code
    return color_hash

def laod_thresh(inFile):    
    # if specified, grab threshold file
    thresh_dict = {}
    with open(thresh_fn,'r') as thresh_f:
        for line in thresh_f:
            line_parts = line.strip().split()
            thresh = float(line_parts[2])
            if line_parts[0] not in thresh_dict:
                thresh_dict[line_parts[0]] = {}
            thresh_dict[line_parts[0]][line_parts[1]] = thresh
    return thresh_dict
    
        # open up files for writing
        # bars_fn = write_dir + "/bars.file"
        # bars_f =  open(bars_fn,"w")
        # bar_header = "\t"
        # obs_states = migration_matrix[migration_matrix.keys()[0]].keys()
        # for state in obs_states:
        #     bar_header += state + "\t"
        # bars_f.write(bar_header + "\n")
    
        # cluster_fn = write_dir + "/cluster.file"
        # cluster_f =  open(cluster_fn,"w")
        # prune_fn = write_dir + "/prune.file"
        # prune_f =  open(prune_fn,"w")
        # cdist_fn = write_dir + "/cdist.file"
        # cdist_f =  open(cdist_fn,"w")


#--- lik --#
#lik_fn = write_dir + "/lik.file"
#lik_f =  open(lik_fn,"w")


def build_tree(treeFile):
    # build the tree #
    sys.stderr.write('Building tree\n')
    with open(treeFile,"r") as inFH:
        tree_string = inFH.read().strip()
        tree = multitree.multitree()
        tree.build(tree_string)
    return tree


def rm_zero_branches(tree, outgroup):
    # root it and remove zero branch lengths
    min_len = min([b.length for b in tree.branch_list if b.length > 0.0])
    for b in tree.branch_list:
        if b.length <= 0.0:
            b.length = min_len
        names_1 = b.ends[0].name_dict[b.ends[1]]
        names_2 = b.ends[1].name_dict[b.ends[0]]
        if names_1 == outgroup or names_2 == outgroup:
            sys.stderr.write('Rooting tree\n')
            tree.rootify(b)


def write_labels(migration_matrix, color_hash, full_f):
    # write the labels
    habitats = migration_matrix.keys()
    label_line = "LABELS"
    rings = color_hash.keys()
    rings.sort()
    for ring in rings:
        symbols = color_hash[ring].keys()
        symbols.sort()
        for symbol in symbols:
            label_line += "," + symbol
    color_line = "COLORS"
    for ring in rings:
        symbols = color_hash[ring].keys()
        symbols.sort()
        for symbol in symbols:
            color_line += "," + color_hash[ring][symbol]

    full_f.write(label_line + "\n")
    full_f.write(color_line + "\n")
    

#if thresh_fn is not None:
#    prune_f.write(label_line + "\n")
#    prune_f.write(color_line + "\n")
#    cluster_f.write(label_line + "\n")
#    cluster_f.write(color_line + "\n")


def add_var_to_tree(tree, migration_matrix, mu, color_hash):
    # embed important variables into tree
    tree.migration_matrix = migration_matrix
    tree.mu = mu
    tree.color_hash = color_hash
    tree.states = None   # obs_states
    tree.thresh_dict = None  #thresh_dict


def learn_habitats(tree, mu, migration, outgroup):
    sys.stderr.write('Learn habitat assignments\n')

    # wipe the likelihoods off of the tree (just to be sure)
    ML.TreeWipe(tree.a_node)
    # learn the likelihoods
    ML.LearnLiks(tree,mu,migration,outgroup)


def estimate_states(tree, outgroup, migration_matrix):
    # estimate the states (by making each node trifurcating...)
    root = tree.root
    kids = root.GetKids()
    if kids[0].name in outgroup:
        true_root = kids[1]
    else:
        true_root = kids[0]
    lik_score = ML.EstimateStates(true_root)
    k = 1 + len(migration_matrix)*(len(migration_matrix.values()[0])-1)
    aic_score = 2*k - 2*lik_score

    return true_root, lik_score, aic_score
#    lik_f.write("likelihood: " + str(lik_score) + "\n")
#    lik_f.write("aic: " + str(aic_score) + "\n")
    

def write_habitat_by_leaf(tree, true_root, migration_f):
    # write out the habitat assignment of each leaf
    for leaf in true_root.leaf_nodes:
        this_str = str(leaf) + "\t" + leaf.habitat
        migration_f.write(this_str + "\n")


def draw_tree(tree, true_root, full_f, write_dir):
    # draw out the tree
    files = {}
    files['full'] = full_f

    #if thresh_fn is not None:
    #    files['cluster'] = cluster_f
    #    files['bar'] = bars_f
    #    files['prune'] = prune_f
    #    files['cdist'] = cdist_f

    # figure out how far each leaf is from the outgroup:
    min_dist = 0.001
    limit_dist = 0.42
    leaf_dist_fn = os.path.join(write_dir, 'leaves.dist')
    leaf_dist_f =  open(leaf_dist_fn,"w")
    for leaf in tree.leaf_node_list:
        this_dist = leaf.DistTo(true_root,None,0)[1]
        leaf_dist_f.write(str(this_dist) + "\n")
        # truncate branches that are too long:
        if this_dist > limit_dist:
            leaf.TruncateDist(true_root,this_dist,min_dist,limit_dist)
    leaf_dist_f.close()

    # write out the tree (cheating a little to make the root branches shorter)
    for branch in tree.root.child_branches:
        branch.length = min_dist
    
    itol_fn = os.path.join(write_dir, 'itol.tree')
    with open(itol_fn,"w") as outFH:
        outFH.write(true_root.treePrint("") + ";")

    sys.stderr.write('Writting out results\n')
    true_root.FulliTol(files)



def write_full(tree, true_root, files, write_dir):    
    
    strain_fn = os.path.join(write_dir, 'strain.names')
    strain_f = open(strain_fn,"w")

    # find the divergence points
    divergers = true_root.GetDivergencePoints()
    true_root.divergers = divergers

    params = {'cdist' : False}
    #true_root.ClusterTest(files,params)
    true_root.DrawSubclusters(None,cluster_f)
    tree.DrawLeaves(files)

    # print lists of constituents in each cluster
    cluster_roots = filter(lambda a: a.cluster_root,tree.node_dict.values())
    for cluster_root in cluster_roots:
        strains = cluster_root.leaf_nodes
        strain_f.write(str(len(strains)) + "\t" + str(cluster_root) +"\n")
        for strain in strains:
            strain_f.write("\t" + str(strain) + "\n")
        strain_f.write("\n")

    # retain significant clusters
    all_leaves = true_root.leaf_nodes
    for leaf in all_leaves:
        if not leaf.in_cluster:
            leaf.RemoveLeaf()
    true_root.UpPrune(files)

    # print pruned topology
    prune_topo_fn = write_dir + "/prune.tree"
    prune_topo_f =  open(prune_topo_fn,"w")
    prune_topo_f.write(true_root.treePrint("") + ";")
    prune_topo_f.close()
    
    full_f.close()
    migration_f.close()
    if thresh_fn is not None:
        cluster_f.close()
        lik_f.close()
        prune_f.close()
        cdist_f.close()
        bars_f.close()
        strain_f.close()
    

def jointML(uargs):
    # loading files
    migration = load_migration(uargs['-H'])
    mu = load_mu(uargs['-m'])
    colors = load_colors(uargs['<color>'])
    
    # build tree
    outgroup = [uargs['<outgroup>']]
    tree = build_tree(uargs['<tree>'])
    rm_zero_branches(tree, outgroup)

    # open output
    write_dir = uargs['-w'] 
    full_fn = os.path.join(write_dir, 'full.file')
    full_f =  open(full_fn,"w")
    migration_fn = os.path.join(write_dir, 'habitat.file')
    migration_f =  open(migration_fn,"w")

    # labels
    write_labels(migration, colors, full_f)
    
    # adding params to tree object
    add_var_to_tree(tree, migration, mu, colors)
    
    # learning habitats
    learn_habitats(tree, mu, migration, outgroup)

    # estimate states
    true_root, lik_score, aic_score = estimate_states(tree, outgroup, migration)

    # write out habitat by leaf
    write_habitat_by_leaf(tree, true_root, migration_f)

    # draw tree
    draw_tree(tree, true_root, full_f, write_dir)

    # writing full iTOL table
    #write_full(tree, true_root, files, write_dir)
    
    # closing
    full_f.close()
    migration_f.close()


if __name__ == '__main__':
    # user-defined args
    uargs = docopt(__doc__, version='0.1', options_first=True)    
    uargs['-w'] = os.path.abspath(uargs['-w'])

    # jointML 
    jointML(uargs)
