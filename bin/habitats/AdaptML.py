#!/usr/bin/env python

#--- Option parsing ---#
"""
AdaptML: Automatically partition a gene phylogeny by using 
         genetic and ecological similarity

Usage:
  AdaptML [options] <tree> <outgroup>
  AdaptML -h | --help
  AdaptML --version

Options:
  <tree>       Tree file in newick format.
  <outgroup>   Outgroup name.
  -i=<i>       Initial habitat number. [Default: 16]
  -c=<c>       Collapse threshold. [Default: 0.1]
  -v=<v>       Converge threshold. [Default: 0.001]
  -r=<r>       Method for inferring mu. [Default: avg]
  -m=<m>       mu parameters. [Default: 1.00000000001]
  -w=<w>       Write directory. [Default: .]
  -h --help    Show this screen.
  --version    Show version.

Description:
  First step in the AdaptML analysis pipeline. 

  `tree` file:
     An input phylogenetic tree that incorporates ecological data into 
     sequence filenames. Due to Newick file idiosyncrasies it is recommended 
     that PhyML be used to generate input trees. Gene sequences should be 
     named according to the following format: "EcologyID_SequenceID",
     where "EcologyID" is a string shared in common by all sequences with 
     identical ecology and SequenceID is a unique identifier such that 
     no two sequences with the same ecology information share the same 
     SequenceID.

  `i` option:
     The number of random habitats AdaptML will initialize with.
     If the ultimate number of inferred habitats is equal to this initial 
     number, try re-running AdaptML with more initial habitats.

  `m` option:
     The mu parameter (the average habitat transition rate).

  `r` option:
     Method for inferring mu.
     Default is "avg", which is a fast approximative method.
     A more precise, but also more time-consuming option is "num", 
     which uses SciPy's numerical optimization toolbox 

  `c` option:
     Threshold value for collapsing redundant habitats. 
     The value should range between (0,1). Higher values will 
     lead to fewer habitats being inferred.

  `v` option:
     Threshold value for declaring habitat distributions to have converged.
      Value should range from (0,1).
"""

# import
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

sys.setrecursionlimit(25000)



def build_tree(treeFile):
    """Build tree
    """
    sys.stderr.write('Building Tree\n')
    tree_file = open(treeFile,'r')
    tree_string = tree_file.read().strip()
    tree = multitree.multitree()
    tree.build(tree_string)
    if tree.root != None:
        msg = 'Input tree is rooted. You must use unrooted tree'
        sys.stderr.write(msg + '\n')
        sys.exit()
    tree_file.close()
    return tree


def rm_zero_branches(tree):
    """Remove zero-length branches from tree
    """
    min_len = min([b.length for b in tree.branch_list if b.length > 0.0])
    for b in tree.branch_list:
        if b.length <= 0.0:
            b.length = min_len
    return tree


def create_habitat_list(tree, hab_num=16):
    """How many species are there?
    """
    hab_num = int(hab_num)
    species_dict = tree.species_count
    total_leaves = sum(species_dict.values())
    habitat_list = []
    filter_list = species_dict.keys()
    for i in range(hab_num):
        habitat_list.append("habitat " + str(i))
    return habitat_list, filter_list


def create_habitat_mtx(habitat_list, filter_list):
    """Init habitat matrix
    """
    sys.stderr.write('Instantiating Habitat Matrix\n')
    habitat_matrix = {}
    for habitat in habitat_list:
        habitat_matrix[habitat] = {}
        for filt in filter_list:
            habitat_matrix[habitat][filt] = random.random()

        # normalize
        scale = sum(habitat_matrix[habitat].values())
        for filt in filter_list:
            habitat_matrix[habitat][filt] /= scale    
    return habitat_matrix


def remove_similar_habitats(habitat_matrix, habitat_thresh):
    """Remove similar groups
    """
    print "Removing Redundant Habitats"
    new_habitats = {}
    for habitat_1 in habitat_matrix:
        old_habitat = habitat_matrix[habitat_1]
        add_habitat = True
        for habitat_2 in new_habitats:
            new_habitat = new_habitats[habitat_2]
            score = 0
            for this_filter in old_habitat:
                diff = old_habitat[this_filter] - new_habitat[this_filter]
                score += math.pow(diff,2)
            if score < habitat_thresh:
                add_habitat = False
        if add_habitat:
            new_habitats[habitat_1] = habitat_matrix[habitat_1]
    return new_habitats


def learn_habitats(tree, habitat_matrix, mu, rateopt, 
                   converge_thresh, habitat_thresh):
    """Learn habitats
    """
    sys.stderr.write('Learning Habitats:\n')

    # setting/checking params
    score = -9999.99999999
    diff = 1.0
    old_diff = 1.0
    msg = 'The convergence threshold ({}) must be between 0 & 1'
    assert 0 <= converge_thresh <= 1, msg.format(converge_thresh)
    msg = 'The collapse threshold ({}) must be between 0 & 1'
    assert 0 <= habitat_thresh <= 1, msg.format(habitat_thresh)

    stats_header = ['counter', 'habs', 'ML_score', 'mu', 'habitat dist diff']
    stats = [stats_header]
    while 1:
        msg = "\t{} habitats\tRefinement Steps [d(Habitat Score)]:\n"
        sys.stderr.write(msg.format(len(habitat_matrix)))
        counter = 0
        
        stats_line = '{} habitats'.format(len(habitat_matrix))
        stats.append([stats_line] + stats_header)
            
        while 1:
            msg = '\t\t{}\t{}\n'
            sys.stderr.write(msg.format(counter, diff))
            stats.append([counter, len(habitat_matrix), score, mu, diff])
    
            # wipe the likelihoods off of the tree
            ML.TreeWipe(tree)
    
            # learn the likelihoods
            ML.LearnLiks(tree,mu,habitat_matrix)
    
            # estimate the states (by making each node trifurcating...)
            ML.EstimateStates(tree.a_node,habitat_matrix)
            
            # upgrade guesses for mu and habitat matrix
            this_migrate = habitat_matrix
            mu, habitat_matrix = ML.LearnRates(tree,mu,habitat_matrix,rateopt)
            new_migrate = habitat_matrix
    
            # stop?
            old_diff = diff
            score, diff = ML.CheckConverge(tree,new_migrate,this_migrate)
    
            if diff < converge_thresh:
                break
    
            # this should break the loop if you end up bouncing back and
            # forth between the same values
            sig_figs = 8
            diff1 = math.floor(diff*math.pow(10,sig_figs))
            diff2 = math.floor(old_diff*math.pow(10,sig_figs))        
            if diff1 > 0:
                if diff1 == diff2:
                    break
            if counter > 500:
                break
            counter += 1


        # remove similar habitats
        new_habitats = remove_similar_habitats(habitat_matrix, habitat_thresh)
        if len(new_habitats) == len(habitat_matrix):
            break
        habitat_matrix = new_habitats
        if len(habitat_matrix) < 2:
            break

    msg = 'Learned {} habitats in {} seconds\n'
    sys.stderr.write(msg.format(len(habitat_matrix), time.clock()))

    return habitat_matrix, mu, stats

    
def write_results(habitat_matrix, mu, stats, write_dir):
    """Write out the results
    """
    ## mu value
    outFile = os.path.join(write_dir, 'mu.val')
    with open(outFile,'w') as outFH:
        outFH.write(str(mu))
    sys.stderr.write('File written: {}\n'.format(outFile))

    ## habitat matrix
    outFile = os.path.join(write_dir, 'habitat.matrix')
    with open(outFile,'w') as outFH:
        outFH.write(str(habitat_matrix))
    sys.stderr.write('File written: {}\n'.format(outFile))
    
    ## stats file
    outFile = os.path.join(write_dir, 'stats.txt')
    with open(outFile,'w') as outFH:
        for x in stats:
            x = [str(y) for y in x]
            outFH.write('\t'.join(x) + '\n')
    sys.stderr.write('File written: {}\n'.format(outFile))


def build_init_rate_matrix(tree, uargs):
    """Build an initial rate matrix 
    """
    # create habitat list
    habitat_list, filter_list = create_habitat_list(tree, uargs['-i'])

    # create O(n^3) habitat matrix
    habitat_matrix = create_habitat_mtx(habitat_list, filter_list)

    # learn habitat matrix
    habitat_matrix, mu, stats = learn_habitats(tree, habitat_matrix, 
                                               mu=uargs['-m'], 
                                               rateopt=uargs['-r'], 
                                               habitat_thresh=uargs['-c'],
                                               converge_thresh=uargs['-v'])
    
    # writing out the results
    write_results(habitat_matrix, mu, stats, write_dir=uargs['-w'])


if __name__ == '__main__':
    # user-defined args
    uargs = docopt(__doc__, version='0.1', options_first=True)
    uargs['-w'] = os.path.abspath(uargs['-w'])
    uargs['-i'] = int(uargs['-i'])
    for k in ['-m', '-c', '-v']:
        uargs[k] = float(uargs[k])
    
    # building tree
    tree = build_tree(uargs['<tree>'])
    ## rm zero branch length
    tree = rm_zero_branches(tree)

    # build initial rate matrix
    build_init_rate_matrix(tree, uargs)
