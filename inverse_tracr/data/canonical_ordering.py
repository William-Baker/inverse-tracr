# A generated program will look like this:
#   PROGRAM_START
#   Map LAM_ADD tokens NA v1
#   Map LAM_MUL v1 NA v2
#   Map LAM_MUL indices NA v3
#   Select v1 v2 PRED_NEQ v4
#   Aggregate v4 v3 NA v5
#   Map LAM_ADD indices NA v6
#   SequenceMap LAM_OR v6 v5 v7
#   PROGRAM_END
#   <PAD>
#   <PAD>


# A program forms a tree like this each node is a line in the program:
#   (Program Above)      (Random pretty example)
#       t      i     
#      |      | \             t            
#      1      3  6          / | \          
#     / \     |  |         /  |  \         
#    |   2   |  |         2   |   3        
#    \  /   |  |           \ /   /        
#     4    |  |             5   6         
#      \  /  |               \ /        
#        5  |                 7        
#         \|                     
#          7                   

# The canonical ordering first orders sets of attributes according to thier depth in the tree
#  where depth is max(parent_depths)+1
# 
# Yeilding an ordered list of sets
#  [{t, i}, {1, 3, 6}, {2}, {4}, {5}, {7}]
#  [{t}, {2, 3}, {5, 6}, {7}]
# 
# Each set is then ordered according to defined orderings bellow, where input tokens preceed
#  rasp_ops and rasp_ops preceed uni_lambdas and then sequence lambdas
#  the order within these categories is defined as the order used in the original definitions
#  from dataset.py

#%%

from typing import Sequence
from inverse_tracr.data.rasp_operators import RASP_OPS, UNI_LAMBDAS, SEQUENCE_LAMBDAS, NAMED_PREDICATES
import numpy as np

# In decending order of predence here is our operator ordering
INPUT_TOKENS = ["tokens", "indices"] # List of the input tokens
RASP_OPS = list(RASP_OPS.cls_name)
NAMED_PREDICATES = list(NAMED_PREDICATES.values())
UNI_LAMBDAS = [lam_name[-1] for lam_name in UNI_LAMBDAS]
SEQUENCE_LAMBDAS = [lam_name[-1] for lam_name in SEQUENCE_LAMBDAS]

# develop scoring that can be applied to each token in a line of a program, giving a score for that unique line vs others
scores = np.arange(len(NAMED_PREDICATES))
token_weight = dict(zip(NAMED_PREDICATES, scores))
for names in [SEQUENCE_LAMBDAS, UNI_LAMBDAS, RASP_OPS, INPUT_TOKENS]:
    scores = np.arange(len(names))
    token_weight = token_weight | dict(zip(names, 4 * max(token_weight.values()) + scores))

#%%




class ComputationalTree:
    class Node:
        def __init__(self, name: str, parents = {}, children = {}, meta = None) -> None:
            self.name = name
            self.parents = parents
            self.children = children
            self.depth = None
            self.meta = meta
        def __repr__(self) -> str:
            return f"{self.name}, parents: {self.parents}, children: {self.children}, depth: {self.depth}"
    def __init__(self) -> None:
        self.nodes = dict()
    
    def add_node(self, name: str, parent_names: Sequence, meta=None):
        parents = []
        for parent_name in parent_names:
            if parent_name not in self.nodes:
                self.nodes[parent_name] = ComputationalTree.Node(parent_name)
            parents.append(self.nodes[parent_name])

        if name in self.nodes:
            node =  self.nodes[name]
            node.parents = set(parents)
            assert node.name not in parent_names
        else:
            self.nodes[name] = ComputationalTree.Node(name, parents=set(parents))

        self.nodes[name].meta = meta

        

    def complete_graph(self):
        for name, node in self.nodes.items():
            for child in node.children:
                child.parent.add(node)
            for parent in node.parents:
                parent.children.add(node)
    
    def calc_depth(self, node) -> float:
        if node.depth is  None:
            depths = [-1]
            for parent in node.parents:
                parent_depth = self.calc_depth(parent)
                depths.append(parent_depth)
            node.depth = max(depths) + 1
        return node.depth

    def calc_all_depths(self):
        for node in self.nodes.values():
            self.calc_depth(node)


def __build_tree__(prog):
    """
    returns a computational tree and a program mask
    """
    tree = ComputationalTree()
    program_mask = []
    for line in prog:
        values = list(line.values())
        if values[0] in RASP_OPS:
            program_mask.append(1)
            args = values[1:4]
            args = list(filter(lambda x: (not x.startswith('LAM')) and (not x.startswith('PRED')) and x!='NA', args))
            return_var = values[-1] # the name of this instruction
            tree.add_node(return_var, args, meta=line)
        else:
            program_mask.append(0)
            

    tree.calc_all_depths()

    return tree, program_mask




def __compute_depth_sorted__(tree):
    """
    computes an ordered list of lists that is ordered by increasing depth in the computational tree
    """
    max_depth = max(node.depth for node in tree.nodes.values())
    depth_sorting = [[] for x in range(max_depth + 1)]


    for node in tree.nodes.values():
        depth_sorting[node.depth].append(node.meta)
    return depth_sorting




def __sort_depths_within_depth_sorting__(depth_sorting):
    sorted_prog = []
    for depth_set in depth_sorting:
        scores = []
        lines = []
        for line in depth_set:
            if line is not None:
                score = 0
                tokens = [line['op'], line['p1'], line['p2'], line['p3']]
                for token in tokens:
                    if token in token_weight:
                        score += token_weight[token]
                scores.append(score)
                lines.append(line)
        # print(depth_set)
        # print(scores)
        sorting = np.argsort(scores)
        sorted_depth = [lines[x] for x in sorting]
        #sorted_depth = [line for _, line in sorted(scored_lines)]
        sorted_prog += sorted_depth

    #sorted_prog = sum(sorted_prog, [])
    return sorted_prog

def remap_vars(prog):
    var_mapper = dict(zip([x['r'] for x in prog], [f'v{i+1}' for i in range(len(prog))]))
    new_prog = []
    for line in prog:
        new_line = {}
        for k,v in line.items():
            if v not in var_mapper:
                new_line[k] = v
            else:
                new_line[k] = var_mapper[v]
        new_prog.append(new_line)
    return new_prog

ARG_ORDERING = UNI_LAMBDAS + SEQUENCE_LAMBDAS + NAMED_PREDICATES + ['indices', 'tokens']

def sort_args(prog):
    local_arg_ordreing = ARG_ORDERING + [f"v{i}" for i in range(len(prog))]  + ['NA']
    for line in prog:
        args = [line['p1'], line['p2'], line['p3']]
        token_weights = [local_arg_ordreing.index(arg) for arg in args]
        sorted_args = [args[i] for i in np.argsort(token_weights)]
        line['p1'] = sorted_args[0]
        line['p2'] = sorted_args[1]
        line['p3'] = sorted_args[2]
    return prog



def sort_program(prog):
    """
    inputs: a program as a list of instructions, where each instruction is a list of an operation name, followed by 4 variable/lambda/predicate/NA names

    returns:
        sorted_program - the input program sorted canonically
        program_mask - a mask over the input, value 1 for program instructions otherwise 0 for start/end tokens or padding
    """
    tree, _ = __build_tree__(prog)
    depth_sorting = __compute_depth_sorted__(tree)
    sorted_program = __sort_depths_within_depth_sorting__(depth_sorting)
    sorted_program = remap_vars(sorted_program)
    sorted_program = sort_args(sorted_program)
    return sorted_program

#%%



if __name__ == "__main__":
    prog = [
        ['PROGRAM_START', 'NA', 'NA', 'NA', 'NA' ],
        ['Map', 'LAM_ADD', 'tokens', 'NA', 'v1'],
        ['Map', 'LAM_MUL', 'v1', 'NA', 'v2'],
        ['Map', 'LAM_MUL', 'indices', 'NA', 'v3'],
        ['Select', 'v1', 'v2', 'PRED_NEQ', 'v4'],
        ['Aggregate', 'v4', 'v3', 'NA', 'v5'],
        ['Map', 'LAM_ADD', 'indices', 'NA', 'v6'],
        ['SequenceMap', 'LAM_OR', 'v6', 'v5', 'v7'],
        ['PROGRAM_END', 'NA', 'NA', 'NA', 'NA' ],
        ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
        ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
    ]

    prog = [dict(op=x[0], p1=x[1], p2=x[2], p3=x[3], r=x[4]) for x in prog]

    sorted_prog = sort_program(prog)

    assert( sorted_prog == [{'op': 'Map', 'p1': 'LAM_ADD', 'p2': 'tokens', 'p3': 'NA', 'r': 'v1'},
 {'op': 'Map', 'p1': 'LAM_ADD', 'p2': 'indices', 'p3': 'NA', 'r': 'v2'},
 {'op': 'Map', 'p1': 'LAM_MUL', 'p2': 'indices', 'p3': 'NA', 'r': 'v3'},
 {'op': 'Map', 'p1': 'LAM_MUL', 'p2': 'v1', 'p3': 'NA', 'r': 'v4'},
 {'op': 'Select', 'p1': 'PRED_NEQ', 'p2': 'v1', 'p3': 'v4', 'r': 'v5'},
 {'op': 'Aggregate', 'p1': 'v3', 'p2': 'v5', 'p3': 'NA', 'r': 'v6'},
 {'op': 'SequenceMap', 'p1': 'LAM_OR', 'p2': 'v2', 'p3': 'v6', 'r': 'v7'}])
    
    prog = [
            ['PROGRAM_START', 'NA', 'NA', 'NA', 'NA' ],
            ['Map', 'LAM_MUL', 'v1', 'NA', 'v2'],
            ['Map', 'LAM_MUL', 'indices', 'NA', 'v3'],
            ['Aggregate', 'v4', 'v3', 'NA', 'v5'],
            ['SequenceMap', 'LAM_OR', 'v6', 'indices', 'v7'],
            ['Map', 'LAM_ADD', 'v7', 'NA', 'v8'],
            ['Map', 'LAM_SUB', 'v8', 'NA', 'v9'],
            ['Map', 'LAM_SUB', 'v9', 'NA', 'v10'],
            ['SequenceMap', 'LAM_OR', 'v10', 'v5', 'v1'],
            ['Map', 'LAM_ADD', 'indices', 'NA', 'v6'],
            ['Map', 'LAM_LT', 'v6', 'NA', 'v11'],
            ['Select', 'v11', 'v11', 'PRED_NEQ', 'v4'],
            ['Aggregate', 'v4', 'v1', 'NA', 'v12'],
            ['Map', 'LAM_SUB', 'v12', 'NA', 'v13'],
            ['SequenceMap', 'LAM_SUB', 'v13', 'v2', 'v14'],
            ['PROGRAM_END', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
            ['<PAD>', 'NA', 'NA', 'NA', 'NA' ],
    ]

    prog = [dict(op=x[0], p1=x[1], p2=x[2], p3=x[3], r=x[4]) for x in prog]

    sorted_prog = sort_program(prog)

    assert sorted_prog == [{'op': 'Map', 'p1': 'LAM_ADD', 'p2': 'indices', 'p3': 'NA', 'r': 'v1'},
 {'op': 'Map', 'p1': 'LAM_MUL', 'p2': 'indices', 'p3': 'NA', 'r': 'v2'},
 {'op': 'Map', 'p1': 'LAM_LT', 'p2': 'v1', 'p3': 'NA', 'r': 'v3'},
 {'op': 'SequenceMap', 'p1': 'LAM_OR', 'p2': 'indices', 'p3': 'v1', 'r': 'v4'},
 {'op': 'Select', 'p1': 'PRED_NEQ', 'p2': 'v3', 'p3': 'v3', 'r': 'v5'},
 {'op': 'Map', 'p1': 'LAM_ADD', 'p2': 'v4', 'p3': 'NA', 'r': 'v6'},
 {'op': 'Aggregate', 'p1': 'v2', 'p2': 'v5', 'p3': 'NA', 'r': 'v7'},
 {'op': 'Map', 'p1': 'LAM_SUB', 'p2': 'v6', 'p3': 'NA', 'r': 'v8'},
 {'op': 'Map', 'p1': 'LAM_SUB', 'p2': 'v8', 'p3': 'NA', 'r': 'v9'},
 {'op': 'SequenceMap', 'p1': 'LAM_OR', 'p2': 'v7', 'p3': 'v9', 'r': 'v10'},
 {'op': 'Aggregate', 'p1': 'v5', 'p2': 'v10', 'p3': 'NA', 'r': 'v11'},
 {'op': 'Map', 'p1': 'LAM_MUL', 'p2': 'v10', 'p3': 'NA', 'r': 'v12'},
 {'op': 'Map', 'p1': 'LAM_SUB', 'p2': 'v11', 'p3': 'NA', 'r': 'v13'},
 {'op': 'SequenceMap', 'p1': 'LAM_SUB', 'p2': 'v12', 'p3': 'v13', 'r': 'v14'}]

