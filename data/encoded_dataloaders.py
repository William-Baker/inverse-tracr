import sys
sys.path.append('tracr/')

from typing import Union, TypeVar, Sequence, Callable, Optional
from random import choice, randint, choices
from functools import partial
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp
from tracr.craft.transformers import MultiAttentionHead, MLP
from dataclasses import dataclass
from data.canonical_ordering import sort_program
from data.rasp_operators import *
import numpy as np
from data.sigterm import guard_timeout, TimeoutException
import inspect
from collections import defaultdict
from enum import Enum
import tracr.compiler.lib as lib
from tracr.rasp import rasp

from data.dataset import choose_vocab_and_ops, build_program_of_length, compile_program_into_craft_model, program_craft_generator, program_generator

#============================= Data Encoding ==============================================

def iter_var_names(prefix='v'):
    i = 0
    while True:
        i += 1
        yield prefix + str(i)




def encode_ops(ops):
    features = []
    var_name_iter = iter_var_names()
    var_names = dict(tokens= 'tokens', indices='indices')
    for op in ops:
        op_name = op.operator.__name__
        params = [NO_PARAM] * 3
        for i, inp in enumerate(op.inputs):
            if isinstance(inp, str):
                if inp in var_names:
                    params[i] = var_names[inp]
                else:
                    var_names[inp] = next(var_name_iter)
                    params[i] = var_names[inp]
            # elif isinstance(inp, partial): # ignore the value of the parameter when using a given lambda
            elif inp in NAMED_PREDICATES.keys():
                params[i] = NAMED_PREDICATES[inp]  
            elif isinstance(inp, Callable):
                assert op.lambda_name != None
                params[i] = op.lambda_name
            
            
        return_var = op.output
        if return_var in var_names:
            ret_name = var_names[return_var]
        else:
            var_names[return_var] = next(var_name_iter)
            ret_name = var_names[return_var]
        
        feature = dict(
            op = op_name,
            p1 = params[0],
            p2 = params[1],
            p3 = params[2],
            r = ret_name
        )
        features.append(feature)
    
    # Apply our canonical program ordering to the program
    features = sort_program(features)
    
    return features


def encode_craft_model(craft_model):
    model_params = []
    for block in craft_model.blocks:
        if isinstance(block, MultiAttentionHead):
            for attention_head in block.heads():
                model_params.append(dict(
                    HEAD = dict(
                        w_qk = attention_head.w_qk.matrix,
                        w_ov = attention_head.w_ov.matrix
                    )
                ))
        elif isinstance(block, MLP):
            model_params.append(dict(
                MLP = dict(
                    fst = block.fst.matrix,
                    snd = block.snd.matrix
                )
            ))
        else:
            raise NotImplementedError()
    return model_params



# ====== encoder with timeout ==================
from collections import deque
from statistics import mean

from utils.time_sensitive import time_sensitive

def program_craft_generator_bounded(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool, timeout_multiplier=1.0):
    max_prog_complexity = max(ops_range)
    CRAFT_TIMEOUT = 0.2 + 0.00001 * max_prog_complexity ** 4 # 10 op programs take 0.2 seconds, 15 op programs take 0.5, 30 op programs take 4 seconds
    CRAFT_TIMEOUT *= timeout_multiplier
    
    n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)

    # Self optimising timeout to adapt to local compute performance
    # if the average of the tally of successes to failures is less thatn the target, 
    # the time will increase to be successful more often
    IDEAL_FAILURE_RATIO = 0.4
    termination_tally = deque([0]*int((1-IDEAL_FAILURE_RATIO)*10) + [1]*int(IDEAL_FAILURE_RATIO*10),maxlen=30)


    def timed():
        try:
            program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
        except Exception as E:
            if isinstance(E, np.core._exceptions._ArrayMemoryError):
                print("mem alloc err")
            else:
                print(E)
                return None
        craft_model = compile_program_into_craft_model(program, vocab, max(numeric_range))
        encoded_ops = encode_ops(actual_ops)
        encoded_model = encode_craft_model(craft_model)
        return encoded_model, encoded_ops
    
    while True:
        ret = None
        while ret == None:
            ret = time_sensitive(timed, timeout=CRAFT_TIMEOUT)
            if ret == None:
                termination_tally.append(1) # we had a failure
                CRAFT_TIMEOUT = max(0.2, CRAFT_TIMEOUT + mean(termination_tally) - IDEAL_FAILURE_RATIO) # update timeout
        
        termination_tally.append(0)
        CRAFT_TIMEOUT = max(0.2, CRAFT_TIMEOUT + mean(termination_tally) - IDEAL_FAILURE_RATIO) # update timeout
        yield ret









def program_craft_generator_unbounded(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool):
    while True:
        craft_model, actual_ops = program_craft_generator(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
        encoded_ops = encode_ops(actual_ops)
        encoded_model = encode_craft_model(craft_model)

        yield encoded_model, encoded_ops




# ========================= User Friendly Generators ============================================


def get_vocabs(max_ops: int):
    OP_VOCAB = ['<PAD>'] + list(RASP_OPS.cls.apply(lambda x: x.__name__))
    var_name_iter = iter_var_names()
    VAR_VOCAB = ['<PAD>'] + ['tokens', 'indices'] \
                    + list(NAMED_PREDICATES.values()) \
                    + list(x[-1] for x in UNI_LAMBDAS + SEQUENCE_LAMBDAS) + [NO_PARAM] \
                    + [next(var_name_iter) for x in range(0, max_ops)] 
    return OP_VOCAB, VAR_VOCAB

def craft_dataset(ops_range=(10,10), vocab_size_range=(6,6), numeric_range=(6,6), func=program_craft_generator_unbounded, timeout_multiplier=None, numeric_inputs_possible=False):
    """
    Compile a generator of transformer weights and programs
    params:
        ops_range = (min, max) - the range of program lengths to generate BEFORE retracing reduction
        vocab_size_range = (min, max) - the range of vocabulary sizes to build the transformer with
                                            - will scale the number of parameter
        max_sequence_lengths_range = (min, max) - the range of sequence lengths to build the code AND transformer with 
                                            - will scale the number of constants used in lambdas
                                            - and the number of model parameters

    returns:
        gen - generator that yields (model_params, program)
        OP_VOCAB - vocab for the craft functions ['Map', 'Select', 'SequenceMap', 'Aggregate', 'SelectorWidth']
        VAR_VOCAB - [ 'tokens', 'indices', 'NA',
                    'PRED_EQ', 'PRED_FALSE', 'PRED_TRUE', 'PRED_GEQ', 'PRED_GT', 'PRED_LEQ', 'PRED_LT', 'PRED_NEQ',
                    'LAM_LT', 'LAM_LE', 'LAM_GT', 'LAM_GE', 'LAM_NE', 'LAM_EQ', 'LAM_IV', 'LAM_ADD', 'LAM_MUL', 'LAM_SUB', 'LAM_AND', 'LAM_OR', 
                    'vXXX' - where XXX is the ID of variable - in range (0, ops_range_max)
                            - NOTE previously I said it was in range (0, ops_range_max * 2), but now i dont think so
                    ]
    """
    
    OP_VOCAB, VAR_VOCAB = get_vocabs(max(ops_range))
    
    if timeout_multiplier is not None:
        lambda x,y,z: func(x,y,z, timeout_multiplier=timeout_multiplier)
    
    gen = lambda: func(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
            
    
    return gen, OP_VOCAB, VAR_VOCAB


def program_dataset(ops_range=(10,10), vocab_size_range=(6,6), numeric_range=(6,6), numeric_inputs_possible: bool = False):
    """
    Compile a generator of programs ONLY
    params:
        ops_range = (min, max) - the range of program lengths to generate BEFORE retracing reduction
        vocab_size_range = (min, max) - the range of vocabulary sizes to build the transformer with
                                            - will scale the number of parameter
        max_sequence_lengths_range = (min, max) - the range of sequence lengths to build the code AND transformer with 
                                            - will scale the number of constants used in lambdas
                                            - and the number of model parameters

    returns:
        gen - generator that yields (model_params, program)
        OP_VOCAB - vocab for the craft functions ['Map', 'Select', 'SequenceMap', 'Aggregate', 'SelectorWidth']
        VAR_VOCAB - [ 'tokens', 'indices', 'NA',
                    'PRED_EQ', 'PRED_FALSE', 'PRED_TRUE', 'PRED_GEQ', 'PRED_GT', 'PRED_LEQ', 'PRED_LT', 'PRED_NEQ',
                    'LAM_LT', 'LAM_LE', 'LAM_GT', 'LAM_GE', 'LAM_NE', 'LAM_EQ', 'LAM_IV', 'LAM_ADD', 'LAM_MUL', 'LAM_SUB', 'LAM_AND', 'LAM_OR', 
                    'vXXX' - where XXX is the ID of variable - in range (0, ops_range_max)
                            - NOTE previously I said it was in range (0, ops_range_max * 2), but now i dont think so
                    ]
    """
    OP_VOCAB, VAR_VOCAB = get_vocabs(max(ops_range))
                    
    def gen():
        while True:
            _, actual_ops = program_generator(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
            encoded_ops = encode_ops(actual_ops)
            yield encoded_ops
    
    return gen, OP_VOCAB, VAR_VOCAB
