import sys
sys.path.append('tracr/')


from tracr.craft.transformers import MultiAttentionHead, MLP
from dataclasses import dataclass

from data.rasp_operators import *
import numpy as np
from collections import defaultdict
import tracr.compiler.lib as lib
from tracr.rasp import rasp

from data.dataset import choose_vocab_and_ops, build_program_of_length, compile_program_into_craft_model, program_craft_generator, program_generator
from data.dataloaders import ProgramEncoder
from functools import partial

#============================= Data Encoding ==============================================









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

"""
# Old version that can only be used as a generator
def program_craft_generator_bounded(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool, timeout_multiplier=1.0):
    max_prog_complexity = max(ops_range)
    CRAFT_TIMEOUT = 0.2 + 0.00001 * max_prog_complexity ** 4 # 10 op programs take 0.2 seconds, 15 op programs take 0.5, 30 op programs take 4 seconds
    CRAFT_TIMEOUT *= timeout_multiplier
    
    n_ops, vocab = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible, small_v_large_bias=3)

    # Self optimising timeout to adapt to local compute performance
    # if the average of the tally of successes to failures is less thatn the target, 
    # the time will increase to be successful more often
    IDEAL_FAILURE_RATIO = 0.4
    termination_tally = deque([0]*int((1-IDEAL_FAILURE_RATIO)*10) + [1]*int(IDEAL_FAILURE_RATIO*10),maxlen=30)


    def timed():
        try:
            program, actual_ops = build_program_of_length(vocab, numeric_range, MIN_PROG_LENGTH=max(2, n_ops-2), MAX_PROG_LENGTH=min(n_ops+2, ops_range[1]))
        except Exception as E:
            if isinstance(E, np.core._exceptions._ArrayMemoryError):
                print("mem alloc err")
            else:
                print(E)
                return None
        craft_model = compile_program_into_craft_model(program, vocab, max(numeric_range))
        encoded_ops = ProgramEncoder.encode_ops(actual_ops)
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

"""


def program_craft_generator_bounded(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool, timeout_multiplier=1.0):
    max_prog_complexity = max(ops_range)
    CRAFT_TIMEOUT = 0.2 + 0.00001 * max_prog_complexity ** 4 # 10 op programs take 0.2 seconds, 15 op programs take 0.5, 30 op programs take 4 seconds
    CRAFT_TIMEOUT *= timeout_multiplier
    
    n_ops, vocab = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible, small_v_large_bias=3)

    # Self optimising timeout to adapt to local compute performance
    # if the average of the tally of successes to failures is less thatn the target, 
    # the time will increase to be successful more often
    IDEAL_FAILURE_RATIO = 0.4
    termination_tally = deque([0]*int((1-IDEAL_FAILURE_RATIO)*10) + [1]*int(IDEAL_FAILURE_RATIO*10),maxlen=30)


    def timed():
        try:
            program, actual_ops = build_program_of_length(vocab, numeric_range, MIN_PROG_LENGTH=max(2, n_ops-2), MAX_PROG_LENGTH=min(n_ops+2, ops_range[1]))
        except Exception as E:
            if isinstance(E, np.core._exceptions._ArrayMemoryError):
                print("mem alloc err")
            else:
                print(E)
                return None
        craft_model = compile_program_into_craft_model(program, vocab, max(numeric_range))
        encoded_ops = ProgramEncoder.encode_ops(actual_ops)
        encoded_model = encode_craft_model(craft_model)
        return encoded_model, encoded_ops
    
    def generate_sample(timed, CRAFT_TIMEOUT, IDEAL_FAILURE_RATIO, termination_tally):
        ret = None
        while ret == None:
            ret = time_sensitive(timed, timeout=CRAFT_TIMEOUT)
            if ret == None:
                termination_tally.append(1) # we had a failure
                CRAFT_TIMEOUT = max(0.2, CRAFT_TIMEOUT + mean(termination_tally) - IDEAL_FAILURE_RATIO) # update timeout
        
        termination_tally.append(0)
        CRAFT_TIMEOUT = max(0.2, CRAFT_TIMEOUT + mean(termination_tally) - IDEAL_FAILURE_RATIO) # update timeout
        return ret

    return partial(generate_sample, timed, CRAFT_TIMEOUT, IDEAL_FAILURE_RATIO, termination_tally)





def program_craft_generator_unbounded(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool):
    def generate_sample():
        craft_model, actual_ops = program_craft_generator(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
        encoded_ops = ProgramEncoder.encode_ops(actual_ops)
        encoded_model = encode_craft_model(craft_model)

        return encoded_model, encoded_ops
    return generate_sample




# ========================= User Friendly Generators ============================================




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
    
    OP_VOCAB, VAR_VOCAB = ProgramEncoder.get_vocabs(max(ops_range))
    
    if timeout_multiplier is not None:
        lambda x,y,z: func(x,y,z, timeout_multiplier=timeout_multiplier)
    
    gen = func(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
            
    
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
    OP_VOCAB, VAR_VOCAB = ProgramEncoder.get_vocabs(max(ops_range))
                    
    def gen():
        while True:
            _, actual_ops = program_generator(ops_range, vocab_size_range, numeric_range, numeric_inputs_possible=numeric_inputs_possible)
            encoded_ops = ProgramEncoder.encode_ops(actual_ops)
            yield encoded_ops
    
    return gen, OP_VOCAB, VAR_VOCAB






from data.dataloaders import ProgramEncoder
from data.dataset import traverse_prog, gen_vocab, compile_program_into_craft_model
def encode_rasp_program(program, PROG_LEN, lambdas=[], numeric_vars: bool = False):
    actual_ops = traverse_prog(program, lambdas)
    vocab = gen_vocab(PROG_LEN, prefix='t', numeric=numeric_vars)
    craft_model = compile_program_into_craft_model(program, vocab, PROG_LEN)

    encoded_ops = ProgramEncoder.encode_ops(actual_ops)
    encoded_model = encode_craft_model(craft_model)
    return encoded_model, encoded_ops