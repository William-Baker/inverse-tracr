
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


#%%

import jax
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
from tracr.rasp import rasp
from dataclasses import dataclass
import pandas as pd

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')






SOp = rasp.SOp
Selector = rasp.Selector


NumericValue = Union[int, float]
SOpNumericValue = Union[SOp, int, float]
Value = Union[None, int, float, str, bool]
lambda1 = TypeVar("lambda1")
lambda2 = TypeVar("lambda2")
NO_PARAM = 'NA'


df = [
    # Class name,         input types,                 output type, weight
    [ rasp.Map,          [lambda1, SOp],                  SOp     , 4],
    [ rasp.Select,       [SOp, SOp, rasp.Predicate],           Selector, 3],
    [ rasp.SequenceMap,  [SOp, SOpNumericValue, lambda2], SOp,      2],
    [ rasp.Aggregate,    [Selector, SOp],                 SOp,      2],
    [ rasp.SelectorWidth,[Selector],                      SOp,      2],
    # [ rasp.SelectorOr,   [Selector, Selector],            Selector, 1],  # These arent implemented correclty in RASP
    # [ rasp.SelectorAnd,  [Selector, Selector],            Selector, 1], 
    # [ rasp.SelectorNot,  [Selector, Selector],            Selector, 1], 
]

df = pd.DataFrame(df, columns = ['cls', 'inp', 'out', 'weight'])
df_no_selector = pd.DataFrame(df[df.inp.apply(lambda x: Selector not in x)])
df_returns_sop = pd.DataFrame(df[df.out == SOp])



UNI_LAMBDAS = [
    (lambda x, y: x < y,  'LAM_LT'), 
    (lambda x, y: x <= y, 'LAM_LE'),       
    (lambda x, y: x > y,  'LAM_GT'),      
    (lambda x, y: x >= y, 'LAM_GE'),       
    (lambda x, y: x != y, 'LAM_NE'),     
    (lambda x, y: x == y, 'LAM_EQ'),      
    (lambda x, y: not x,  'LAM_IV'),   
]

SEQUNCE_LAMBDAS = [
    (lambda x, y: x + y,  [] , 'LAM_ADD'),
    (lambda x, y: x * y,  [] , 'LAM_MUL'),
    (lambda x, y: x - y,  [] , 'LAM_SUB'),
    #(lambda x, y: x / y,  [0], 'LAM_DIV'),
    (lambda x, y: x and y,[] , 'LAM_AND'),
    (lambda x, y: x or y, [] , 'LAM_OR'),
]


PREDICATES = [
    rasp.Comparison.EQ,
    rasp.Comparison.FALSE,
    rasp.Comparison.TRUE,
    rasp.Comparison.GEQ,
    rasp.Comparison.GT,
    rasp.Comparison.LEQ,
    rasp.Comparison.LT,
    rasp.Comparison.NEQ,
]

named_predicates = dict([
    (rasp.Comparison.EQ,    'PRED_EQ'),
    (rasp.Comparison.FALSE, 'PRED_FALSE'),
    (rasp.Comparison.TRUE,  'PRED_TRUE'),
    (rasp.Comparison.GEQ,   'PRED_GEQ'),
    (rasp.Comparison.GT,    'PRED_GT'),
    (rasp.Comparison.LEQ,   'PRED_LEQ'),
    (rasp.Comparison.LT,    'PRED_LT'),
    (rasp.Comparison.NEQ,   'PRED_NEQ'),
])



from collections import defaultdict
from enum import Enum
class Cat(Enum):
    numeric = 1
    categoric = 2
    boolean = 3


class Scope:
    def __init__(self, vocabulary: Sequence[Union[str, int, bool]], max_seq_length) -> None:    
        self.scope = dict()
        self.names = set()
        self.types = set()
        self.counter = defaultdict(lambda: 0)
        self.names_by_type = defaultdict(lambda: [])
        self.names_by_type_and_cat = defaultdict(lambda: defaultdict(lambda: []))
        self.type_cat = dict()
        self.sampling_weights = [1,1]
        if type(vocabulary[0]) == str:
            token_type = Cat.categoric
        else:
            token_type = Cat.numeric
        self.__add__(SOp, "tokens", token_type, weight=1)
        self.__add__(SOp, "indices", Cat.numeric, weight=1)
        self.sampling_weights = self.sampling_weights[2:]
        self.max_seq_length = max_seq_length
        self.vocabulary = vocabulary

    def add(self, t: type, cat: Cat, weight: Optional[int] = None) -> None:
        self.counter[t] += 1
        return self.__add__(t, str(t) + ' ' + str(self.counter[t]), cat, weight)
    
    def __add__(self, t: type, name: str, cat: Cat, weight: Optional[int] = None) -> None:
        if name in self.names:
            raise ValueError(f'{name} is already present in the set')
        self.scope[name] = t
        self.type_cat[name] = cat
        self.names.add(name)
        self.types.add(t)
        new_weight = (self.sampling_weights[-1] * 2 - self.sampling_weights[-2] + 1)
        sample_weight = new_weight if weight is None else weight
        self.sampling_weights.append(sample_weight)
        self.names_by_type[t].append((name, sample_weight))
        self.names_by_type_and_cat[t][cat].append((name, sample_weight))
        
        return name
    
    def get_cat(self, name: str):
        return self.type_cat[name]
    
    def weighted_sample(lst):
        weights = [x[1] for x in lst]
        sample = choices(lst, weights, k=1)
        return sample[0][0]

    def pick_var(self, y: type): 
        return Scope.weighted_sample(self.names_by_type[y])
    
    def pick_var_cat(self, y: type, cat: Cat):
        return Scope.weighted_sample(self.names_by_type_and_cat[y][cat])
    
    def var_exists(self, desired_type: type):
        return len(self.names_by_type[desired_type]) > 0
    
    def var_exists_cat(self, desired_type: type, desired_cat: Cat):
        return len(self.names_by_type_and_cat[desired_type][desired_cat]) > 0
    
    def matching_var_exists(self, src, desired_type: type):
        """returns True if a variable with the desired type is in scope with category matching the source"""
        return self.var_exists_cat(desired_type, self.get_cat(src))
    
    def gen_const(self, target_cat: Cat):
        if target_cat == Cat.numeric:
            return randint(0, self.max_seq_length)
        elif target_cat == Cat.categoric:
            return choice(self.vocabulary)
        else:
            raise NotImplementedError()

@dataclass
class Operation:
    operator: type
    inputs: Sequence[Union[str, Callable, rasp.Predicate ]]
    output: str
    lambda_name: Optional[str] = None


# =================================== Program Sampling ====================================


def sample_function(scope: Scope, ops, df=df):
    if scope.var_exists(Selector):
        sampled = df.sample(weights=df.weight).iloc[0]
    else:
        sampled = df_no_selector.sample(weights=df_no_selector.weight).iloc[0]


    if sampled.cls == rasp.Map:
        # [lambda1, SOp],
        
        return_type = SOp
        

        if randint(0,1) == 0: # Boolean operators
            return_cat = Cat.boolean
            
            s1 = scope.pick_var(SOp)
            f1, lambda_name = choice(UNI_LAMBDAS)
            
            obj_cat = scope.get_cat(s1)
            y = None
            if s1 == "tokens":
                y = scope.gen_const(Cat.categoric)
            elif obj_cat == Cat.numeric:
                y = scope.gen_const(Cat.numeric)
            elif obj_cat == Cat.categoric:
                y = scope.gen_const(Cat.categoric)
            elif obj_cat == Cat.boolean:
                y = bool(randint(0,1))
            else:
                raise NotImplementedError()
            
            func = partial(f1, y)

        else: # linear operators
            return_cat = Cat.numeric
            s1 = scope.pick_var_cat(SOp, return_cat)
            # TODO we can guarentee that the operand is non-zero, so lets make it possible to do division
            f1, bad_vals, lambda_name = choice(SEQUNCE_LAMBDAS)# + [(lambda x, y: x / y,  [0])]) 
            if  randint(0,2) <= 1: # 2/3 of the time generate an int
                s2 = scope.gen_const(Cat.numeric)
            else: # occasionally generate a float - may have a different impl
                s2 = float(scope.gen_const(Cat.numeric)) + randint(0,100)/100

            while s2 in bad_vals: # prevent bad values, such as div 0
                if isinstance(s2, float):
                    s2 += 0.1
                else:
                    s2 += 1

            func = partial(f1, s2)

        # Allocate a variable to hold the return value
        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [func, s1], allocated_name, lambda_name=lambda_name)
        ops.append(op)

    elif sampled.cls == rasp.SequenceMap: # must have double the weight of Map
        # [SOp, SOpNumericValue, lambda2],
        # todo chance of const full selector/sop
        s1 = scope.pick_var_cat(SOp, Cat.numeric)
        s1_cat = scope.get_cat(s1)
        if scope.var_exists_cat(SOp, s1_cat): # s2 wil be an SOp
            s2 = scope.pick_var_cat(SOp, s1_cat)
        else: 
            print("dead end")
            return
            
        f1, bad, lambda_name = choice(SEQUNCE_LAMBDAS)

        return_type = SOp
        return_cat = s1_cat

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [f1, s1, s2], allocated_name, lambda_name)
        ops.append(op)

    elif sampled.cls == rasp.Select:
        # [SOp, SOp, rasp.Predicate],    
        # todo chance of const full selector/sop
        s1 = scope.pick_var(SOp)
        s2 = scope.pick_var_cat(SOp, scope.get_cat(s1))
        pred = choice(PREDICATES)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2, pred], allocated_name)
        ops.append(op)

    elif sampled.cls == rasp.Aggregate:
        # [Selector, SOp],
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)
        s2 = scope.pick_var_cat(SOp, Cat.numeric)

        return_type = SOp
        return_cat = Cat.numeric

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2], allocated_name)
        ops.append(op)


    elif sampled.cls == rasp.SelectorWidth:
        # [Selector],                     
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)

        return_type = SOp
        return_cat = Cat.numeric

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1], allocated_name)
        ops.append(op)


    elif (sampled.cls == rasp.SelectorOr) or (sampled.cls == rasp.SelectorAnd):
        # [Selector, Selector],           
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)
        s2 = scope.pick_var(Selector)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2], allocated_name)
        ops.append(op)

    elif sampled.cls == rasp.SelectorNot:
        s1 = scope.pick_var(Selector)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1], allocated_name)
        ops.append(op)

    else:
        raise NotImplementedError()


def generate_ops(max_ops: int, vocab: Sequence, max_seq_len: int):
    scope = Scope(vocab, max_seq_len)
    ops = []

    for i in range(0, max_ops-1):
        sample_function(scope, ops, df )
    sample_function(scope, ops, df_returns_sop)

    return ops

def compile_program(ops):
    @dataclass
    class Program:
        ops: Sequence[Operation]
        
        def __post_init__(self):
            self.named_ops = dict((op.output, op) for op in self.ops)
            
    actual_ops = []
    def populate_params(op: Operation, prog: Program):
        actual_ops.append(op)
        params = []
        for inp in op.inputs:
            if isinstance(inp, str):
                if inp == 'tokens':
                    params.append(rasp.tokens)
                elif inp == 'indices':
                    params.append(rasp.indices)
                else:
                    child = populate_params(prog.named_ops[inp], prog)
                    params.append(child)
            elif isinstance(inp, Callable):
                params.append(inp)
            elif isinstance(inp, Union[float, int]):
                params.append(inp)
        ret = op.operator(*params)
        named_ret = ret.named(op.output)
        return named_ret


    program = populate_params(ops[-1], Program(ops))

    # discard duplicates
    seen = []
    actual_ops = list(filter(lambda x: seen.append(x.output) is None if x.output not in seen else False, actual_ops))
    actual_ops = actual_ops[::-1] # reverse the traversal
    #print(f"Program Length: {len(actual_ops)}")


    return program, actual_ops

def compile_program_into_craft_model(program, vocab, max_seq_len):

    COMPILER_BOS = "compiler_bos"
    COMPILER_PAD = "compiler_pad"



    compiler_bos = COMPILER_BOS
    compiler_pad = COMPILER_PAD
    mlp_exactness = 100

    if compiler_bos in vocab:
        raise ValueError("Compiler BOS token must not be present in the vocab. "
                        f"Found '{compiler_bos}' in {vocab}")

    if compiler_pad in vocab:
        raise ValueError("Compiler PAD token must not be present in the vocab. "
                        f"Found '{compiler_pad}' in {vocab}")

    rasp_model = rasp_to_graph.extract_rasp_graph(program)
    graph, sources, sink = rasp_model.graph, rasp_model.sources, rasp_model.sink

    basis_inference.infer_bases(
        graph,
        sink,
        vocab,
        max_seq_len,
    )


    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
        mlp_exactness=mlp_exactness,
    )

    craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

    return craft_model


def gen_vocab(vocab_size: int, prefix='t'):
    return [prefix+str(x) for x in range(vocab_size)]


def build_program_of_length(n_ops, vocab, max_seq_len, TARGET_PROGRAM_LENGTH):
    program_length = 0
    while program_length < TARGET_PROGRAM_LENGTH:
        ops = generate_ops(n_ops, vocab, max_seq_len)
        program, actual_ops = compile_program(ops)
        program_length = len(actual_ops)
    return program, actual_ops

def program_generator(ops_range: tuple, vocab_size_range: tuple, max_sequence_lenghts_range: tuple):
    n_ops = randint(*ops_range)
    vocab_size = randint(*vocab_size_range)
    max_seq_len = randint(*max_sequence_lenghts_range)
    TARGET_PROGRAM_LENGTH = max(ops_range) // 2
    vocab = gen_vocab(n_ops, prefix='t')
    program, actual_ops = build_program_of_length(n_ops, vocab, max_seq_len, TARGET_PROGRAM_LENGTH)
    return actual_ops


def program_craft_generator(ops_range: tuple, vocab_size_range: tuple, max_sequence_lenghts_range: tuple):
    n_ops = randint(*ops_range)
    vocab_size = randint(*vocab_size_range)
    max_seq_len = randint(*max_sequence_lenghts_range)
    TARGET_PROGRAM_LENGTH = max(ops_range) // 2
    vocab = gen_vocab(n_ops, prefix='t')
    program, actual_ops = build_program_of_length(n_ops, vocab, max_seq_len, TARGET_PROGRAM_LENGTH)
    craft_model = compile_program_into_craft_model(program, vocab, max_seq_len)
    return craft_model, actual_ops


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
            elif inp in named_predicates.keys():
                params[i] = named_predicates[inp]  
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
    return features

def program_dataset(ops_range=(10,10), vocab_size_range=(6,6), max_sequence_lenghts_range=(6,6)):
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
    OP_NAME_VOCAB = list(df.cls.apply(lambda x: x.__name__))
    var_name_iter = iter_var_names()
    VAR_VOCAB = ['tokens', 'indices'] \
                    + [next(var_name_iter) for x in range(0, max(ops_range))]  \
                    + list(named_predicates.values()) \
                    + list(x[-1] for x in UNI_LAMBDAS + SEQUNCE_LAMBDAS) + [NO_PARAM]
    def gen():
        while True:
            actual_ops = program_generator(ops_range, vocab_size_range, max_sequence_lenghts_range)
            encoded_ops = encode_ops(actual_ops)
            yield encoded_ops
    
    return gen, OP_NAME_VOCAB, VAR_VOCAB


START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

def encode_program(program, op_encoder, var_encoder):
    encoded = np.zeros((len(program)+1, 5), np.int32)
    encoded[0, 0] = op_encoder[START_TOKEN]
    for t, instruction in enumerate(program):
        # Loop through each operation which cotains list of {'op': 'SelectorWidth', 'p1': 'v1', 'p2': 'NA', 'p3': 'NA', 'r': 'v2'}
        encoded[t+1, 0] = op_encoder[instruction['op']]
        encoded[t+1, 1] = var_encoder[instruction['p1']]
        encoded[t+1, 2] = var_encoder[instruction['p2']]
        encoded[t+1, 3] = var_encoder[instruction['p3']]
        encoded[t+1, 4] = var_encoder[instruction['r']]
    encoded[-1, 0] = op_encoder[START_TOKEN]
    return encoded

def encoded_program_to_onehot(encoded, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE):
    one_hot = np.zeros((encoded.shape[0], OP_NAME_VOCAB_SIZE + 4 * VAR_VOCAB_SIZE))
    segment_sizes = [OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]
    for t in range(encoded.shape[0]):
        ptr = 0
        # Loop through each operation which cotains list of 5 integer id's for each token
        for i in range(len(segment_sizes)):
            id = encoded[t, i]
            #print(f"ID: {id}, x: {ptr + id}, y: {t}")
            one_hot[t, ptr + id] = 1
            ptr += segment_sizes[i]
    return one_hot





def decoder_generator(_program_gen, op_encoder, var_encoder, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE):
    while True:
        program = next(_program_gen)
        encoded_program = encode_program(program, op_encoder, var_encoder)
        onehot_program = encoded_program_to_onehot(encoded_program, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)
        yield onehot_program, encoded_program

from functools import partial
from jax import numpy as jnp

class ProgramDataset(torch.utils.data.Dataset):
    def __init__(self, prog_len, sample_count=1000):
        self.prog_len = prog_len
        self.sample_count = sample_count
        gen, OP_NAME_VOCAB, VAR_VOCAB = program_dataset(ops_range=(prog_len,prog_len))
        OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
        op_encoder = dict(zip(OP_NAME_VOCAB, [i for i in range(OP_NAME_VOCAB_SIZE)]))
        
        op_encoder[START_TOKEN] = OP_NAME_VOCAB_SIZE # Add a token for the start of the program
        OP_NAME_VOCAB_SIZE += 1
        op_encoder[END_TOKEN] = OP_NAME_VOCAB_SIZE # Add a token for the end of the program
        OP_NAME_VOCAB_SIZE += 1
        
        var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))
        self.data_iterator = decoder_generator(gen(), op_encoder, var_encoder, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)

        self.OP_NAME_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.op_encoder, self.var_encoder = \
                    OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, op_encoder, var_encoder
        
        self.op_decoder = dict(zip(op_encoder.values(), op_encoder.keys()))
        self.var_decoder = dict(zip(var_encoder.values(), var_encoder.keys()))

        self.segment_sizes = [OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]

    def __len__(self):
        'Denotes the total number of samples'
        return self.sample_count

    def __getitem__(self, index):
        return next(self.data_iterator)
    
    def collate_fn(self, data):
        inputs = [torch.tensor(d[0]) for d in data]
        targets = [torch.tensor(d[1]) for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        ammount_to_pad = self.prog_len + 2 - targets.shape[1]
        targets = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(targets) # pad the target to the max possible length for the problem
        return jnp.array(inputs), jnp.array(targets)

    def get_collate_fn(self):
        return partial(ProgramDataset.collate_fn, self)

    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], 5))
        logits = np.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax()
                ptr += seg_size
        return classes

    def logit_classes_jnp(self, logits):
        classes = jnp.zeros((logits.shape[0], 5))
        logits = jnp.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax().item()
                ptr += seg_size
        return classes
    
    def decode_pred(self, y, batch_index: int):
        pred = y[batch_index, :, :]

        if pred.shape[-1] > 5: # compute the argmax in each segment
            pred = self.logit_classes_np(pred)

        translated = str()
        for t in range(pred.shape[0]):
            if pred[t, :].sum().item() == 0:
                translated += "<PAD>\n"
                continue
            op = self.op_decoder[pred[t, 0].item()]
            translated += op
            if op not in [START_TOKEN, END_TOKEN]:
                for i in range(1,5):
                    translated += " " + self.var_decoder[pred[t, i].item()]
            translated += "\n"
        return translated

#%%

from models import EncoderDecoder, TransformerEncoder, EncoderBlock
from jax import random
import jax.numpy as jnp
from utils.jax_helpers import JaxSeeder
import os
CHECKPOINT_PATH = ".logs/"
from torch.utils.tensorboard import SummaryWriter
import jax
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm
import numpy as np
import torch

torch.cuda.is_available = lambda : False

from torch.utils.data import DataLoader
from utils.dataloaders import ProgramDataset




dataset = ProgramDataset(30, 20000)
train_dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.get_collate_fn() , num_workers=2)#, prefetch_factor=2, pin_memory=True)


it = iter(train_dataloader)
x,y = next(it)