
import torch
import numpy as np
from data.rasp_operators import *
from typing import Union, TypeVar, Sequence, Callable, Optional
from data.canonical_ordering import sort_program

START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'



from jax import numpy as jnp






class ProgramEncoder:
    def __init__(self, max_prog_length) -> None:
        self.OP_VOCAB, self.VAR_VOCAB = ProgramEncoder.get_vocabs(max_prog_length)
        self.OP_VOCAB_SIZE, self.VAR_VOCAB_SIZE = len(self.OP_VOCAB), len(self.VAR_VOCAB)
        self.op_encoder = dict(zip(self.OP_VOCAB, [i for i in range(self.OP_VOCAB_SIZE)]))
        
        self.op_encoder[START_TOKEN] = self.OP_VOCAB_SIZE # Add a token for the start of the program
        self.OP_VOCAB_SIZE += 1
        self.op_encoder[END_TOKEN] = self.OP_VOCAB_SIZE # Add a token for the end of the program
        self.OP_VOCAB_SIZE += 1
        
        self.var_encoder = dict(zip(self.VAR_VOCAB, [i for i in range(self.VAR_VOCAB_SIZE)]))


        
        self.op_decoder = dict(zip(self.op_encoder.values(), self.op_encoder.keys()))
        self.var_decoder = dict(zip(self.var_encoder.values(), self.var_encoder.keys()))

        self.segment_sizes = [self.OP_VOCAB_SIZE, 
                              self.VAR_VOCAB_SIZE, 
                              self.VAR_VOCAB_SIZE, 
                              self.VAR_VOCAB_SIZE, 
                              self.VAR_VOCAB_SIZE]
    

    
    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], self.OP_VOCAB_SIZE))
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
    
    def decode_pred(self, pred):
        translated = str()
        for t in range(pred.shape[0]):
            # if pred[t].sum().item() == 0: # skip padding
            #     continue
            op = self.op_decoder[pred[t, 0].item()]
            translated += op
            for i in range(1,5):
                translated += " " + self.var_decoder[pred[t, i].item()]
            translated += "\n"
        return translated
    
    def tokenise_program(self, program):
        encoded = np.zeros((len(program)+2, 5), np.int32)
        encoded[0, 0] = self.op_encoder[START_TOKEN]
        for t, instruction in enumerate(program):
            # Loop through each operation which cotains list of {'op': 'SelectorWidth', 'p1': 'v1', 'p2': 'NA', 'p3': 'NA', 'r': 'v2'}
            encoded[t+1, 0] = self.op_encoder[instruction['op']]
            encoded[t+1, 1] = self.var_encoder[instruction['p1']]
            encoded[t+1, 2] = self.var_encoder[instruction['p2']]
            encoded[t+1, 3] = self.var_encoder[instruction['p3']]
            encoded[t+1, 4] = self.var_encoder[instruction['r']]
        encoded[-1, 0] = self.op_encoder[END_TOKEN]
        return encoded
    
    def tokens_to_onehot(self, encoded, ignore_padding=False):
        one_hot = np.zeros((encoded.shape[0], self.OP_VOCAB_SIZE + 4 * self.VAR_VOCAB_SIZE))
        for t in range(encoded.shape[0]):
            ptr = 0
            # Loop through each operation which cotains list of 5 integer id's for each token
            for i in range(len(self.segment_sizes)):
                id = encoded[t, i]
                if not (ignore_padding and id == 0):
                    one_hot[t, ptr + id] = 1
                ptr += self.segment_sizes[i]
        return one_hot
    
    
    def program_to_onehot(self, program):
        encoded = self.tokenise_program(program)
        onehot = self.tokens_to_onehot(encoded)
        return onehot
    
    def iter_var_names(prefix='v'):
        i = 0
        while True:
            i += 1
            yield prefix + str(i)

    def get_vocabs(max_ops: int):
        OP_VOCAB = ['<PAD>'] + list(RASP_OPS.cls.apply(lambda x: x.__name__))
        var_name_iter = ProgramEncoder.iter_var_names()
        VAR_VOCAB = ['<PAD>'] + ['tokens', 'indices'] \
                        + list(NAMED_PREDICATES.values()) \
                        + list(x[-1] for x in UNI_LAMBDAS + SEQUENCE_LAMBDAS) + [NO_PARAM] \
                        + [next(var_name_iter) for x in range(0, max_ops)] 
        return OP_VOCAB, VAR_VOCAB
    
    def encode_ops(ops):
        features = []
        var_name_iter = ProgramEncoder.iter_var_names()
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



