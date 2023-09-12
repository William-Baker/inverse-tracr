#%%
from tracr.compiler.compiling import compile_rasp_to_model
from tracr.rasp import rasp



# SelectorWidth'>, inputs=["<class 'tracr.rasp.rasp.Selector'> 2"], output="<class 'tracr.rasp.rasp.SOp'> 3", lambda_name=None),
# SequenceMap'>, inputs=[<function <lambda> at 0x7f1e2b026290>, "<class 'tracr.rasp.rasp.SOp'> 3", "<class 'tracr.rasp.rasp.SOp'> 4"], output="<class 'tracr.rasp.rasp.SOp'> 6", lambda_name='LAM_SUB'),
# Select'>, inputs=['indices', 'indices', <Comparison.EQ: '=='>], output="<class 'tracr.rasp.rasp.Selector'> 1", lambda_name=None),
# Aggregate'>, inputs=["<class 'tracr.rasp.rasp.Selector'> 1", 'indices'], output="<class 'tracr.rasp.rasp.SOp'> 2", lambda_name=None),
# Select'>, inputs=['tokens', 'tokens', <Comparison.LT: '<'>], output="<class 'tracr.rasp.rasp.Selector'> 2", lambda_name=None),
# SelectorWidth'>, inputs=["<class 'tracr.rasp.rasp.Selector'> 2"], output="<class 'tracr.rasp.rasp.SOp'> 4", lambda_name=None),
# Select'>, inputs=["<class 'tracr.rasp.rasp.SOp'> 4", "<class 'tracr.rasp.rasp.SOp'> 2", <Comparison.NEQ: '!='>], output="<class 'tracr.rasp.rasp.Selector'> 3", lambda_name=None),
# Aggregate'>, inputs=["<class 'tracr.rasp.rasp.Selector'> 3", "<class 'tracr.rasp.rasp.SOp'> 6"], output="<class 'tracr.rasp.rasp.SOp'> 7", lambda_name=None)]

# vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
# max_seq_len = 8

se1 = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y)
so2 = rasp.Aggregate(se1, rasp.indices)
se2 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x < y)
so3 = rasp.SelectorWidth(se2)
se3 = rasp.Select(so3, so2, lambda x, y: x!=y)
so6 = rasp.SequenceMap(lambda x,y: x-y, so3, so3)
so7 = rasp.Aggregate(se3, so6)

vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
max_seq_len = 8
program = so7

assembled_model, craft_model, rasp_model = compile_rasp_to_model(
    program, vocab, max_seq_len)

