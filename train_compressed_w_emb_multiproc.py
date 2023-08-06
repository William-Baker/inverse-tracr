# 
import sys
sys.path.append('tracr/')
from flax.training import train_state
from utils.compile_with_compressed import compile_with_compressed, COMPILER_BOS
from utils.plot import *
import jax, optax, torch, os
import jax.numpy as jnp
from tqdm import tqdm
from itertools import product
from argparse import Namespace
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
from datetime import datetime
from utils.time_sensitive import time_sensitive
from shutil import move as move_dir
from utils.export_compressed_params import export_params
jax.config.update('jax_platform_name', 'cpu')
#jax.config.update("jax_debug_nans", True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# 
process_args = Namespace(
    run_id =   str(datetime.now().strftime("%m-%d %H.%M.%S.%f"))
)

#
from random import choice
args = Namespace(
    factor=0.10,
    compression = 2.0,
    idty = False, # True, # Whether to use a noisy identity to initialise the embedding
    LR = 5e-3, # 4e-3, # 5e-2 worked so far but some nans
    EPOCHS = 20,
    trn_all = False, # True,
    loss = 'L2', #'L2', #  'L2', 'L1', 'SoftMax'
    batch_size = 512,
    vocab_batch_size=64,
    mult = False, #True, #True,
    sched = 'cosine',
    #mpow = 2,
    div=20,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join("Compressed Tracr All" if args.trn_all == True else "Compressed Tracr emb_W", 
                            process_args.run_id)# + f"{args.LR}")
    logger = SummaryWriter(log_dir=log_dir)


    jax.config.update('jax_default_matmul_precision', 'float32') # 'bfloat16'

    def failure(id: int):
        logger.add_scalar("progress",  0, 4)
        logger.add_scalar("fail", id, id)
        logger.close()
        move_dir(log_dir, log_dir.replace('/', ' fail/'))
        sys.exit(1)

    def finish():
        logger.add_scalar("progress",  1,10)
        logger.close()
        move_dir(log_dir, log_dir.replace('/', ' success/'))
        sys.exit(0)


    #  =================== init program and compile transformer programs ===========================
    program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_ops, ops_range = [None]*7

    from data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
    ops_range=(10, 15)
    numeric_range=(5, 8)
    vocab_size_range=(5, 8)
    numeric_inputs_possible=True
    max_seq_len = np.random.randint(4, 9)
    CRAFT_TIMEOUT = 2
    def timed_func():
        assembled_model, compressed_assembled_model, actual_ops = None, None, None
        
        n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
        print(n_ops, vocab, TARGET_PROGRAM_LENGTH)
        try:
            program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
        except np.core._exceptions._ArrayMemoryError as E:
            print("mem alloc err")
            return None
        try:
            assembled_model, compressed_assembled_model, craft_model, rasp_model = compile_with_compressed(
                program, vocab, max_seq_len, compression=args.compression,
                CRAFT_TIMEOUT=CRAFT_TIMEOUT)
        except ValueError as E:
            print("val err")
            return None
        except KeyError as E:
            print("key err")
            return None
        except TimeoutError:
            print("craft timeout")
            return None

        return assembled_model, compressed_assembled_model, actual_ops, vocab, program






    ret = None
    for i in range(20):
        ret = time_sensitive(timed_func, 10)
        if ret is not None:
            break
    if ret is None:
        failure(1)

    assembled_model, compressed_assembled_model, actual_ops, vocab, program = ret
    print(len(actual_ops))
    logger.add_scalar("prog len", len(actual_ops), 1)
    logger.add_scalar("progress", 0.1, 1)


    if args.idty: # init embedding to be noisy identiy?
        compressed_assembled_model.params['compressed_transformer']['w_emb'] = jnp.eye(*compressed_assembled_model.params['compressed_transformer']['w_emb'].shape)
        compressed_assembled_model.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), compressed_assembled_model.params['compressed_transformer']['w_emb'].shape) / 10
    else:
        compressed_assembled_model.params['compressed_transformer']['w_emb'] /= args.div

    def init_all_params(params):
        rng = jax.random.PRNGKey(0)
        initializer = jax.nn.initializers.glorot_uniform()
        for key, val in params.items():
            for comp, weight in val.items():
                if 'compressed_transformer' in key + comp:
                    rng, nrng = jax.random.split(rng, 2)
                    if len(params[key][comp].shape) > 1:
                        params[key][comp] =initializer(nrng, params[key][comp].shape, jnp.float32)
                    else:
                        params[key][comp] = jax.random.normal(nrng, params[key][comp].shape) / 1000
        return params

    if args.trn_all:
        compressed_assembled_model.params = init_all_params(compressed_assembled_model.params)

    else:
        # we should only be initialising w_emb if we're not training all
        for key, val in compressed_assembled_model.params.items():
            for comp, weight in val.items():
                if 'compressed_transformer' in key + comp:
                    if comp != 'w_emb':
                        assert (weight == assembled_model.params[key.replace('compressed_transformer', 'transformer')][comp]).all()




    # ======================== Dataloader ======================================


    class VocabDataset:
        def __init__(self, vocab, max_seq_len, encoder_fn, length=25000) -> None:
            self.vocab = vocab
            self.inputs = list(product(*[vocab]*(max_seq_len-1)))
            self.encoder_fn = encoder_fn
            self.length = length
        def __len__(self):
            #return len(self.inputs)
            return self.length
        def __getitem__(self, idx):
            formatted_input = [COMPILER_BOS] + list(self.inputs[idx%len(self.inputs)])
            encoded_tokens =  self.encoder_fn(formatted_input)
            return formatted_input, np.array(encoded_tokens)
        def collate_fn(data):
                formatted_input = [d[0] for d in data]
                encoded_tokens = [d[1] for d in data]
                encoded_tokens = np.stack(encoded_tokens, axis=0).squeeze()
                return formatted_input, encoded_tokens
                
    class RandomDataset:
        def __init__(self, max_seq_len, residual_size, length=25000) -> None:
            self.max_seq_len = max_seq_len
            self.residual_size = residual_size
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            rnd = np.random.rand(self.max_seq_len, self.residual_size)
            return None, rnd
        def collate_fn(data):
            formatted_input = [d[0] for d in data]
            encoded_tokens = [d[1] for d in data]
            encoded_tokens = np.stack(encoded_tokens, axis=0)
            return formatted_input, np.array(encoded_tokens)

                

    def make_teacher_call(teacher, teacher_forward):
        def fun(encoded_tokens):
            output = teacher_forward(teacher.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
            target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
            # decoded = np.array(teacher.decode_all_outputs(output))
            target_ids = jnp.argmax(teacher.residual_to_logits(output), axis=-1)
            return target_outs, target_ids
        return fun

    def make_validation_teacher_call(teacher, teacher_forward):
        def fun(encoded_tokens):
            output = jax.jit(teacher_forward)(teacher.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
            target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
            decoded = np.array(teacher.decode_all_outputs(output))
            target_ids = jnp.argmax(teacher.residual_to_logits(output), axis=-1)
            return target_outs, target_ids, decoded
        return fun


    train_vocab_dataloader = DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=args.vocab_batch_size, collate_fn=VocabDataset.collate_fn, shuffle=True, num_workers=1, prefetch_factor=2)
    #train_teacher_call = make_teacher_call(assembled_model, assembled_model.forward)

    train_random_dataloader = DataLoader(RandomDataset(max_seq_len, len(assembled_model.residual_labels)), batch_size=args.batch_size, collate_fn=RandomDataset.collate_fn, num_workers=1, prefetch_factor=2)
    train_teacher_call = make_teacher_call(assembled_model, assembled_model.forward_no_emb)


    validation_teacher_call = make_validation_teacher_call(assembled_model, assembled_model.forward)

    logger.add_scalar("progress", 0.2, 2)

    # ==================== Schedulers ==========================================


    class CustomSchedule:
        def __init__(self, LR) -> None:
            self.history = []
            self.LR = LR
            self.initial_LR = LR
        def log(self, loss):
            self.history.append(loss)
            if len(self.history) > 600:
                history = self.history
                avg_loss = np.mean(self.history[-100:])
                # less than 2% change in 2 epochs per epoch
                # if avg_loss < 1e-4:
                #     self.LR = self.initial_LR / 600*5
                # elif avg_loss < 5e-4:
                #     self.LR = self.initial_LR / 600
                # elif avg_loss < 1e-3:
                #     self.LR = self.initial_LR / 125
                # elif avg_loss < 5e-2:
                #     self.LR = self.initial_LR / 25
                # elif avg_loss < 1e-1:
                #     self.LR = self.initial_LR / 5
                # else:
                #     self.LR = self.initial_LR
                self.LR = self.initial_LR * min(0.01 * (np.exp(30 * avg_loss) - 1), 1)


    cs = CustomSchedule(args.LR)




    # cosine anealing + warmup scheduler
    def create_learning_rate_fn(warmup_epochs, num_epochs, base_learning_rate, steps_per_epoch):
        """Creates learning rate schedule."""
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=base_learning_rate,
            transition_steps=warmup_epochs * steps_per_epoch)
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * steps_per_epoch])
        return schedule_fn




    LR_fn = None
    if args.sched == 'cosine': # cosine anealing scheduler
        LR_fn = create_learning_rate_fn(1, args.EPOCHS, args.LR, len(train_random_dataloader))    
    elif args.sched == 'custom': # custom scheduler
        #  ensure you uncomment the line in the train loop to use
        LR_fn = lambda x: cs.LR

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.01),  # Clip gradients at norm 1
        #optax.clip(1e-3),
        #optax.adamw(args.LR, weight_decay=0.0001)
        #optax.sgd(learning_rate=args.LR)
        #optax.sgd(LR_fn)
        optax.adamw(LR_fn, weight_decay=0.0001) ,
        
    )


    # ================= setup frozen grads ==========================================

    if not args.trn_all:
        # helpers for zero grads on all parameters other than compressed_transformer/w_emb
        from flax.core.frozen_dict import unfreeze
        from utils.jax_helpers import zero_grads, create_mask

        optimizer = optax.multi_transform({'adam': optimizer, 'zero': zero_grads()},
                                create_mask(compressed_assembled_model.params, lambda s: s != 'compressed_transformer'))
        compressed_assembled_model.params = unfreeze(compressed_assembled_model.params)




    # jax.tree_map(lambda x: jnp.isnan(x).any(), compressed_assembled_model.params)
    # jax.tree_map(lambda x: jnp.isnan(x).any(), assembled_model.params)

    # ============== init train state ===============================


    forward_fn = compressed_assembled_model.forward_no_emb

    # Initialize training state
    state = train_state.TrainState.create(
        apply_fn=forward_fn, params=compressed_assembled_model.params, tx=optimizer)



    def calculate_loss(params, batch):
        encoded_tokens, targets, target_ids, encoded_vocab, vocab_target_ids = batch
        output = state.apply_fn(params, encoded_tokens)
        compressed_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
        vocab_output = state.apply_fn(params, encoded_vocab)

        #loss = 0.0

        loss = jnp.mean((targets - compressed_outs)** 2) 


        # Additional logit error term
        logits = compressed_assembled_model.residual_to_logits(vocab_output)
        loss += optax.softmax_cross_entropy_with_integer_labels(logits, vocab_target_ids).mean() * args.factor
        
        return loss

    def train_step(state, encoded_tokens, encoded_vocab):
        extra_encoded_tokens = assembled_model.forward_emb(assembled_model.params, encoded_vocab)
        #encoded_tokens = jnp.concatenate([encoded_tokens, extra_enoded_tokens], axis=0)
        target_outs, target_ids =  train_teacher_call(encoded_tokens) 
        _, vocab_target_ids =  train_teacher_call(extra_encoded_tokens) 
        batch = (encoded_tokens, target_outs, target_ids, extra_encoded_tokens, vocab_target_ids)
        loss_fn = lambda params: calculate_loss(params, batch)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    train_step = jax.jit(train_step)


    def measure_accuracy():
        VAL_SAMPLES = 30
        avg_acc = 0.0
        it = iter(train_vocab_dataloader)
        for idx in range(VAL_SAMPLES):        
            batch = next(it)
            (formatted_input, encoded_tokens) = batch
            target_outs, target_ids, decoded = validation_teacher_call(encoded_tokens)
            output = jax.jit(compressed_assembled_model.forward)(state.params, encoded_tokens)
            pred_decoded = compressed_assembled_model.decode_all_outputs(output)
            acc = np.equal(pred_decoded , decoded).mean()
            avg_acc += acc
            
        avg_acc /= VAL_SAMPLES
        logger.add_scalar("acc", avg_acc, 1)
        return avg_acc



    # ======================= Train loop =====================================

    logger.add_scalar("progress", 0.3, 3)



    avg_loss = 0.0
    avg_acc = 0.0
    global_idx = 0
    for epoch in range(args.EPOCHS):
        with tqdm(total=len(train_random_dataloader), unit='batch') as tepoch:
            total_loss = 0.0
            vocab_iter = iter(train_vocab_dataloader)
            for idx, batch in enumerate(train_random_dataloader):

                (formatted_vocab, encoded_vocab) = next(vocab_iter)
                (formatted_input, encoded_tokens)  = batch 


                state, loss = train_step(state, encoded_tokens, encoded_vocab)
                
                
                tepoch.set_postfix({'Batch': idx, 'Train Loss': loss})

                tepoch.update(1)
                total_loss += loss
                global_idx += 1
                # wandb.log({"loss": loss.item()}) # if more than 10 processes, exceeds rate limits
                # if (global_idx % 50) == 0:
                logger.add_scalar("loss", loss.item(), global_idx)

                if np.isnan( loss.item() ):
                    failure(2)

            
            avg_loss = total_loss / len(train_random_dataloader)
            tepoch.set_postfix({'Batch': idx, 'Avg Loss': avg_loss})
            logger.add_scalar("avg loss", avg_loss.item(), epoch)

            if avg_loss > 4: # probrably wont converge if we haven't already
                    failure(3)
                
            if (args.EPOCHS - epoch ) in [1, 2,3,4]: # export the 4th to last to the 2nd to last
                if (args.EPOCHS - epoch) == 4:
                    avg_acc = measure_accuracy()
                    if avg_acc < 0.9:
                        failure(5)
                if (args.EPOCHS - epoch) == 1:
                    if avg_loss > 0.05:
                        failure(4)
                    export_params(state.params, max(ops_range), actual_ops, args.trn_all, process_args.run_id)
                else:
                    if avg_loss < 0.05:
                        export_params(state.params, max(ops_range), actual_ops, args.trn_all, process_args.run_id + f" {epoch}")
                logger.add_scalar("progress", epoch/args.EPOCHS, (epoch/args.EPOCHS)*10)

    

    

    logger.add_scalar("progress", 0.4, 4)


        
    # VAL_SAMPLES = 30
    # avg_acc = 0.0
    # with tqdm(total=VAL_SAMPLES, unit='batch') as tepoch:
    #     it = iter(train_vocab_dataloader)
    #     for idx in range(VAL_SAMPLES):        
    #         batch = next(it)
    #         (formatted_input, encoded_tokens) = batch
    #         target_outs, target_ids, decoded = validation_teacher_call(encoded_tokens)
    #         output = jax.jit(compressed_assembled_model.forward)(state.params, encoded_tokens)
    #         pred_decoded = compressed_assembled_model.decode_all_outputs(output)
    #         acc = np.equal(pred_decoded , decoded).mean()
    #         avg_acc += acc
    #         tepoch.set_postfix({'Batch': idx, 'Acc': acc})
    #         tepoch.update(1)
    #     avg_acc /= VAL_SAMPLES
    #     tepoch.set_postfix({'Avg Acc': avg_acc})
    #     logger.add_scalar("acc", avg_acc, 1)

    fig = show_emb(state.params, show=False)
    logger.add_figure('emb', fig, global_step=global_idx)

    logger.add_hparams(vars(process_args) | vars(args), {'loss': avg_loss.item(), 'acc': avg_acc} )


    with open(log_dir.split('/')[0] + "/stats.csv",'a+') as f:
            f.write(f"{process_args.run_id},{avg_acc},{avg_loss},{args.EPOCHS},{args.LR},{args.batch_size}\n")


    logger.add_scalar("progress", 1, 10)




    
    finish()
except Exception as E:
    from os import makedirs
    makedirs('logs', exist_ok=True)
    with open(f'logs/{process_args.run_id}.txt','w') as f:
        import traceback
        f.write(str(E))
        tb = traceback.format_exc()
        f.write(str(tb))




