#%%
# srun -t 05:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --gres=gpu:1 --partition=ampere -A MLMI-WB326-SL2-GPU --pty bash
# conda activate venv
# source venv/bin/activate
# jupyter lab --no-browser --ip=* --port=8081

# module load cuda/11.8 cudnn/8.9_cuda-11.8
# module load cuda/12.1 cudnn/8.9_cuda-12.1

# ??
# module load rhel8/default-gpu
# module load rhel8/default-amp    

# srun --jobid $JOBID --pty bash
# squeue -u wb326
# showq -u wb326
# qstat -u wb326
# scancel <jobid>

# On local
# ssh -L 8081:gpu-e-14:8081 wb326@login-cpu.hpc.cam.ac.uk

# ImportError: libcupti.so.11.7: cannot open shared object file: No such file or directory
# export PATH=$PATH:/home/wb326/miniconda3/envs/venv/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wb326/miniconda3/envs/venv/lib



import os
import jax

# jax.distributed.initialize()
# print(f"connected to {jax.local_device_count()} compute devices")
#os.environ["CUDA_VISIBLE_DEVICES"]=""
#os.environ["XLA_FLAGS"]="--xla_dump_to=xla_dump.txt"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# from jax import config
# config.update("jax_disable_jit", True)

# from jax_smi import initialise_tracking
# initialise_tracking()

# pip install nvidia-cublas-cu11      nvidia-cublas-cu12      nvidia-cuda-cupti-cu11  nvidia-cuda-cupti-cu12  nvidia-cuda-nvcc-cu12   nvidia-cuda-nvrtc-cu11  nvidia-cuda-runtime-cu11nvidia-cuda-runtime-cu12nvidia-cudnn-cu11       nvidia-cudnn-cu12       nvidia-cufft-cu11       nvidia-cufft-cu12       nvidia-curand-cu11      nvidia-cusolver-cu11    nvidia-cusolver-cu12    nvidia-cusparse-cu11    nvidia-cusparse-cu12    nvidia-nccl-cu11        nvidia-nvjitlink-cu12   nvidia-nvtx-cu11        

from jax import random
import jax.numpy as jnp
import os
from torch.utils.tensorboard import SummaryWriter
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm
import numpy as np
import torch
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
from data.parameter_program_dataloader import TorchParameterProgramDataset
from data.plot_true_v_pred import plot_orginal_heatmaps, figure_to_array
from utils.jax_helpers import JaxMemUsage
JaxMemUsage.launch(interval=0.01)
from dill import dump, load
from jaxlib.xla_extension import XlaRuntimeError
from data.dataset import example_program_dataset, encode_rasp_program

from models import GPT2, GPT2Config





#%%
#import jax.profiler
CHECKPOINT_PATH = ".logs/"

class TrainerModule:

    def __init__(self, model, model_name, exmp_batch, max_iters, dataset, lr=1e-3, warmup=100, seed=42):
        """
        Inputs:
            model_name - Name of the model. Used for saving and checkpointing
            exmp_batch - Example batch to the model for initialization
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            seed - Seed to use for model init
        """
        super().__init__()
        self.model_name = model_name
        self.max_iters = max_iters
        self.lr = lr
        self.warmup = warmup
        self.seed = seed
        self.seg_sizes=src_dataset.segment_sizes
        self.dataset = dataset
        self.model = model
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.create_functions()
        self.init_model(exmp_batch)
        

    
    def init_model(self, exmp_batch):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        exmp_input, examp_output, loss_mask, attention_mask, pos_id = exmp_batch
        params = self.model.init({'params': init_rng, 'dropout': dropout_init_rng}, exmp_input, attention_mask=attention_mask, position_ids=pos_id, train=True)['params']
        
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=self.warmup,
            decay_steps=self.max_iters,
            end_value=0.0
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adam(lr_schedule)
            #optax.adamw(learning_rate=config.lr, weight_decay=config.weight_decay)
        )
        
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
    
    def raw_apply(self, encoded_model, encoded_ops):
        post_encoded_program = TorchParameterProgramDataset.encode_program(encoded_ops, src_dataset.op_encoder, src_dataset.var_encoder)
        x,y,loss_mask,attention_mask = TorchParameterProgramDataset.post_process_step(self.dataset.prog_len, x=np.array(encoded_model), y=post_encoded_program)
        x,y, loss_mask, attention_mask, pos_ids = collate_fn( data=[[x, y, loss_mask, attention_mask]])
        logits, fig = self.apply(x, attention_mask=attention_mask, pos_id=pos_ids, labels=y)
        return logits, fig


    def apply(self, inp_data, attention_mask, pos_id, labels=None, seed=0):
        rng = jax.random.PRNGKey(seed)
        rng, dropout_apply_rng = random.split(rng)
        logits = self.model.apply({'params': self.state.params}, inp_data, attention_mask=attention_mask, train=False, position_ids=pos_id, rngs={'dropout': dropout_apply_rng})
        
        def logit_classes_jnp(logits):
            classes = []
            logits = jnp.array(logits)
            
            ptr = 0
            for i, seg_size in enumerate(self.seg_sizes):
                classes.append(logits[:, :, ptr:ptr + seg_size].argmax(axis=2))
                ptr += seg_size
            classes = jnp.stack(classes, axis=2)
            return classes
        classes = logit_classes_jnp(logits)
        
        if labels is not None:
            max_prog_len = self.dataset.prog_len
            heat_img = plot_orginal_heatmaps(labels[:, -max_prog_len-2:, :], classes[:, -max_prog_len-2:, :], self.dataset, return_fig=True)
            return logits, heat_img
        else:
            return logits, None

    
    def get_accuracy_function(self):
        def accuracy(logits, labels, loss_mask):
            def logit_classes_jnp(logits):
                classes = []
                logits = jnp.array(logits)
                
                ptr = 0
                for i, seg_size in enumerate(self.seg_sizes):
                    classes.append(logits[:, :, ptr:ptr + seg_size].argmax(axis=2))
                    ptr += seg_size
                classes = jnp.stack(classes, axis=2)
                return classes
            classes = logit_classes_jnp(logits)
            
            time_steps = labels.shape[1]

            repeated_loss_mask = jnp.repeat(loss_mask[:, :, jnp.newaxis], classes.shape[2], axis=2)

            relevant_classes = classes[:, :time_steps, :] * repeated_loss_mask
            relevant_labels = labels[:, :time_steps, :] * repeated_loss_mask
            relevant_labels += 1 - repeated_loss_mask # ensure the masked out values are different
            acc = relevant_classes == relevant_labels
            acc = acc.sum() / (loss_mask.sum() * relevant_labels.shape[2])
            return acc
        return accuracy

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            # Input data has shape (batch_size, time_steps, features)
            # Labels has shape (batch_size, time_steps, 5)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            #time_steps = inp_data.shape[1]
            time_steps = labels.shape[1]
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply({'params': params}, inp_data, attention_mask=attention_mask, train=train, position_ids=pos_id, rngs={'dropout': dropout_apply_rng})
            ptr = 0
            loss = 0
            for i, seg_size in enumerate(self.seg_sizes):
                loss += optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * loss_mask
                ptr += seg_size

            loss = loss.mean()
            acc = self.accuracy_fn(logits, labels, loss_mask)
            return loss, (acc, rng)
        return calculate_loss
    



    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()



        # Training function
        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc
        self.train_step = jax.jit(train_step)




        # Evaluation function
        def eval_step(state, rng, batch):
            loss, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return loss, acc, rng
        self.eval_step = jax.jit(eval_step)


        def verbose_step(state, batch, step):
            # labels = (batch_size, max_time_steps, ordinal_features)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            #rng, dropout_apply_rng = random.split(rng)

            # logits = (batch_size, time_steps, features)
            logits = self.model.apply({'params': state.params}, inp_data, attention_mask=attention_mask, position_ids=pos_id, train=False)#, rngs={'dropout': dropout_apply_rng})
            
     
            ptr = 0
            loss = []
            for i, seg_size in enumerate(self.seg_sizes):
                loss.append(np.array(optax.softmax_cross_entropy_with_integer_labels(logits[:, :, ptr:ptr + seg_size], labels[:, :, i]) * loss_mask))
                ptr += seg_size

            # loss = (batch_size, time_steps, features)
            loss = np.stack(loss, axis=2)
            assert (loss.shape[0] == labels.shape[0]) and (loss.shape[2] == labels.shape[2])
           
            acc = self.accuracy_fn(logits, labels, loss_mask)

            def logit_classes_jnp(logits):
                classes = []
                logits = jnp.array(logits)
                
                ptr = 0
                for i, seg_size in enumerate(self.seg_sizes):
                    classes.append(logits[:, :, ptr:ptr + seg_size].argmax(axis=2))
                    ptr += seg_size
                classes = jnp.stack(classes, axis=2)
                return classes
            classes = logit_classes_jnp(logits)
            
            max_prog_len = self.dataset.prog_len
            heat_img = plot_orginal_heatmaps(labels[:, -max_prog_len-2:, :], classes[:, -max_prog_len-2:, :], self.dataset, loss=loss[:, -max_prog_len-2:, :])
            #heat_img = plot_orginal_heatmaps(labels, classes * jnp.expand_dims(loss_mask, axis=2).repeat(classes.shape[-1], axis=2), self.dataset, loss=loss)

            self.logger.add_image("verbose/heatmap", heat_img, global_step=step, dataformats='HWC')

            self.logger.add_histogram("verbose/output", np.array(logits), global_step=step)

            self.logger.add_scalar("verbose/acc", acc.item(), global_step=step)


        #self.verbose_step = jax.jit(verbose_step)
        self.verbose_step = verbose_step

        self.accuracy_fn = self.get_accuracy_function()


    # TODO optimise away item() to be minimised
    def train_epoch(self, train_loader, epoch, LOGS_PER_EPOCH=3, validation_loader=None, VALS_PER_EPOCH = 1):
        # Train model for one epoch, and log avg loss and accuracy
        DATALOADER_LENGTH = len(train_loader)
        LOGGING_INTERVAL = DATALOADER_LENGTH // LOGS_PER_EPOCH
        VALIDATION_INTERVAL = DATALOADER_LENGTH // VALS_PER_EPOCH
        best_eval_loss = np.inf
        with tqdm(total=len(train_loader), unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # ======================================= Training =======================================
            acc_sum, loss_sum, count = 0.0, 0.0, 0
            for idx, batch in enumerate(train_loader):
                try:
                    # -------------------------- Train ---------------------------------------------
                    self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
                    

                    # ----------- metrics -------------
                    loss, accuracy = loss.item(), accuracy.item()
                    loss_sum += loss
                    acc_sum += accuracy

                    
                    # ----------- TF metrics ----------
                    global_step = idx * batch[0].shape[0] + (epoch - 1) * DATALOADER_LENGTH * batch[0].shape[0]
                    self.logger.add_scalar('train_hf/loss', loss, global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy', accuracy, global_step=global_step)
                    
                    
                    # ------------ Low freq metrics --------------
                    if (idx + 1) % LOGGING_INTERVAL == 0:
                        self.verbose_step(state=self.state, batch=batch, step=global_step)
                    

                    # ------------ Evaluation Step ---------------
                    if validation_loader is not None and (idx + 1) % VALIDATION_INTERVAL == 0:
                        eval_acc, eval_loss = self.eval_model(validation_loader)
                        trainer.logger.add_scalar('val/accuracy', eval_acc, global_step=global_step)
                        trainer.logger.add_scalar('val/loss', eval_loss, global_step=global_step)
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            trainer.save_model(step=global_step)
                        self.eval_programs(step=global_step)
                        

                    # ----------- TQDM ----------------
                    tepoch.set_postfix({'Batch': idx, 'Train Loss': loss, 'Acc': accuracy, 'MaxMem': JaxMemUsage.max_usage_str, 'Mem': JaxMemUsage.usage_str})
                    tepoch.update(1)
                    
                    count += 1

                except XlaRuntimeError as E:
                    print(E)
                    print(batch[0].shape)
                    jax.lib.xla_bridge.get_backend().defragment()
                    if isinstance(E, KeyboardInterrupt):
                        raise(E)
            
            self.logger.add_scalar('train/loss', loss_sum / count, global_step=epoch)
            self.logger.add_scalar('train/accuracy', acc_sum / count, global_step=epoch)
            trainer.logger.flush()

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        acc_sum, loss_sum, count = 0.0, 0.0, 0
        for batch in data_loader:
            loss, acc, self.rng = self.eval_step(self.state, self.rng, batch)

            bs = batch[0].shape[0]
            loss, acc = loss.item(), acc.item()
            acc_sum += acc * bs
            loss_sum += loss * bs
            count += bs
        eval_acc = acc_sum / count
        eval_loss = loss_sum / count
        return eval_acc, eval_loss
    
    def eval_programs(self, step=0):
        for program_lam, lam_names, name in example_program_dataset:
            program = program_lam()
            encoded_model, encoded_ops = encode_rasp_program(program, args.PROG_LEN, lam_names)
            logits, fig = self.raw_apply(encoded_model, encoded_ops)
            img = figure_to_array(fig)
            self.logger.add_image("examples/"+name, img, global_step=step, dataformats='HWC')
    




    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)
        dump(self.state.opt_state, open(os.path.join(self.log_dir, "optimiser_state.pkl"), "wb"))
        

    def load_model(self, log_dir=None):
        log_dir = self.log_dir if log_dir is None else log_dir
        
        if not os.path.isdir(os.path.join(CHECKPOINT_PATH, log_dir)): raise FileNotFoundError("Could not find the model directory")

        params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, log_dir), target=self.state.params)
        opt_state = load( open(os.path.join(CHECKPOINT_PATH, log_dir, "optimiser_state.pkl"), "rb" ) )
        
        self.state = train_state.TrainState(
                            step=0,
                            apply_fn=self.model.apply,
                            params=params,
                            tx=self.state.tx,
                            opt_state=opt_state)




#%%

from argparse import Namespace

args = Namespace(
    batch_size=128,
    PROG_LEN = 15,
    max_epochs = 200,
    LEARNING_RATE=1e-4,
    input_dropout_prob = 0.05,
    max_timesteps = 40,
)

src_dataset = TorchParameterProgramDataset(args.PROG_LEN)
from data.dataloader_streams import ZipStreamReader

class WrappedDataset(ZipStreamReader):
    def __init__(self, dir: str, max_prog_len: int, max_time_step_reduction_sample: int) -> None:
        super().__init__(dir)
        self.max_prog_len = max_prog_len
        self.max_timesteps = max_time_step_reduction_sample
    
    def __getitem__(self, idx):
        # first rejection sample under the max timestep
        x_shape = self.max_timesteps + 1
        offset = 0
        while x_shape > self.max_timesteps:
            circular_index = (idx + offset) % self.__len__()
            x,y = super().__getitem__(circular_index)
            x,y,loss_mask,attention_mask = TorchParameterProgramDataset.post_process_step(self.max_prog_len, x=x, y=y)
            x_shape = x.shape[0]
            offset += 1
        return x,y,loss_mask,attention_mask


def make_collate_fn(PROG_LEN):
    ########################### Input Format ##########################################
    # | Timestep | Terminal Block Flag | Parameter Block | Architecture    |
    # | e.g. fst |          0          | <PARAMS>        | <ARCH_ENCODING> |
    # |      fst |          1          | <PARAMS>        | <ARCH_ENCODING> |
    # |     w_ov |          1          | <PARAMS>        | <ARCH_ENCODING> |
    # |  <PAD>   |          0          | 0               | 0               |
    # ...
    # | PROGRAM_START |  @ timestep T - PROG_LEN - 2
    # ... x PROG_LEN
    # | PROGRAM_END  |

    ########################## Output Format ##########################################
    # |       OP      | VAR_1 | VAR_2 | VAR_3 |  RET  |
    # | <PAD>         |
    # ... 
    # | PROGRAM_START | <PAD> | <PAD> | <PAD> | <PAD> |
    # | e.g. SELECT   |   V1  |    V1 | <PAD> |   V2  |
    # | ... |
    # | PROGRAM_END | <PAD> | <PAD> | <PAD> | <PAD> |
    # | <PAD> |
    # |  ...  |

    from torch.nn.utils.rnn import pad_sequence
    from data.parameter_encoder import ONEHOT_TIMESTEP_ENCODER
    


    INPUT_PROGRAM_FLAGS = torch.tensor(np.stack([ONEHOT_TIMESTEP_ENCODER[token] for token in ['PROGRAM_START'] + (['PAD'] * PROG_LEN) + ['PROGRAM_END']]))


    def collate_fn(data):
            inputs = [torch.tensor(d[0], device='cpu') for d in data]
            targets = [torch.tensor(d[1], device='cpu') for d in data]
            loss_masks = [torch.tensor(d[2], device='cpu') for d in data]
            attention_masks = [torch.tensor(d[3], device='cpu') for d in data]
            inputs = pad_sequence(inputs, batch_first=True)
            attention_masks = pad_sequence(attention_masks, batch_first=True)
            

            targets = torch.stack(targets)
            loss_masks = torch.stack(loss_masks)

            bs, parameter_timesteps, parameter_features = inputs.shape
            # Grow width to feature size
            program_flags =  torch.nn.ConstantPad2d((0, parameter_features - INPUT_PROGRAM_FLAGS.shape[1], 0, 0), 0)(INPUT_PROGRAM_FLAGS)
            program_flags = program_flags.unsqueeze(0).repeat(bs, 1, 1) # repeat across batches

            inputs = torch.concatenate([inputs, program_flags], axis=1)
            attention_masks = torch.concatenate([attention_masks, torch.ones(program_flags.shape[0:2])], axis=1)
            pos_ids = np.expand_dims(np.arange(1, inputs.shape[1]+1), axis=0).repeat(bs, 1)

            padding = torch.zeros(targets.shape[0], parameter_timesteps, targets.shape[2])
            targets = torch.concatenate((padding, targets), axis=1)
            loss_masks = torch.concatenate((padding[:, :, 0], loss_masks), axis=1)
            

            return np.array(inputs), np.array(targets).astype(int), np.array(loss_masks), np.array(attention_masks), pos_ids
    return collate_fn

#dataset = WrappedDataset('.data/iTracr_dataset_train/', args.PROG_LEN, args.max_timesteps)
dataset = WrappedDataset('.data/iTracrTrain.zip', args.PROG_LEN, args.max_timesteps)
test_dataset = WrappedDataset('.data/iTracrTest.zip', args.PROG_LEN, args.max_timesteps)


print(f"Dataset contains: {len(dataset)} samples" )

collate_fn = make_collate_fn(args.PROG_LEN)



train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=2, shuffle=True)#, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, shuffle=True)#, pin_memory=True)
num_train_iters = len(train_dataloader) * args.max_epochs


#%%

def testing_loaders():
    it = iter(test_dataloader)
    x,y,_,_, _ = next(it)
    src_dataset.decode_pred(y, 0)

    it = iter(train_dataloader)
    x,y,_,_, _ = next(it)

    # print(src_dataset.decode_pred(y, 0))


testing_loaders()


#%%



from data.parameter_encoder import ONEHOT_TIMESTEP_ENCODER
def decode_timesteps(x, batch=0):
    TIMESTEP_TOKEN_SIZE = list(ONEHOT_TIMESTEP_ENCODER.values())[0].shape[0]
    this_batch = x[batch, :, :]
    for timestep in range(this_batch.shape[0]):
        index = np.array(this_batch[timestep, : TIMESTEP_TOKEN_SIZE]).argmax()
        print(list(ONEHOT_TIMESTEP_ENCODER.keys())[index])

test_it = iter(test_dataloader)
def decode_test_sample():
    sample = next(test_it)
    decode_timesteps(sample[0], batch=1)
    print(src_dataset.decode_pred(sample[1], 0))
decode_test_sample()

#%%

x,y, loss_mask, attention_mask = next(iter(dataset))


import json
with open('utils/gpt2_configs/gpt2_large.json') as f: # GPT2 Large - 774M
  config_json = json.load(f)
model_config = GPT2Config(**config_json)


# GPT2 Medium - 335M
# model_config = GPT2Config(vocab_size=next(test_it)[0].shape[2], n_positions=1024, n_embd=1024, n_layer=24, n_head=16, 
#                                 n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
#                                 initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
#                                 summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
#                                 scale_attn_weights = True, use_cache = True, bos_token_id = next(test_it)[0].shape[2], eos_token_id = next(test_it)[0].shape[2], 
#                                 scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )




model = GPT2(num_classes=sum(src_dataset.segment_sizes), gpt_config=model_config, input_dropout_prob=args.input_dropout_prob)

#%%
trainer = TrainerModule(model, f'PARAM_GPT2_LARGE LR {args.LEARNING_RATE} bs: {args.batch_size} nembed: {model_config.n_embd} n_layer: {model_config.n_layer} n_head: {model_config.n_head}',
                        next(test_it), 
                        num_train_iters, 
                        dataset=src_dataset, 
                        lr=args.LEARNING_RATE)
_ = open(os.path.join(trainer.log_dir, "hyperparameters"), "w").write(f"{args}\n{model_config}")

#%%

# trainer.eval_programs()
#trainer.load_model(log_dir="PARAM_GPT2_MEDIUM_v2 LR 0.0001 bs: 128 nembed: 1024 n_layer: 24 n_head: 16")

#%%


for epoch_idx in range(1, args.max_epochs+1):
    trainer.train_epoch(train_dataloader, epoch=epoch_idx, validation_loader=test_dataloader, VALS_PER_EPOCH=10 )


#%%

