 
#%%
import os
import jax
#os.environ["CUDA_VISIBLE_DEVICES"]=""
#os.environ["XLA_FLAGS"]="--xla_dump_to=xla_dump.txt"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# from jax import config
# config.update("jax_disable_jit", True)

# from jax_smi import initialise_tracking
# initialise_tracking()

from jax import random
import jax.numpy as jnp
import os
from torch.utils.tensorboard import SummaryWriter
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm
import numpy as np
import torch
from functools import partial
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
from data.parameter_program_dataloader import TorchParameterProgramDataset
from data.plot_true_v_pred import plot_orginal_heatmaps, figure_to_array
from transformers.models.gptj.configuration_gptj import GPTJConfig
from utils.jax_helpers import JaxMemUsage
JaxMemUsage.launch(interval=0.01)
from dill import dump, load
from jaxlib.xla_extension import XlaRuntimeError


from models import GPTJ





#%%
#import jax.profiler
CHECKPOINT_PATH = ".logs/"

class TrainerModule:

    def __init__(self, model,  model_name, exmp_batch, max_iters, dataset, lr=1e-3, warmup=100, seed=42):
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
        # Create empty model. Note: no parameters yet
        self.model = model
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
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
        pos_ids = np.arange(1, x.shape[0]+1)
        x,y, loss_mask, attention_mask, pos_ids = TorchParameterProgramDataset.collate_fn_w_posid( PROG_LEN = self.dataset.prog_len, data=[[x, y, loss_mask, attention_mask, pos_ids]])
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

        time_steps = min(labels.shape[1], classes.shape[1])
        
        if labels is not None:
            heat_img = plot_orginal_heatmaps(labels[:, :time_steps, :], classes[:, :time_steps, :], self.dataset, return_fig=True)
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
            _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return acc, rng
        self.eval_step = jax.jit(eval_step)


        def verbose_step(state, batch, step):
            # labels = (batch_size, max_time_steps, ordinal_features)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            #rng, dropout_apply_rng = random.split(rng)

            # logits = (batch_size, time_steps, features)
            # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
            logits = self.model.apply({'params': state.params}, inp_data, attention_mask=attention_mask, position_ids=pos_id, train=False)#, rngs={'dropout': dropout_apply_rng})
            
            #time_steps = inp_data.shape[1]
            time_steps = labels.shape[1]
            ptr = 0
            loss = []
            for i, seg_size in enumerate(self.seg_sizes):
                loss.append(np.array(optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * loss_mask))
                ptr += seg_size

            # loss = (batch_size, time_steps, features)
            loss = np.stack(loss, axis=2)
            assert (loss.shape[0] == labels.shape[0]) and (loss.shape[2] == labels.shape[2])
           
            acc = self.accuracy_fn(logits, labels, loss_mask)

            def logit_classes_jnp(logits):
                classes = [] #jnp.zeros((logits.shape[0], 5))
                logits = jnp.array(logits)
                
                ptr = 0
                for i, seg_size in enumerate(self.seg_sizes):
                    #classes[:, i] = logits[:, ptr:ptr + seg_size].argmax(axis=1)
                    classes.append(logits[:, :, ptr:ptr + seg_size].argmax(axis=2))
                    ptr += seg_size
                classes = jnp.stack(classes, axis=2)
                return classes
            classes = logit_classes_jnp(logits)
            

            heat_img = plot_orginal_heatmaps(labels[:, :time_steps, :], classes[:, :time_steps, :], self.dataset, loss=loss)

            self.logger.add_image("verbose/heatmap", heat_img, global_step=step, dataformats='HWC')

            self.logger.add_histogram("verbose/output", np.array(logits), global_step=step)

            self.logger.add_scalar("verbose/acc", acc.item(), global_step=step)


        #self.verbose_step = jax.jit(verbose_step)
        self.verbose_step = verbose_step

        self.accuracy_fn = self.get_accuracy_function()


    # TODO optimise away item() to be minimised
    def train_epoch(self, train_loader, epoch, LOGS_PER_EPOCH=3):
        # Train model for one epoch, and log avg loss and accuracy
        dataloader_len = len(train_loader)
        LOGGING_INTERVAL = dataloader_len // LOGS_PER_EPOCH

        with tqdm(total=len(train_loader), unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # ======================================= Training =======================================
            accs, losses = [], []
            # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
            for idx, batch in enumerate(train_loader):
                try:
                    #print(batch[0].shape[1])
                    # -------------------------- Train ---------------------------------------------
                    self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
                    

                    # ----------- metrics -------------
                    losses.append(loss)
                    accs.append(accuracy)

                    
                    # ----------- TF metrics ----------
                    self.logger.add_scalar('train_hf/loss', loss.item(), global_step=idx + (epoch - 1) * dataloader_len)
                    self.logger.add_scalar('train_hf/accuracy', accuracy.item(), global_step=idx + (epoch - 1) * dataloader_len)
                    
                    
                    # ------------ Low freq metrics --------------
                    if (idx + 1) % LOGGING_INTERVAL == 0:
                        self.verbose_step(state=self.state, batch=batch, step=idx + epoch * dataloader_len)

                    
                    
                    # ----------- TQDM ----------------
                    tepoch.set_postfix({'Batch': idx, 'Train Loss': loss.item(), 'Acc': accuracy.item(), 'MaxMem': JaxMemUsage.max_usage_str, 'Mem': JaxMemUsage.usage_str})
                    tepoch.update(1)
                except XlaRuntimeError as E:
                    print(E)
                    print(batch[0].shape)
                    jax.lib.xla_bridge.get_backend().defragment()
                    if isinstance(E, KeyboardInterrupt):
                        raise(E)
            # jax.profiler.save_device_memory_profile("memory.prof")

            avg_loss = np.stack(jax.device_get(losses)).mean()
            avg_acc = np.stack(jax.device_get(accs)).mean()
            self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
            self.logger.add_scalar('train/accuracy', avg_acc, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        correct_class, count = 0, 0
        for batch in data_loader:
            acc, self.rng = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc
    
    #@profile
    def train_model(self, train_loader, val_loader, num_epochs=500):
        # Train model for defined number of epochs
        best_acc = 0.0
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar('val/accuracy', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()
                

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)
        # trained_params = jax.example_libraries.optimizers.unpack_optimizer_state(self.state.opt_state)
        # pickle.dump(trained_params, open(os.path.join(self.log_dir, "best_ckpt.pkl"), "wb"))
        dump(self.state.opt_state, open(os.path.join(self.log_dir, "optimiser_state.pkl"), "wb"))
        

    def load_model(self, pretrained=False, log_dir=None):
        # Load model. We use different checkpoint for the pretrained model
        log_dir = self.log_dir if log_dir is None else log_dir
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=self.state.params)

        opt_state = load( open(os.path.join(CHECKPOINT_PATH, log_dir, "optimiser_state.pkl"), "rb" ) )
        # self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx, opt_state=opt_state)
        train_state.TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx,
            opt_state=opt_state)
        
        # best_params = pickle.load(open(os.path.join(self.log_dir, "best_ckpt.pkl"), "rb"))
        # self.state.opt_state = jax.example_libraries.optimizers.pack_optimizer_state(best_params)
        #self.state.opt_state = load( open(os.path.join(CHECKPOINT_PATH, log_dir, "optimiser_state.pkl"), "rb" ) )


    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))



#%%


batch_size=128
PROG_LEN = 15
max_epochs = 200

src_dataset = TorchParameterProgramDataset(PROG_LEN)

from data.dataloader_streams import StreamReader


class WrappedDataset(StreamReader):
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
        pos_ids = np.arange(1, x.shape[0]+1)
        return x,y,loss_mask,attention_mask, pos_ids


dataset = WrappedDataset('.data/iTracr_dataset_train/', PROG_LEN, 40)
test_dataset = WrappedDataset('.data/iTracr_dataset_test/', PROG_LEN, 40)



collate_fn = partial(TorchParameterProgramDataset.collate_fn_w_posid, PROG_LEN)

#%%


train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=2, shuffle=True)#, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, shuffle=True)#, pin_memory=True)
num_train_iters = len(train_dataloader) * max_epochs


def testing_loaders():
    it = iter(test_dataloader)
    x,y,_,_, _ = next(it)
    src_dataset.decode_pred(y, 0)

    it = iter(train_dataloader)
    x,y,_,_, _ = next(it)

    print(src_dataset.decode_pred(y, 0))


testing_loaders()


#%%


LEARNING_RATE=1e-4#1e-4
test_it = iter(test_dataloader)



config_gptj = GPTJConfig(
        vocab_size=None,
        n_positions=1024,
        n_embd=1024,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False
)
    


# 

model = GPTJ(num_classes=sum(src_dataset.segment_sizes), gpt_config=config_gptj, input_dropout_prob=0.05) # if you forget input dense must match gpt hidden

#%%
trainer = TrainerModule(model, f'PARAM_GPTJ_v1 temp 2 LR {LEARNING_RATE} bs: {batch_size} nembed: {config_gptj.n_embd} n_layer: {config_gptj.n_layer} n_head: {config_gptj.n_head}',
                        #'no mean shuffled inputs pose in hid',#f'11 big lr: {LEARNING_RATE} bs: {batch_size} epcs: {max_epochs}', 
                        next(test_it), 
                        num_train_iters, 
                        dataset=src_dataset, 
                        lr=LEARNING_RATE, PROG_LEN=PROG_LEN)


#%%

#trainer.train_model(train_dataloader, test_dataloader, num_epochs=max_epochs)

trainer.load_model(log_dir=f'PARAM_GPTJ_v1 LR {LEARNING_RATE} bs: {batch_size} nembed: {config_gptj.n_embd} n_layer: {config_gptj.n_layer} n_head: {config_gptj.n_head}')

#%%

best_acc = 0.0
for epoch_idx in range(1, max_epochs+1):
    trainer.train_epoch(train_dataloader, epoch=epoch_idx)
    if epoch_idx % 5 == 0:
        eval_acc = trainer.eval_model(test_dataloader)
        trainer.logger.add_scalar('val/accuracy', eval_acc, global_step=epoch_idx)
        if eval_acc >= best_acc:
            best_acc = eval_acc
            trainer.save_model(step=epoch_idx)
        trainer.logger.flush()

#%%

import sys
sys.path.append('tracr/')
import tracr.compiler.lib as lib
from data.dataset import encode_rasp_program
from tracr.rasp import rasp

# program = lib.make_length()#lib.make_hist()#lib.make_frac_prevs((rasp.tokens == "x"))#lib.make_frac_prevs((rasp.tokens == "x"))#lib.make_hist()#lib.make_length()


# encoded_model, encoded_ops = encode_rasp_program(program, PROG_LEN, [])



example_program_dataset = [
    # Program generating lambda,      lambda names
    (lambda: lib.make_length(), [], "length"),
    (lambda: lib.make_hist(), [], "histogram"), 
    (lambda: lib.make_frac_prevs((rasp.tokens == "x")), ['LAM_EQ'], "frac_prev"),
    
    # Requires Linear Sequence Map
    # (lambda: lib.make_shuffle_dyck(pairs=["()", "{}"]), ['LAM_LT'], "2_shuffle_dyck"),
    # (lambda: lib.make_shuffle_dyck(pairs=["()", "{}", "[]"]), ['LAM_LT'], "3_shuffle_dyck"),
    
    # Expects numeric inputs
    # (lambda: lib.make_sort(
    #         rasp.tokens, rasp.tokens, max_seq_len=4, min_key=1), ['LAM_MUL'], 'sort_4'),
    # (lambda: lib.make_sort(
    #         rasp.tokens, rasp.tokens, max_seq_len=8, min_key=1), ['LAM_MUL'], 'sort_8'),

    (lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens), [], 'sort_unique'),
    
    # ???
    # (lambda: lib.make_sort_freq(max_seq_len=3), ['LAM_MUL', 'LAM_MUL'], 'sort_freq 3'),
    # (lambda: lib.make_sort_freq(max_seq_len=7), ['LAM_MUL', 'LAM_MUL'], 'sort_freq 7'),

    # Requires Linear Sequence Map
    # (lambda: lib.make_pair_balance(
    #         sop=rasp.tokens, open_token="(", close_token=")"), [], 'pair_balance'),
    (lambda: rasp.Map(lambda x: x == "t4", rasp.tokens), ['LAM_GT'], 'map_eq_t4'),
    #(lambda: rasp.Map(lambda x: x > 2, rasp.tokens), ['LAM_GT'], 'map_gt_2'),
    (lambda: rasp.Map(lambda x: x > 1, rasp.Map(lambda x: x == "t1", rasp.tokens)), ['LAM_EQ', 'LAM_GT'], 'map_map')        
]

for program_lam, lam_names, name in example_program_dataset:
    print(name)
    program = program_lam()
    encoded_model, encoded_ops = encode_rasp_program(program, PROG_LEN, lam_names)
    logits, fig = trainer.raw_apply(encoded_model, encoded_ops)
    img = figure_to_array(fig)
    trainer.logger.add_image("examples/"+name, img, global_step=0, dataformats='HWC')





#%%


# %%
