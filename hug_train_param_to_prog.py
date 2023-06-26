
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
from data.plot_true_v_pred import plot_orginal_heatmaps
#from transformers import FlaxGPT2Model, 
from transformers import GPT2Config
from models import GPT2
from utils.jax_helpers import JaxMemUsage
JaxMemUsage.launch(interval=0.001)
from dill import dump, load
from jaxlib.xla_extension import XlaRuntimeError
# for storing the optimiser state
#from jax.experimental import optimizers

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
        exmp_input, examp_output, loss_mask, attention_mask = exmp_batch
        params = self.model.init({'params': init_rng, 'dropout': dropout_init_rng}, exmp_input, attention_mask=attention_mask, train=True)['params']
        
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
            inp_data, labels, loss_mask, attention_mask = batch
            #time_steps = inp_data.shape[1]
            time_steps = labels.shape[1]
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply({'params': params}, inp_data, attention_mask=attention_mask, train=train, rngs={'dropout': dropout_apply_rng})
            ptr = 0
            loss = 0
            for i, seg_size in enumerate(self.seg_sizes):
                loss += optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * loss_mask
                ptr += seg_size

            loss = loss.mean()
            acc = self.accuracy_fn(logits, labels, loss_mask)
            return loss, (acc, rng)
        return calculate_loss
    
    jnp.newaxis

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
            inp_data, labels, loss_mask, attention_mask = batch
            #rng, dropout_apply_rng = random.split(rng)

            # logits = (batch_size, time_steps, features)
            # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
            logits = self.model.apply({'params': state.params}, inp_data, attention_mask=attention_mask, train=False)#, rngs={'dropout': dropout_apply_rng})
            
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

from threading import Lock
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
        return x,y,loss_mask,attention_mask


dataset = WrappedDataset('.data/iTracr_dataset_train/', PROG_LEN, 40)
test_dataset = WrappedDataset('.data/iTracr_dataset_test/', PROG_LEN, 40)



collate_fn = partial(TorchParameterProgramDataset.collate_fn, PROG_LEN)


train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=2, shuffle=True)#, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, shuffle=True)#, pin_memory=True)
num_train_iters = len(train_dataloader) * max_epochs


def testing_loaders():
    it = iter(test_dataloader)
    x,y,_,_ = next(it)
    src_dataset.decode_pred(y, 0)

    it = iter(train_dataloader)
    x,y,_,_ = next(it)

    print(src_dataset.decode_pred(y, 0))


testing_loaders()


#%%


LEARNING_RATE=1e-4#1e-4
test_it = iter(test_dataloader)

# config_GPT2_medium = GPT2Config(vocab_size=next(test_it)[0].shape[2], n_positions=1024, n_embd=1024, n_layer=24, n_head=16, 
#                                 n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
#                                 initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
#                                 summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
#                                 scale_attn_weights = True, use_cache = True, bos_token_id = next(test_it)[0].shape[2], eos_token_id = next(test_it)[0].shape[2], 
#                                 scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )


# config_GPT2_medium = GPT2Config(vocab_size=next(test_it)[0].shape[2], n_positions=1024, n_embd=1024, n_layer=48, n_head=16, 
#                                 n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
#                                 initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
#                                 summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
#                                 scale_attn_weights = True, use_cache = True, bos_token_id = next(test_it)[0].shape[2], eos_token_id = next(test_it)[0].shape[2], 
#                                 scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )


config_GPT2_medium = GPT2Config(vocab_size=next(test_it)[0].shape[2], n_positions=1024, n_embd=1024, n_layer=36, n_head=16, 
                                n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
                                initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
                                summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
                                scale_attn_weights = True, use_cache = True, bos_token_id = next(test_it)[0].shape[2], eos_token_id = next(test_it)[0].shape[2], 
                                scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )

# n_positions should be max input length which is the maximum number of blocks which in our dataset is unbounded,
#  the max happens to be 184 so 512 should work
# config_GPT2_medium = GPT2Config(vocab_size=next(test_it)[0].shape[2], n_positions=512, n_embd=512, n_layer=12, n_head=8, 
#                                 n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
#                                 initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
#                                 summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
#                                 scale_attn_weights = True, use_cache = True, bos_token_id = next(test_it)[0].shape[2], eos_token_id = next(test_it)[0].shape[2], 
#                                 scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )


# 

model = GPT2(num_classes=sum(src_dataset.segment_sizes), gpt_config=config_GPT2_medium, input_dropout_prob=0.05) # if you forget input dense must match gpt hidden

#%%


trainer = TrainerModule(model, f'PARAM_v5_AttenMask_1_GPT2_36 LR {LEARNING_RATE} bs: {batch_size} nembed: {config_GPT2_medium.n_embd} n_layer: {config_GPT2_medium.n_layer} n_head: {config_GPT2_medium.n_head}',
                        #'no mean shuffled inputs pose in hid',#f'11 big lr: {LEARNING_RATE} bs: {batch_size} epcs: {max_epochs}', 
                        next(test_it), 
                        num_train_iters, 
                        dataset=src_dataset, 
                        lr=LEARNING_RATE)


# %%

#%%

#trainer.train_model(train_dataloader, test_dataloader, num_epochs=max_epochs)

#trainer.load_model(log_dir='PARAM_v3_1_GPT2')

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

