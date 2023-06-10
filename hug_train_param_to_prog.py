
#%%
from jax_smi import initialise_tracking
initialise_tracking()

from jax import random
import jax.numpy as jnp
import os
from torch.utils.tensorboard import SummaryWriter
import jax
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
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2BlockCollection
from flax import linen as nn
from models import PositionalEncoding
from jax import jit
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



    
    def get_accuracy_function(self):
        def accuracy(logits, labels):
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
          
          #time_steps = logits.shape[1]
          time_steps = labels.shape[1]
          acc = (classes[:, :time_steps, :] == labels[:, :time_steps, :]).mean()
          return acc
        return accuracy

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            # Input data has shape (batch_size, time_steps, features)
            # Labels has shape (batch_size, time_steps, 5)
            inp_data, labels, mask = batch
            #time_steps = inp_data.shape[1]
            time_steps = labels.shape[1]
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply({'params': params}, inp_data, train=train, rngs={'dropout': dropout_apply_rng})
            # print(inp_data.shape[1])
            # print(labels.shape[1])
            # print("")
            ptr = 0
            loss = 0
            for i, seg_size in enumerate(self.seg_sizes):
                loss += optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * mask
                ptr += seg_size

            loss = loss.mean()
            acc = self.accuracy_fn(logits, labels)
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
            inp_data, labels, mask = batch
            #rng, dropout_apply_rng = random.split(rng)

            # logits = (batch_size, time_steps, features)
            # with jax.profiler.trace("jax-trace", create_perfetto_link=True):
            logits = self.model.apply({'params': state.params}, inp_data, train=False)#, rngs={'dropout': dropout_apply_rng})
            
            #time_steps = inp_data.shape[1]
            time_steps = labels.shape[1]
            ptr = 0
            loss = []
            for i, seg_size in enumerate(self.seg_sizes):
                loss.append(np.array(optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * mask))
                ptr += seg_size

            # loss = (batch_size, time_steps, features)
            loss = np.stack(loss, axis=2)
            assert (loss.shape[0] == labels.shape[0]) and (loss.shape[2] == labels.shape[2])
           
            acc = self.accuracy_fn(logits, labels)

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

    def init_model(self, exmp_batch):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        exmp_input, examp_output, mask = exmp_batch
        params = self.model.init({'params': init_rng, 'dropout': dropout_init_rng}, exmp_input, train=True)['params']
        
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

    

    # TODO optimise away item() to be minimised
    def train_epoch(self, train_loader, epoch, LOGS_PER_EPOCH=2):
        # Train model for one epoch, and log avg loss and accuracy
        dataloader_len = len(train_loader)
        LOGGING_INTERVAL = dataloader_len // LOGS_PER_EPOCH

        with tqdm(total=len(train_loader), unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # ======================================= Training =======================================
            accs, losses = [], []
            for idx, batch in enumerate(train_loader):
                
                # -------------------------- Train ---------------------------------------------
                self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)

                # ----------- metrics -------------
                losses.append(loss)
                accs.append(accuracy)

                
                # ----------- TF metrics ----------
                self.logger.add_scalar('train_hf/loss', loss.item(), global_step=idx + (epoch - 1) * dataloader_len)
                self.logger.add_scalar('train_hf/accuracy', accuracy.item(), global_step=idx + (epoch - 1) * dataloader_len)
                
                
                # ------------ Low freq metrics --------------
                if (idx + 1) % LOGGING_INTERVAL == 10:
                    self.verbose_step(state=self.state, batch=batch, step=idx + epoch * dataloader_len)

                
                
                # ----------- TQDM ----------------
                tepoch.set_postfix({'Batch': idx, 'Train Loss': loss.item(), 'Acc': accuracy.item()})
                tepoch.update(1)

                

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

    def load_model(self, pretrained=False, log_dir=None):
        # Load model. We use different checkpoint for the pretrained model
        log_dir = self.log_dir if log_dir is None else log_dir
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))



#%%


batch_size=32
PROG_LEN = 15
max_epochs = 200

src_dataset = TorchParameterProgramDataset(PROG_LEN)

from data.dataloader_streams import StreamReader


class WrappedDataset(StreamReader):
    def __init__(self, dir: str, max_prog_len: int) -> None:
        super().__init__(dir)
        self.max_prog_len = max_prog_len
    
    def __getitem__(self, idx):
        x,y = super().__getitem__(idx)
        x,y,mask = TorchParameterProgramDataset.post_process_step(self.max_prog_len, x=x, y=y)
        return x,y,mask


dataset = WrappedDataset('.data/iTracr_dataset/', PROG_LEN)



collate_fn = partial(TorchParameterProgramDataset.collate_fn, PROG_LEN)


train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=3, shuffle=True)#, pin_memory=True)
num_train_iters = len(train_dataloader) * max_epochs

it = iter(train_dataloader)
x,y,mask = next(it)

print(src_dataset.decode_pred(y, 0))




#%%


# PARAMS_2 fine

LEARNING_RATE=1e-4



#%%
from transformers import GPT2Config



# config_GPT2_medium = GPT2Config(vocab_size=x.shape[2], n_positions=1024, n_embd=1024, n_layer=24, n_head=16, 
#                                 n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
#                                 initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
#                                 summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
#                                 scale_attn_weights = True, use_cache = True, bos_token_id = x.shape[2], eos_token_id = x.shape[2], 
#                                 scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )

config_GPT2_medium = GPT2Config(vocab_size=x.shape[2], n_positions=1024, n_embd=1024, n_layer=12, n_head=8, 
                                n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, layer_norm_epsilon=1e-05,
                                initializer_range=0.02, summary_type='cls_index', summary_use_proj = True, 
                                summary_activation = None, summary_proj_to_labels = True, summary_first_dropout = 0.1, 
                                scale_attn_weights = True, use_cache = True, bos_token_id = x.shape[2], eos_token_id = x.shape[2], 
                                scale_attn_by_inverse_layer_idx = False, reorder_and_upcast_attn = False )

#%%




class GPT_Decoder(nn.Module):
    num_classes: int
    gpt_config: GPT2Config
    input_dropout_prob: float = 0.0
    input_dense: int =512
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.input_dense)
        self.input_pos_encoder = PositionalEncoding(self.input_dense)
        self.h = FlaxGPT2BlockCollection(self.gpt_config)
        
        self.output_net = [
            nn.Dense(1024),
            nn.LayerNorm(),
            nn.relu,
            #nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]
    
    def __call__(self, x, mask=None, train=True):
        x = self.input_dropout(x, deterministic=not train)
        i = self.input_layer(x)
        i = self.input_pos_encoder(i)
        hidden_states, all_hidden_states, all_attentions, all_cross_attentions = self.h(i)
        o = hidden_states
        for l in self.output_net:
            o = l(o) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)
        return o

model = GPT_Decoder(num_classes=sum(src_dataset.segment_sizes), gpt_config=config_GPT2_medium, input_dropout_prob=0.0, input_dense=1024) # if you forget input dense must match gpt hidden

#%%
trainer = TrainerModule(model, 'PARAM_11_GPT2',#'no mean shuffled inputs pose in hid',#f'11 big lr: {LEARNING_RATE} bs: {batch_size} epcs: {max_epochs}', 
                        next(it), 
                        num_train_iters, 
                        dataset=src_dataset, 
                        lr=LEARNING_RATE)


#%%

#trainer.load_model(log_dir='PARAM_2')

#%%

trainer.train_model(train_dataloader, train_dataloader, num_epochs=max_epochs)


#%%

