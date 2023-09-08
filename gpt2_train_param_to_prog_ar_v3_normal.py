#%%
# srun -t  2:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --gres=gpu:1 --partition=ampere -A MLMI-WB326-SL2-GPU --pty bash
# sintr -t 1:00:00 --nodes=1 --ntasks-per-node=1 --ntasks=1 --gres=gpu:1 --partition=ampere -A MLMI-WB326-SL2-GPU --qos INTR

# =========== To run - use the following commands first ==========
# cd /rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr
# source venv/bin/activate

# you may need this if not running on ampere partition
# module load cuda/11.8 cudnn/8.9_cuda-11.8 

# If .so file is not found run this
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib


# On local
# ssh -L 8089:127.0.0.1:8089 wb326@login.hpc.cam.ac.uk
# On remote session that starts
# tensorboard --logdir "/home/wb326/rds/rds-dsk-lab-eWkDxBhxBrQ/iTracr/inverse-tracr/.logs" --port 8089 --samples_per_plugin=images=10000

# Causal Masking setup
# input  target
# W1     0
# W2     0
# W3     0
# PAD    PAD
# ...    ...
# START  START
# START  R1
# R1     R2
# R2     R3
# R3     R4
# R4     END
# END    0 


import os
import jax


#os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


from jax import random
import jax.numpy as jnp
import os
from torch.utils.tensorboard import SummaryWriter
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm
import numpy as np
import torch, flax
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
from data.parameter_program_dataloader import TorchParameterProgramDataset
from data.plot_true_v_pred import plot_orginal_heatmaps, figure_to_array, plot_orginal_heatmaps_ar
from data.dataset import example_program_dataset
from data.encoded_dataloaders import encode_rasp_program
from utils.export_compressed_params import compress_params, encode_jax_params
from utils.jax_helpers import JaxMemUsage
JaxMemUsage.launch(interval=0.01)
from dill import dump, load
from jaxlib.xla_extension import XlaRuntimeError
from models import GPT2, GPT2Config, GPTNeo, GPTJ
from transformers.models.gptj.configuration_gptj import GPTJConfig
from argparse import Namespace
from data.dataloaders import ProgramEncoder
from functools import partial


# GPT Large Train config
# args = Namespace(
#     batch_size=384,# 256 for medium
#     PROG_LEN = 15,
#     max_epochs = 40,
#     LEARNING_RATE=1e-5,
#     input_dropout_prob = 0.2,
#     in_noise = 0.30, # inverse fraction of the standard deviation of the noise to add
#     max_timesteps = 40,
#     model = 'GPT2',
#     config = 'TINY', #'MEDIUM', # 'LARGE'
#     trail_name='arv3_test',
#     task='Stock', # 'Stock', 'Compressed', 'Natural'
#     autoregressive=True,
# )


args = Namespace(
    batch_size=512,# 256 for medium
    PROG_LEN = 15,
    max_epochs = 40,
    LEARNING_RATE=1e-7,
    input_dropout_prob = 0.0,#2,
    parameter_noise = 0.0, # 30, # inverse fraction of the standard deviation of the noise to add
    ar_input_noise=0.2, # absolute max value of noise
    max_timesteps = 40,
    model = 'GPTNEO', # 'GPT2', 'GPTJ', 'GPTNEO'
    config = 'pythia_125m', #'MEDIUM', # 'LARGE'
    trail_name='arv3_normal_7_slow',
    task='Stock', # 'Stock', 'Compressed', 'Natural'
    autoregressive=True,
)


CHECKPOINT_PATH = ".logs/"

dataset_path = None
if args.task == 'Stock':
    from data.dataloader_streams import ZipPickleStreamReader as StoreReader
    from data.parameter_encoder import CRAFT_TIMESTEPS as TIMESTEPS
    from data.parameter_encoder import CRAFT_ARCH as ARCH
    dataset_path = '.data/iTracr_dataset_v2_train.zip'
elif args.task == 'Compressed':
    from data.dataloader_streams import ZipPickleStreamReader as StoreReader
    from data.parameter_encoder import JAX_TIMESTEPS as TIMESTEPS
    from data.parameter_encoder import JAX_ARCH as ARCH
    dataset_path = 'cp_dataset_train_w.zip'
elif args.task == 'Natural':
    from data.dataloader_streams import ZipPickleStreamReader as StoreReader
    from data.parameter_encoder import JAX_TIMESTEPS as TIMESTEPS
    from data.parameter_encoder import JAX_ARCH as ARCH
    dataset_path = 'cp_dataset_train_all.zip'


#%%
class TrainerModule:

    def __init__(self, model, model_name, exmp_batch, max_iters, dataset, max_output_length: int, lr=1e-3, warmup=100, seed=42):
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
        self.seg_sizes=src_dataset.get_segment_sizes()
        self.dataset = dataset
        self.model = model
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.create_functions()
        self.init_model(exmp_batch)
        self.src_dataset = src_dataset
        self.max_output_length = max_output_length

    
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
        post_encoded_program = self.src_dataset.tokens_to_onehot(encoded_ops)
        x,y,loss_mask,attention_mask = TorchParameterProgramDataset.post_process_step(self.dataset.prog_len, x=np.array(encoded_model), y=post_encoded_program, TIMESTEPS=TIMESTEPS, ARCH_LABELS=ARCH)
        x,y, loss_mask, attention_mask, pos_ids = collate_fn( data=[[x, y, loss_mask, attention_mask]])
        logits, fig = self.apply(x, attention_mask=attention_mask, pos_id=pos_ids, labels=y)
        return logits, fig


    def apply(self, inp_data, attention_mask, pos_id, labels=None, loss_mask=None, seed=0):
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
            if loss_mask is None:
                return logits, heat_img
            else:
                acc = self.accuracy_fn(logits, labels, loss_mask)
                return logits, heat_img, acc
        else:
            return logits, None

    
    def get_accuracy_function(self):
        def accuracy(logits, labels, loss_mask):
            """
              logits.shape = (BS, Timesteps, sum(seg_sizes))  - since onehot encoded
              labels.shape = (BS, Timesteps, segements)               - since ordinal encoded
              loss_mask.shape = (BS, Timesteps)            - mask over the timesteps to use
          
              We have predictions
                                                 ---> BS batches
                  Batch 1     |   Batch 2      |  Batch BS |  ^
                    PAD       |     PAD        |           |  |  Timesteps
               00000010000000 |     PAD        |           |  v
               00000000001000 | 0000010000000  |           |
                    ...             ...
              <------------->
                sum(seg_sizes)
              
              Example where BS = 2, timesteps = 3, seg sizes = 3, first timestep in second batch is padded
              logits = np.array([[[0, 1, 0],[0, 1, 0], [0, 0, 1]], 
                                  [[0, 0, 0],[1, 0, 0],[0, 1, 0]]])
              labels = np.expand_dims(np.array([[1, 1, 2],
                                                [0, 0, 1]]), axis=2)
              loss_mask = np.array([[1, 1, 1],
                                  [0, 1, 1]])
              seg_sizes = [3]
            """
            
            def logit_classes_jnp(logits):
                classes = []
                logits = jnp.array(logits)
                
                ptr = 0
                for i, seg_size in enumerate(self.seg_sizes):
                    classes.append(logits[:, :, ptr:ptr + seg_size].argmax(axis=2))
                    ptr += seg_size
                classes = jnp.stack(classes, axis=2)
                return classes
            classes = logit_classes_jnp(logits) # (BS, Timesteps, segements)
            
            # (BS, Timesteps, segements)
            repeated_loss_mask = jnp.repeat(loss_mask[:, :, jnp.newaxis], classes.shape[2], axis=2)

            relevant_classes = classes * repeated_loss_mask
            relevant_labels = labels * repeated_loss_mask
            relevant_labels += 1 - repeated_loss_mask # ensure the masked out values are different
            acc_times_timesteps_ish = relevant_classes == relevant_labels
            acc_times_timesteps_ish = acc_times_timesteps_ish.sum(axis=[1,2])
            acc_batch = acc_times_timesteps_ish /  repeated_loss_mask.sum(axis=[1,2])
            return acc_batch
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
    
    def get_ar_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            # Input data has shape (batch_size, time_steps, features)
            # Labels has shape (batch_size, time_steps, 5)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            batch_size, time_steps, features = inp_data.shape
            out_features = sum(self.seg_sizes)
            
            rng, dropout_apply_rng = random.split(rng)
                
            ar_inputs = inp_data
            logits = None
            for ar_timestep in range(self.max_output_length):
                logits = self.model.apply({'params': params}, ar_inputs, attention_mask=attention_mask, train=train, position_ids=pos_id, rngs={'dropout': dropout_apply_rng})
                
                # output timesteps of form:
                # ......... INPUT_DATA .........., <START>, pred_1, ..., pred_{max_output_length}, <END>
                # vvvvv  | ------------------ INPUT_DATA ---------------- |  ---- PREDICTIONS ---- | --------------------- UNSEEN_PREDS ------------- |         PROG_END (1 only if final step)        | Zero for missing prog_start token
                # ar_mask = [0] * (time_steps - self.max_output_length - 2) +  [1] * (ar_timestep+1) + [0] * (self.max_output_length - ar_timestep - 1) + [int(ar_timestep==(self.max_output_length-1))] + [0]
                # ar_mask = jnp.array(ar_mask)
                repeated_ar_mask = jnp.concatenate([
                                jnp.zeros((time_steps - self.max_output_length - 2, out_features)),
                                jnp.ones((ar_timestep+1+ (1 if ar_timestep==(self.max_output_length-1) else 0), out_features)),
                                jnp.zeros((self.max_output_length - ar_timestep - 1 + (0 if ar_timestep==(self.max_output_length-1) else 1) + 1, out_features))],
                            axis=0)
                #repeated_ar_mask = jnp.repeat(ar_mask[:, jnp.newaxis], out_features, axis=1)
                masked_logits = logits * repeated_ar_mask
                
                masked_logits = masked_logits[:, :-1, :] # cut off the final token, there is a 1 timestep shift between predictions and inputs
                
                # Add to the start to shift
                masked_logits = jnp.concatenate([jnp.zeros((batch_size, 1, out_features)), masked_logits], axis=1)
                
                # pad the start of the logits in the area with our parameter tokens
                masked_logits = jnp.concatenate([masked_logits, jnp.zeros((batch_size, time_steps, features-out_features))], axis=2)
                
                ar_inputs = inp_data + masked_logits
                
            ptr = 0
            loss = 0
            for i, seg_size in enumerate(self.seg_sizes):
                loss += optax.softmax_cross_entropy_with_integer_labels(logits[:, :time_steps, ptr:ptr + seg_size], labels[:, :time_steps, i]) * loss_mask
                ptr += seg_size

            loss = loss.mean()
            acc = self.accuracy_fn(logits, labels, loss_mask)
            return loss, (acc, rng)
        return calculate_loss
    
    def get_verbose_fns(self):
        def verbose_jitted_ar_step(state, batch):
            # labels = (batch_size, max_time_steps, ordinal_features)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch

            # logits = (batch_size, time_steps, features)
            logits = self.model.apply({'params': state.params}, inp_data, attention_mask=attention_mask, position_ids=pos_id, train=False)#, rngs={'dropout': dropout_apply_rng})
            
     
            ptr = 0
            loss = []
            for i, seg_size in enumerate(self.seg_sizes):
                loss.append(optax.softmax_cross_entropy_with_integer_labels(logits[:, :, ptr:ptr + seg_size], labels[:, :, i]) * loss_mask)
                ptr += seg_size
            
            loss = jnp.stack(loss, axis=2)
            return loss, logits, inp_data, None
        
        def verbose_jitted_ar_full(state, batch):
            # Input data has shape (batch_size, time_steps, features)
            # labels = (batch_size, max_time_steps, ordinal_features)
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            out_features = sum(self.seg_sizes)
            batch_size, time_steps, features = inp_data.shape
            
            ar_inputs = inp_data
            ar_input_log = []
            logits = None
            for ar_timestep in range(self.max_output_length):
                logits = self.model.apply({'params': state.params}, ar_inputs, attention_mask=attention_mask, train=False, position_ids=pos_id)

                # output timesteps of form:
                # ......... INPUT_DATA .........., <START>, pred_1, ..., pred_{max_output_length}, <END>
                # vvvvv  | ------------------ INPUT_DATA ---------------- |  ---- PREDICTIONS ---- | --------------------- UNSEEN_PREDS ------------- |         PROG_END (1 only if final step)        | Zero for missing prog_start token
                # ar_mask = [0] * (time_steps - self.max_output_length - 2) +  [1] * (ar_timestep+1) + [0] * (self.max_output_length - ar_timestep - 1) + [int(ar_timestep==(self.max_output_length-1))] + [0]
                # ar_mask = jnp.array(ar_mask)
                repeated_ar_mask = jnp.concatenate([
                                jnp.zeros((time_steps - self.max_output_length - 2, out_features)),
                                jnp.ones((ar_timestep+1+ (1 if ar_timestep==(self.max_output_length-1) else 0), out_features)),
                                jnp.zeros((self.max_output_length - ar_timestep - 1 + (0 if ar_timestep==(self.max_output_length-1) else 1) + 1, out_features))],
                            axis=0)
                #repeated_ar_mask = jnp.repeat(ar_mask[:, jnp.newaxis], out_features, axis=1)
                masked_logits = logits * repeated_ar_mask
                
                masked_logits = masked_logits[:, :-1, :] # cut off the final token, there is a 1 timestep shift between predictions and inputs
                
                # Add to the start to shift
                masked_logits = jnp.concatenate([jnp.zeros((batch_size, 1, out_features)), masked_logits], axis=1)
                
                # pad the start of the logits in the area with our parameter tokens
                masked_logits = jnp.concatenate([masked_logits, jnp.zeros((batch_size, time_steps, features-out_features))], axis=2)
                
                ar_input_log.append(ar_inputs)
                
                ar_inputs = inp_data + masked_logits
                
   
            ptr = 0
            loss = []
            for i, seg_size in enumerate(self.seg_sizes):
                loss.append(optax.softmax_cross_entropy_with_integer_labels(logits[:, :, ptr:ptr + seg_size], labels[:, :, i]) * loss_mask)
                ptr += seg_size
            
            loss = jnp.stack(loss, axis=2)
            return loss, logits, ar_inputs, ar_input_log
            
        return verbose_jitted_ar_step, verbose_jitted_ar_full
    
    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()

        ar_loss = self.get_ar_loss_function()

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
            loss, (acc, rng) = ar_loss(state.params, rng, batch, train=False)
            return loss, acc, rng
        self.eval_step = jax.jit(eval_step)

        
        verbose_jitted_ar_step, verbose_jitted_ar_full = [jax.jit(x) for x in self.get_verbose_fns()]
        
        def verbose_step(verbose_fn, state, batch, step, ext: str = ""):
            inp_data, labels, loss_mask, attention_mask, pos_id = batch
            loss, logits, ar_inputs, ar_input_log = verbose_fn(state, batch)
            loss = np.array(loss)
            
            # if ar_inputs is not None:
            #     c_logits, c_inputs, m_logits = ar_inputs
            #     for i in range(len(c_inputs)):
            #         print(f"\n\n\n\n\ntimestep {i}\n")
            #         print(self.dataset.decode_pred(c_logits[i], 0))
            #         print("\n")
            #         print(self.dataset.decode_pred(m_logits[i][:, :, :c_logits[i].shape[2]], 0))
            #         # print(m_logits[i].shape)
            #         # print(m_logits[i])
            #         print("\n")
            #         print(self.dataset.decode_pred(c_inputs[i][:, :, :c_logits[i].shape[2]], 0))

            # loss = (batch_size, time_steps, features)
            
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
            
            if ar_inputs is not None:
                ar_classes = logit_classes_jnp(ar_inputs[:, :, :logits.shape[2]])
                
            
            max_prog_len = self.dataset.prog_len
            for batch_id in range(min(5, labels.shape[0])):
                heat_img = plot_orginal_heatmaps(labels[:, -max_prog_len-2:, :], classes[:, -max_prog_len-2:, :], self.dataset, loss=loss[:, -max_prog_len-2:, :], BATCH_ID = batch_id)
                self.logger.add_image(f"verbose{ext}/heatmap", heat_img, global_step=step+batch_id, dataformats='HWC')
                # if ar_inputs is not None:
                #     heat_img = plot_orginal_heatmaps(ar_classes[:, -max_prog_len-2:, :], classes[:, -max_prog_len-2:, :], self.dataset, loss=loss[:, -max_prog_len-2:, :], BATCH_ID = batch_id)
                #     self.logger.add_image(f"verbose{ext}/input_heatmap", heat_img, global_step=step+batch_id, dataformats='HWC')
            
            if ar_inputs is not None:
                heat_img = plot_orginal_heatmaps_ar(labels, classes, self.dataset, inputs=ar_classes, loss=loss , BATCH_ID = batch_id)
                self.logger.add_image(f"verbose{ext}/input_heatmap", heat_img, global_step=step+batch_id, dataformats='HWC')
                      
            if ar_input_log is not None:
                for i in range(len(ar_input_log)):
                    ar_classes_i = logit_classes_jnp(ar_input_log[i][:, :, :logits.shape[2]])
                    heat_img = plot_orginal_heatmaps_ar(labels, classes, self.dataset, inputs=ar_classes_i, loss=loss , BATCH_ID = batch_id)
                    self.logger.add_image(f"verbose{ext}/input_heatmap_{i}", heat_img, global_step=step+batch_id, dataformats='HWC')
            

            self.logger.add_histogram(f"verbose{ext}/output", np.array(logits), global_step=step)

            self.logger.add_scalar(f"verbose{ext}/acc", acc.mean().item(), global_step=step)


        self.verbose_ar_step = partial(verbose_step, verbose_jitted_ar_step)
        self.verbose_ar_full = partial(verbose_step, verbose_jitted_ar_full)

        self.accuracy_fn = self.get_accuracy_function()



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

                    # if idx == 100:
                    #     jax.profiler.start_trace("jax-profile")
                    # elif idx == 150:
                    #     jax.profiler.stop_trace()
                        
                    # -------------------------- Train ---------------------------------------------
                    self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
                    

                    # ----------- metrics -------------
                    loss, accuracy = loss.item(), np.array(accuracy)
                    loss_sum += loss
                    acc_sum += accuracy.mean()

                    
                    # ----------- TF metrics ----------
                    global_step = idx * args.batch_size + (epoch - 1) * DATALOADER_LENGTH * args.batch_size
                    self.logger.add_scalar('train_hf/loss', loss, global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy', accuracy.mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy100', (accuracy == 1.0).mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy90', (accuracy > 0.9).mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy80', (accuracy > 0.8).mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy70', (accuracy > 0.7).mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy60', (accuracy > 0.6).mean(), global_step=global_step)
                    self.logger.add_scalar('train_hf/accuracy50', (accuracy > 0.5).mean(), global_step=global_step)
                    
                    
                    # ------------ Low freq metrics --------------
                    if (idx) % LOGGING_INTERVAL == 0:
                        print("Verbose Iterations")
                        self.verbose_ar_step(state=self.state, batch=batch, step=global_step, ext="_train")
                        if validation_loader is not None:
                            val_batch = next(iter(validation_loader)) # we shuffle the validation set so it's fine
                            self.verbose_ar_full(state=self.state, batch=val_batch, step=global_step, ext="_val")
                    

                    # ------------ Evaluation Step ---------------
                    if validation_loader is not None and (idx + 1) % VALIDATION_INTERVAL == 0:
                        eval_acc, eval_loss = self.eval_model(validation_loader)
                        self.logger.add_scalar('val/accuracy', eval_acc.mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy100', (eval_acc == 1.0).mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy90', (eval_acc > 0.9).mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy80', (eval_acc > 0.8).mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy70', (eval_acc > 0.7).mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy60', (eval_acc > 0.6).mean(), global_step=global_step)
                        self.logger.add_scalar('val/accuracy50', (eval_acc > 0.5).mean(), global_step=global_step)
                        trainer.logger.add_scalar('val/loss', eval_loss, global_step=global_step)
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            trainer.save_model(step=global_step)
                        
                        

                    # ----------- TQDM ----------------
                    tepoch.set_postfix({'Batch': idx, 'Train Loss': loss, 'Acc': accuracy.mean(), 'MaxMem': JaxMemUsage.max_usage_str, 'Mem': JaxMemUsage.usage_str})
                    tepoch.update(1)
                    
                    count += 1

                except XlaRuntimeError as E:
                    print(E)
                    print(batch[0].shape)
                    jax.lib.xla_bridge.get_backend().defragment()
                    if isinstance(E, KeyboardInterrupt):
                        raise(E)
            # if args.task=='Stock':
            #     self.eval_programs(step=epoch)
            self.logger.add_scalar('train/loss', loss_sum / count, global_step=epoch)
            self.logger.add_scalar('train/accuracy', acc_sum / count, global_step=epoch)
            trainer.logger.flush()

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        loss_sum, count = 0.0, 0
        acc_list = []        
        for batch in tqdm(data_loader, desc='Evaluating'):
            loss, acc, self.rng = self.eval_step(self.state, self.rng, batch)

            bs = batch[0].shape[0]
            loss = loss.item()
            acc_list += list(acc)
            loss_sum += loss * bs
            count += bs
        eval_loss = loss_sum / count
        return np.array(acc_list), eval_loss
    
    def eval_programs(self, step=0):
        for program_lam, lam_names, name, numeric_vars in example_program_dataset:
            program = program_lam()
            encoded_model, encoded_ops = encode_rasp_program(program, args.PROG_LEN, lam_names, numeric_vars=numeric_vars)
            logits, fig = self.raw_apply(encoded_model, encoded_ops)
            img = figure_to_array(fig)
            self.logger.add_image("examples/"+name, img, global_step=step, dataformats='HWC')
    




    def save_model(self, step=0):
        # Save current model at certain training iteration
        try:
            checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)
            dump(self.state.opt_state, open(os.path.join(self.log_dir, "optimiser_state.pkl"), "wb"))    
        except flax.errors.InvalidCheckpointError:
            print(f"failed to save the checkpoint, an newer checkpoint exists than step {step}")

    def load_model(self, log_dir=None, load_state=True):
        log_dir = self.log_dir if log_dir is None else log_dir
        
        if not os.path.isdir(os.path.join(CHECKPOINT_PATH, log_dir)): raise FileNotFoundError("Could not find the model directory")

        params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, log_dir), target=self.state.params)
        opt_state = load( open(os.path.join(CHECKPOINT_PATH, log_dir, "optimiser_state.pkl"), "rb" ) )
        
        if load_state:
            self.state = train_state.TrainState(
                                step=0,
                                apply_fn=self.model.apply,
                                params=params,
                                tx=self.state.tx,
                                opt_state=opt_state)
        else:
            self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)





src_dataset = TorchParameterProgramDataset(args.PROG_LEN)


def add_noise_to_params(params, frac_of_std=0.1):
    def tree_map(f, d):
        return dict(zip(d.keys(), [f(x) for x in d.values()]))
    def add_noise(x):
        multiplier = max(np.std(x) * frac_of_std, 0.00000001)
        noise = np.random.normal(loc=0, scale = 1/multiplier, size=x.shape)
        x = x + noise
        return x
    return tree_map(add_noise, params)

#%%

class WrappedDataset(StoreReader):
    def __init__(self, dir: str, max_prog_len: int, max_time_step_reduction_sample: int, first=None, last=None, parameter_noise=0, autoregressive=False, ar_input_noise=0.0) -> None:
        super().__init__(dir, first, last)
        self.max_prog_len = max_prog_len
        self.max_timesteps = max_time_step_reduction_sample
        self.parameter_noise = parameter_noise
        self.autoregressive = autoregressive
        self.prog_enc = ProgramEncoder(max_prog_len)
        self.ar_input_noise = ar_input_noise
    
    def __getitem__(self, idx):
        # first rejection sample under the max timestep
        x_shape = self.max_timesteps + 1
        offset = 0
        while x_shape > self.max_timesteps:
            circular_index = (idx + offset) % self.__len__()
            x,y = super().__getitem__(circular_index)
            if args.task in ['Compressed', 'Natural']: # we left these samples parameters unencoded
                x = compress_params(x)
                if self.parameter_noise > 0:
                    x = add_noise_to_params(x, self.parameter_noise)
                x = encode_jax_params(x)
            x,y,loss_mask,attention_mask = TorchParameterProgramDataset.post_process_step(self.max_prog_len, x=x, y=y, TIMESTEPS=TIMESTEPS, ARCH_LABELS=ARCH)
            x_shape = x.shape[0]
            offset += 1
            
            if self.autoregressive:
                autoregressive_inputs = y
            else:
                autoregressive_inputs = np.concatenate((y[0:1, :], np.zeros((y.shape[0]-1, y.shape[1]), dtype=np.int32)), axis=0)
            
            y = y[1:, :] # cut off the program start token
            y = np.concatenate((y, np.zeros((1, y.shape[1]), dtype=np.int32)), axis=0)
            autoregressive_inputs = self.prog_enc.tokens_to_onehot(autoregressive_inputs, ignore_padding=True)
            
            if self.parameter_noise > 0:
                noise = np.random.normal(loc=0, scale = self.parameter_noise, size=autoregressive_inputs.shape)
                x = autoregressive_inputs + noise
                
        return np.array(x),np.array(y),np.array(loss_mask),np.array(attention_mask), autoregressive_inputs

#%%
# dataset = WrappedDataset(dataset_path, args.PROG_LEN, args.max_timesteps, first=0.9, autoregressive=True)
# it = iter(dataset)
# #%%
# x, y, loss_mask, attention_mask, autoregressive_mask = next(it)
# y, loss_mask, autoregressive_mask

# #%%

dataset = WrappedDataset(dataset_path, args.PROG_LEN, args.max_timesteps, last=0.1, autoregressive=False)
it = iter(dataset)
#%%
x, y, loss_mask, attention_mask, autoregressive_mask = next(it)
y, loss_mask, autoregressive_mask

print(autoregressive_mask[-17, :])
print(autoregressive_mask[-16, :])
print(autoregressive_mask[-15, :])
print(y)
# #%%
# from data.dataloaders import ProgramEncoder
# prog_enc = ProgramEncoder(args.PROG_LEN)
# onehot = prog_enc.tokens_to_onehot(y, ignore_padding=True)


#%%
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
    from data.parameter_encoder import get_onehot_timestep_encoder
    ONEHOT_TIMESTEP_ENCODER = get_onehot_timestep_encoder(TIMESTEPS)


    INPUT_PROGRAM_FLAGS = torch.tensor(np.stack([ONEHOT_TIMESTEP_ENCODER[token] for token in ['PROGRAM_START'] + (['PAD'] * PROG_LEN) + ['PROGRAM_END']]))


    def collate_fn(data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        loss_masks = [torch.tensor(d[2], device='cpu') for d in data]
        attention_masks = [torch.tensor(d[3], device='cpu') for d in data]
        autoregressive_inputs = [torch.tensor(d[4], device='cpu') for d in data]
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
        
        if autoregressive_inputs[0].shape != torch.Size([]): # if not we're running autoregressively
            autoregressive_inputs =  torch.stack(autoregressive_inputs) # Batch, timesteps, features
            _, ar_timesteps, ar_features = autoregressive_inputs.shape
            inputs = inputs[:, :-ar_timesteps, :] # cut off the old program reserved timesteps
            
            # pad the AR input features to the input feature size
            autoregressive_inputs = torch.concatenate([autoregressive_inputs, torch.zeros((bs, ar_timesteps, parameter_features-ar_features))], axis=2)
            
            # concat the timesteps together
            inputs = torch.concatenate([inputs, autoregressive_inputs], axis=1)
        
        assert ((np.array(inputs) * np.repeat(np.array(attention_masks)[:, :, np.newaxis], inputs.shape[2], axis=2)) == np.array(inputs)).all() # verify that we're not masking out important data
        
        # inputs, targets, loss_masks, attention_masks = np.zeros(inputs.shape), np.zeros(targets.shape), np.ones(loss_masks.shape), np.ones(attention_masks.shape)        
        # indices = np.random.randint(0, targets.shape[2], size=bs)
        # arr = np.arange(0, bs)
        # inputs[arr, -1, indices] = 1
        # targets[arr, 0, indices] = 1   
        
        return np.array(inputs), np.array(targets).astype(int), np.array(loss_masks), np.array(attention_masks), pos_ids
    return collate_fn


# dataset contains 4.1 million samples, 0.3% of this is 15k samples, which should be enough to judge the performance of the model
dataset = WrappedDataset(dataset_path, args.PROG_LEN, args.max_timesteps, first=0.997, autoregressive=True)
test_dataset = WrappedDataset(dataset_path, args.PROG_LEN, args.max_timesteps, last=0.003 )

next(iter(dataset))
next(iter(test_dataset))

# sr = StoreReader(dataset_path,  first=0.9)
# for x in tqdm():
#     pass

# sr = StoreReader(dataset_path,  first=0.9)
# for x in tqdm(np.random.randint(0, len(sr), len(sr))):
#     sr.__getitem__(x)

print(f"Dataset contains: {len(dataset)} samples" )

collate_fn = make_collate_fn(args.PROG_LEN)


# note num_workers * prefetch_factor should be greater than the batch size
train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, prefetch_factor=36, shuffle=True)#, pin_memory=True) num_workers=1, prefetch_factor=2)#
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, prefetch_factor=36, shuffle=True)#, pin_memory=True)
num_train_iters = len(train_dataloader) * args.max_epochs


next(iter(train_dataloader))
next(iter(test_dataloader))

# for x in tqdm(train_dataloader):
#     pass

#%%

def testing_loaders():
    it = iter(test_dataloader)
    x,y,_,_, _ = next(it)
    src_dataset.decode_pred(y, 0)

    it = iter(train_dataloader)
    x,y,_,_, _ = next(it)

    #print(src_dataset.decode_pred(y, 0))


testing_loaders()






from data.parameter_encoder import decode_timesteps


test_it = iter(test_dataloader)
def decode_test_sample():
    sample = next(test_it)
    print(src_dataset.decode_pred(sample[1], 0))
    print(decode_timesteps(sample[0], TIMESTEPS=TIMESTEPS, batch=1))
decode_test_sample()



# x,y, loss_mask, attention_mask = next(iter(dataset))

#%%
model, model_config = None, None
if args.model == 'GPT2':
    import json
    with open(f'utils/gpt2_configs/gpt2_{args.config.lower()}.json') as f: # GPT2 Large - 774M
        config_json = json.load(f)
    model_config = GPT2Config(**config_json)





    model = GPT2(num_classes=sum(src_dataset.get_segment_sizes()), gpt_config=model_config, input_dropout_prob=args.input_dropout_prob)

elif args.model == 'GPTJ':
    import json
    with open(f'utils/gptj_pythia/{args.config}.yml') as f:
        config_json = json.load(f)
        model_config = GPTJConfig(
            vocab_size =          None,
            n_positions =         config_json['hidden_size'],#config_json['max_position_embeddings'],
            n_embd =              config_json['hidden_size'],
            n_layer =             config_json['num_layers'],
            n_head =              config_json['num_attention_heads'],
            rotary_dim =          64,
            n_inner =             None,
            activation_function = "gelu_new",
            resid_pdrop =         config_json['hidden_dropout'],
            embd_pdrop =          config_json['hidden_dropout'],
            attn_pdrop =          config_json['attention_dropout'],
            layer_norm_epsilon =  1e-5,
            initializer_range =   0.02,
            use_cache =           True,
            bos_token_id =        None,
            eos_token_id =        None,
            tie_word_embeddings = False,
        )
    # model_config = GPTJConfig(
    #         vocab_size=None,
    #         n_positions=1024,
    #         n_embd=1024,
    #         n_layer=28,
    #         n_head=16,
    #         rotary_dim=64,
    #         n_inner=None,
    #         activation_function="gelu_new",
    #         resid_pdrop=0.0,
    #         embd_pdrop=0.0,
    #         attn_pdrop=0.0,
    #         layer_norm_epsilon=1e-5,
    #         initializer_range=0.02,
    #         use_cache=True,
    #         bos_token_id=None,
    #         eos_token_id=None,
    #         tie_word_embeddings=False
    # )
        

    #


    model = GPTJ(num_classes=sum(src_dataset.get_segment_sizes()), gpt_config=model_config, input_dropout_prob=args.input_dropout_prob) # if you forget input dense must match gpt hidden


elif args.model == 'GPTNEO':
    
    from transformers import GPTJConfig

    # import yaml

    # with open(r'utils/gptneo_configs/pythia_125m.json') as file:
    #     documents = yaml.full_load(file)


    import json

    with open(f'utils/gptneo_configs/{args.config}.json') as f:
        config_json = json.load(f)


    model_config = GPTJConfig(**config_json)

    model_config.n_embd  = model_config.hidden_size
    model_config.n_head  = model_config.num_heads
    model_config.n_layer = model_config.num_layers

    # model_config.resid_dropout = 0.2
    # model_config.embed_dropout = 0.2
    # model_config.attention_dropout = 0.2


    model = GPTNeo(num_classes=sum(src_dataset.get_segment_sizes()), gpt_config=model_config, input_dropout_prob=args.input_dropout_prob)


trainer = TrainerModule(model, 
                        f'{args.trail_name} {args.model} {args.config} TASK: {args.task} LR: {args.LEARNING_RATE} ParamNoise: {args.parameter_noise} InpDrop: {args.input_dropout_prob} bs: {args.batch_size} nembed: {model_config.n_embd} n_layer: {model_config.n_layer} n_head: {model_config.n_head}',
                        next(test_it), 
                        num_train_iters, 
                        dataset=src_dataset, 
                        lr=args.LEARNING_RATE,
                        max_output_length=args.PROG_LEN)
_ = open(os.path.join(trainer.log_dir, "hyperparameters"), "w").write(f"{args}\n{model_config}")

#%%

# trainer.eval_programs()
# trainer.load_model(log_dir=f"XXX{args.model} cont LR {args.LEARNING_RATE} bs: {args.batch_size} nembed: {model_config.n_embd} n_layer: {model_config.n_layer} n_head: {model_config.n_head}")

trainer.load_model(log_dir=f"arv3_normal_4_slow GPTNEO pythia_125m TASK: Stock LR: 1e-07 ParamNoise: 0.0 InpDrop: 0.0 bs: 64 nembed: 768 n_layer: 12 n_head: 12")

# trainer.load_model(log_dir=f"PARAM_NumVar_GPT2_LARGE cont LR 1e-06 bs: 256 nembed: 1280 n_layer: 36 n_head: 20")
# test_val_acc, test_val_loss = trainer.eval_model(test_dataloader)
# import pandas as pd
# pd.Series(test_val_acc).to_csv('GPT_LARGE_NUMVAR.csv')


#%%

for epoch_idx in range(1, args.max_epochs+1):
    trainer.train_epoch(train_dataloader, epoch=epoch_idx, validation_loader=test_dataloader, VALS_PER_EPOCH=8, LOGS_PER_EPOCH=8 )


#%%

