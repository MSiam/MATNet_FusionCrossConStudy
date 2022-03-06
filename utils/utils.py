import glob

import torch
import os
import pickle
from collections import OrderedDict
import wandb
from typing import Optional, Dict
import pathlib

import yaml

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_parallel(encoder_dict, decoder_dict):
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict


def get_base_params(args, model):
    b = []
    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.res2)
    b.append(model.res3)
    b.append(model.res4)
    b.append(model.res5)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k


def get_skip_params(model):
    b = []
    b.append(model.sk2.parameters())
    b.append(model.sk3.parameters())
    b.append(model.sk4.parameters())
    b.append(model.sk5.parameters())
    b.append(model.bn2.parameters())
    b.append(model.bn3.parameters())
    b.append(model.bn4.parameters())
    b.append(model.bn5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def merge_params(params):
    for j in range(len(params)):
        for i in params[j]:
            yield i


def get_optimizer(optim_name, lr, parameters, weight_decay=0, momentum=0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                              lr=lr, weight_decay=weight_decay,
                              momentum=momentum)
    elif optim_name == 'adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                               lr=lr, weight_decay=weight_decay)
    return opt

def read_and_merge_cfg(args):
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg.items():
            setattr(args, k, v)
    if not hasattr(args, 'third_stream_config'):
        args.third_stream_config = {}
    return args

def save_checkpoint_epoch(args, model, enc_opt, dec_opt, epoch, best=False):
    torch.save(model.module.encoder.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'encoder_{}.pt'.format(epoch)))
    torch.save(model.module.decoder.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'decoder_{}.pt'.format(epoch)))
    if enc_opt is not None:
        torch.save(enc_opt.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'enc_opt_{}.pt'.format(epoch)))
    torch.save(dec_opt.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'dec_opt_{}.pt'.format(epoch)))

    if best:
        torch.save(model.module.encoder.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'encoder.pt'))
        torch.save(model.module.decoder.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'decoder.pt'))
        if enc_opt is not None:
            torch.save(enc_opt.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'enc_opt.pt'))
        torch.save(dec_opt.state_dict(), os.path.join(args.ckpt_path, args.model_name, 'dec_opt.pt'))

    # save parameters for future use
    pickle.dump(args, open(os.path.join(args.ckpt_path, args.model_name, 'args.pkl'), 'wb'))


def load_checkpoint_epoch(model_name, epoch, use_gpu=True, load_opt=True, args=None, ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = args.ckpt_path

    if use_gpu:
        print('Loading from ', os.path.join('ckpts', ckpt_path, model_name, 'encoder_{}.pt'.format(epoch)))
        encoder_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'encoder_{}.pt'.format(epoch)))
        decoder_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'decoder_{}.pt'.format(epoch)))
        if load_opt:
            if os.path.exists(os.path.join('ckpts', ckpt_path, model_name, 'enc_opt_{}.pt'.format(epoch))):
                enc_opt_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'enc_opt_{}.pt'.format(epoch)))
            else:
                enc_opt_dict = None
            dec_opt_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'dec_opt_{}.pt'.format(epoch)))
    else:
        encoder_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'encoder_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'decoder_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'enc_opt_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join('ckpts', ckpt_path, model_name, 'dec_opt_{}.pt'.format(epoch)), map_location=lambda storage, location: storage)
    # save parameters for future use
    if load_opt:
        args = pickle.load(open(os.path.join('ckpts', ckpt_path, model_name, 'args.pkl'), 'rb'))

        return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args
    else:
        return encoder_dict, decoder_dict, None, None, None


def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             entity_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        print('Resuming from wandb path... ', wandb_id_file_path)
        resume_id = wandb_id_file_path.read_text()
        wandb.init(entity=entity_name,
                   project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config)
                   # settings=wandb.Settings(start_method="thread"))
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(entity=entity_name, project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        wandb.config.update(config)

    return config


def clean_checkpoint_dir(model_path, epoch, num_to_keep=3):
    """
    Remove all checkpoints from the checkpointing other than num_to_keep checkpoints.
    Args:
        model_path: path to folder of current job
        epoch: current epoch
    """
    names = glob.glob(model_path + '/*')
    names = [f for f in names if "encoder_" in f]
    assert len(names), "No checkpoints found in '{}'.".format(names)
    epochs_to_remove = list(range(0, epoch-num_to_keep))
    if len(epochs_to_remove) > 0:
        checkpoints_to_remove = []
        for num in epochs_to_remove:
            checkpoints_to_remove.append(os.path.join(model_path, 'encoder_{}.pt'.format(num)))
            checkpoints_to_remove.append(os.path.join(model_path, 'decoder_{}.pt'.format(num)))
            checkpoints_to_remove.append(os.path.join(model_path, 'enc_opt_{}.pt'.format(num)))
            checkpoints_to_remove.append(os.path.join(model_path, 'dec_opt_{}.pt'.format(num)))
        for file in checkpoints_to_remove:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    print('Cant remove: {}'.format(file))

def load_last_checkpoint(model_path, use_gpu=True, load_opt=True, args=None):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    names = glob.glob(model_path + '/*')
    old_epochs = [int(f.split('.')[0].split('_')[-1]) for f in names if "encoder_" in f]
    assert len(old_epochs), "No checkpoints found in"
    epoch = sorted(old_epochs)[-1]
    if use_gpu:
        print('Loading from ', os.path.join(model_path, 'encoder_{}.pt'.format(epoch)))
        encoder_dict = torch.load(os.path.join(model_path, 'encoder_{}.pt'.format(epoch)))
        decoder_dict = torch.load(os.path.join(model_path, 'decoder_{}.pt'.format(epoch)))
        if load_opt:
            if os.path.exists(os.path.join(model_path, 'enc_opt_{}.pt'.format(epoch))):
                enc_opt_dict = torch.load(os.path.join(model_path, 'enc_opt_{}.pt'.format(epoch)))
            else:
                enc_opt_dict = None
            dec_opt_dict = torch.load(os.path.join(model_path, 'dec_opt_{}.pt'.format(epoch)))

    return epoch+1, encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args
