from turtle import forward
import warnings, wandb
import pickle
from collections import defaultdict
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os, sys
import shutil
import time
import queue
sys.path.append(os.getcwd())#slt dir
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized,
    synchronize,
)
from utils.metrics import compute_accuracy, wer_list
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from copy import deepcopy
from ctcdecode import CTCBeamDecoder
from itertools import groupby
from utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans


def get_entropy(p):
    logp = p.log()
    return -(p*logp).sum(dim=-1)

def map_phoenix_gls(g_lower):#lower->upper
    if 'neg-' in g_lower[:4]:
        g_upper = 'neg-'+g_lower[4:].upper()
    elif 'poss-' in g_lower:
        g_upper = 'poss-'+g_lower[5:].upper()
    elif 'negalp-' in g_lower:
        g_upper = 'negalp-'+g_lower[7:].upper()
    else:
        g_upper = g_lower.upper()
    return g_upper

def index2token(index, vocab):
    token = [map_phoenix_gls(vocab[i]) for i in index]
    return token


def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False, return_others=False):  # to-do: output_dir
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    vocab = val_dataloader.dataset.vocab
    split = val_dataloader.dataset.split
    cls_num = len(vocab)

    # Prepare word embeddings if available
    word_emb_tab = []
    if val_dataloader.dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(val_dataloader.dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None

    # Progress bar
    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader),
                           desc=val_dataloader.dataset.split.upper())
    else:
        pbar = None

    # Log epoch or step
    if epoch is not None:
        logger.info(f"--- Evaluation epoch={epoch} split={split} #samples={len(val_dataloader.dataset)} ---")
    elif global_step is not None:
        logger.info(f"--- Evaluation step={global_step} split={split} #samples={len(val_dataloader.dataset)} ---")

    model.eval()
    val_stat = defaultdict(float)
    results = defaultdict(dict)
    name_prob = {}

    # Determine prediction source
    contras_setting = cfg['model']['RecognitionNetwork']['visual_head']['contras_setting']
    if contras_setting and 'only' in contras_setting:
        pred_src = 'word_emb_att_scores'
        if any(x in contras_setting for x in ['l1','l2']): pred_src = 'fea_vect'
        if 'margin' in contras_setting: pred_src = 'gloss_logits'
    else:
        pred_src = 'gloss_logits'
    if cfg['model']['RecognitionNetwork']['visual_head']['variant'] in ['arcface','cosface']:
        pred_src = 'gloss_raw_logits'

    # Handle Phoenix CT C decoding if needed (omitted here)

    with torch.no_grad():
        logits_name_lst = []
        for step, batch in enumerate(val_dataloader):
            batch = move_to_device(batch, cfg['device'])
            forward_output = model(
                is_train=False,
                labels=batch['labels'],
                sgn_videos=batch['sgn_videos'],
                sgn_keypoints=batch['sgn_keypoints'],
                epoch=epoch)

            # Skip Phoenix-specific code...

            # Accumulate loss stats
            if is_main_process():
                for k,v in forward_output.items():
                    if '_loss' in k:
                        val_stat[k] += v.item()

            # Iterate through different prediction heads
            for k, gls_logits in forward_output.items():
                if pred_src not in k or gls_logits is None:
                    continue
                logits_name = k.replace(pred_src, '')
                if any(x in logits_name for x in ['word_fused','xmodal_fused']):
                    continue
                if logits_name not in logits_name_lst:
                    logits_name_lst.append(logits_name)

                decode_output = model.predict_gloss_from_logits(gloss_logits=gls_logits, k=10)

                for i in range(decode_output.shape[0]):
                    name = batch['names'][i]
                    hyp = [d.item() for d in decode_output[i]]
                    results[name][f'{logits_name}hyp'] = hyp

                    # SINGLE vs MULTI-LABEL REF
                    if cfg['data']['isContinuous']==False:
                        ref = batch['labels'][i].item()
                    else:
                        labels_i = batch['labels'][i]
                        ref = labels_i.tolist() if isinstance(labels_i, torch.Tensor) else labels_i
                    results[name]['ref'] = ref

            if pbar:
                pbar(step)
        print()

    # Post-processing, accuracy, return
    per_ins_stat_dict, per_cls_stat_dict = compute_accuracy(
        results, logits_name_lst, cls_num, cfg['device'])
    return per_ins_stat_dict, per_cls_stat_dict, results, name_prob, {}



def sync_results(per_ins_stat_dict, per_cls_stat_dict, save_dir=None, wandb_run=None, sync=True):
    logger = get_logger()

    if sync:
        for d in [per_ins_stat_dict, per_cls_stat_dict]:
            for k,v in d.items():
                synchronize()
                torch.distributed.all_reduce(v)

    evaluation_results = {}
    if is_main_process():
        for k, per_ins_stat in per_ins_stat_dict.items():
            correct, correct_5, correct_10, num_samples = per_ins_stat
            logger.info('#samples: {}'.format(num_samples))
            evaluation_results[f'{k}per_ins_top_1'] = (correct / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_1']))
            evaluation_results[f'{k}per_ins_top_5'] = (correct_5 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_5']))
            evaluation_results[f'{k}per_ins_top_10'] = (correct_10 / num_samples).item()
            logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_ins_top_10']))

        # one class missing in the test set of WLASL_2000
        for k, per_cls_stat in per_cls_stat_dict.items():
            top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat
            evaluation_results[f'{k}per_cls_top_1'] = np.nanmean((top1_t / (top1_t+top1_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_1']))
            evaluation_results[f'{k}per_cls_top_5'] = np.nanmean((top5_t / (top5_t+top5_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_5']))
            evaluation_results[f'{k}per_cls_top_10'] = np.nanmean((top10_t / (top10_t+top10_f)).cpu().numpy())
            logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(k, 100*evaluation_results[f'{k}per_cls_top_10']))

        if wandb_run:
            for k, v in evaluation_results.items():
                wandb.log({f'eval/{k}': 100*v})

        if save_dir:
            with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
                pickle.dump(evaluation_results, f)
    if sync:
        synchronize()
    return evaluation_results


def eval_denoise(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False, return_others=False):
    
    logger = get_logger()
    tot = cor = 0
    for step, batch in enumerate(val_dataloader):
        #forward -- loss
        batch = move_to_device(batch, cfg['device'])
        forward_output = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch,
                               denoise_inputs=batch.get('denoise_inputs', {}))
        logits = forward_output['logits']
        decode_results = model.predict_gloss_from_logits(logits, k=10)
        decode_results = decode_results[:,0]
        labels = batch['denoise_inputs']['labels']
        tot += labels.shape[0]
        cor += (decode_results==labels).sum().item()

    evaluation_results = {'per_ins_top_1': cor/tot}
    print()
    logger.info('-----------------------per-ins accuracy: {:.2f}------------------------'.format(100*cor/tot))
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    parser.add_argument("--save_subdir", default='prediction', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--eval_setting', default='origin', type=str)
    # parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    cfg['rank'] = torch.distributed.get_rank()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction_{}_{}.log'.format(args.eval_setting, cfg['rank']))

    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None
    del vocab; del dataset
    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    
    model = DDP(model, 
            device_ids=[cfg['local_rank']], 
            output_device=cfg['local_rank'],
            find_unused_parameters=True)

    for split in ['dev', 'test']:
        logger.info('Evaluate on {} set'.format(split))
        if args.eval_setting == 'origin':
            dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=True)
            per_ins_stat, per_cls_stat, _, _, others = evaluation(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split), 
                    return_prob=False, return_others=False)
            
            sync_results(per_ins_stat, per_cls_stat)
            # if cfg['data']['dataset_name'] == 'WLASL_2000':
            #     other_vocab = [1000, 300, 100]
            # elif cfg['data']['dataset_name'] == 'MSASL_1000':
            #     other_vocab = [500, 200, 100]
            # else:
            #     other_vocab = []
            # for o in other_vocab:
            #     logger.info('-----------------------Variant: {:d}-------------------------'.format(o))
            #     sync_results(others[str(o)]['per_ins_stat'], others[str(o)]['per_cls_stat'])
        
        elif args.eval_setting in ['3x', '5x', 'model_ens', 'central_random_1', 'central_random_2', '5x_random_1', '5x_random_2',
                                    '3x_pad', '3x_left_mid', '3x_left_mid_pad']:
            if args.eval_setting in ['3x', '3x_pad', '3x_left_mid', '3x_left_mid_pad', '5x']:
                if args.eval_setting == '3x':
                    test_p = ['start', 'end', 'central']
                    test_m = ['pad', 'pad', 'pad']
                elif args.eval_setting == '3x_pad':
                    test_p = ['start', 'end', 'central']
                    test_m = ['start_pad', 'end_pad', 'pad']
                elif args.eval_setting == '3x_left_mid':
                    test_p = ['left_mid', 'right_mid', 'central']
                    test_m = ['pad', 'pad', 'pad']
                elif args.eval_setting == '3x_left_mid_pad':
                    test_p = ['left_mid', 'right_mid', 'central']
                    test_m = ['left_mid_pad', 'right_mid_pad', 'pad']
                else:
                    test_p = ['left_mid', 'right_mid', 'start', 'end', 'central']
                    test_m = ['left_mid_pad', 'right_mid_pad', 'start_pad', 'end_pad', 'pad']
                    # test_m = ['pad', 'pad', 'pad', 'pad', 'pad']
                    # test_p = ['start']
                    # test_m = ['pad']
                all_prob = {}
                for t_p, t_m in zip(test_p, test_m):
                    logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                    new_cfg = deepcopy(cfg)
                    new_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                    new_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                    dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                    per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                            epoch=epoch, global_step=global_step, 
                                            generate_cfg=cfg['testing']['cfg'],
                                            save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                    all_prob[t_p] = name_prob
                with open(os.path.join(model_dir,args.save_subdir,split,'prob_5x.pkl'), 'wb') as f:
                    pickle.dump(all_prob, f)

            elif args.eval_setting == 'model_ens':
                all_prob = {}
                # with open('./results_debug/two_lbsm0.2_wordembsim_dual_top2k_ema_fc2_dec_mixup_0.75_0.8/prediction/test/prob_5x.pkl', 'rb') as f:
                #     all_prob['m1'] = pickle.load(f)
                # with open('./results_debug/two_lbsm0.2_wordembsim_dual_top2k_ema_fc2_dec_mixup_0.75_0.8/prediction/test/results_0.pkl', 'rb') as f:
                #     results = pickle.load(f)
                # with open('./results_debug/two_best_frame32_train/prediction/test/prob_5x.pkl', 'rb') as f:
                #     all_prob['m2'] = pickle.load(f)
                new_cfg = deepcopy(cfg)
                dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                        epoch=epoch, global_step=global_step, 
                                        generate_cfg=cfg['testing']['cfg'],
                                        save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                all_prob['start'] = name_prob

            elif args.eval_setting in ['central_random_1', 'central_random_2', '5x_random_1', '5x_random_2']:
                new_cfg = deepcopy(cfg)
                new_cfg['data']['transform_cfg']['from64'] = 'random'

                if 'central' in args.eval_setting:
                    test_p = ['central']
                    test_m = ['pad']
                else:
                    test_p = ['left_mid', 'right_mid', 'start', 'end', 'central']
                    test_m = ['pad', 'pad', 'pad', 'pad', 'pad']
                
                all_prob = {}
                for t_p, t_m in zip(test_p, test_m):
                    logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                    temp_cfg = deepcopy(new_cfg)
                    temp_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                    temp_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                    dataloader, sampler = build_dataloader(temp_cfg, split, is_train=False, val_distributed=False)

                    times = int(args.eval_setting.split('_')[-1])
                    for i in range(times):
                        per_ins_stat, per_cls_stat, results, name_prob, _ = evaluation(model=model.module, val_dataloader=dataloader, cfg=temp_cfg, 
                                                epoch=epoch, global_step=global_step, 
                                                generate_cfg=cfg['testing']['cfg'],
                                                save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
                        all_prob[t_p+'_'+str(i)] = name_prob
            
            if len(cfg['data']['input_streams']) == 1:
                if type(cfg['data']['num_output_frames']) == int:
                    logits_name_lst = ['']
                else:
                    logits_name = ['frame_ensemble_', '32_', '64_']
            elif len(cfg['data']['input_streams']) == 4:
                logits_name_lst = ['ensemble_last_', 'ensemble_all_']
            else:
                logits_name_lst = ['ensemble_last_', 'fuse_']
            
            evaluation_results = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], 
                all_prob=all_prob, eval_setting=args.eval_setting)
            for logits_name in logits_name_lst:
                logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_1']))
                logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_5']))
                logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_10']))

                # one class missing in the test set of WLASL_2000
                logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_1']))
                logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_5']))
                logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_10']))
                logger.info('-------------------------Evaluation Finished-------------------------')
            
            # if cfg['data']['dataset_name'] == 'WLASL_2000':
            #     other_vocab = [1000, 300, 100]
            # elif cfg['data']['dataset_name'] == 'MSASL_1000':
            #     other_vocab = [500, 200, 100]
            # else:
            #     other_vocab = []
            # for o in other_vocab:
            #     name_lst = dataloader.dataset.other_vars[str(o)][split]
            #     effective_label_idx = dataloader.dataset.other_vars[str(o)]['vocab_idx']
            #     evaluation_results = compute_accuracy(results, logits_name_lst, cls_num, cfg['device'], name_lst, 
            #         effective_label_idx, all_prob, args.eval_setting)
            #     logger.info('-------------------------Variant: {:d}-------------------------'.format(o))
            #     for logits_name in logits_name_lst:
            #         logger.info('-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_1']))
            #         logger.info('-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_5']))
            #         logger.info('-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_ins_top_10']))

            #         # one class missing in the test set of WLASL_2000
            #         logger.info('-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_1']))
            #         logger.info('-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_5']))
            #         logger.info('-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------'.format(logits_name, 100*evaluation_results[logits_name]['per_cls_top_10']))
            #         logger.info('-------------------------Evaluation Finished-------------------------')
