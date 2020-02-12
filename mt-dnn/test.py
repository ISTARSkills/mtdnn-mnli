# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from pytorch_pretrained_bert.tokenization import BertTokenizer
import argparse
from data_utils.task_def import DataFormat
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pytorch_pretrained_bert.modeling import BertConfig
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from data_utils.task_def import TaskType, EncoderModelType
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler, RunTimeDataset
from mt_dnn.model import MTDNNModel


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_feature_extractor(
        text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
    tokens_a = tokenize_fn.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenize_fn.tokenize(text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    if tokens_b:
        input_ids = tokenize_fn.convert_tokens_to_ids(
            ['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * (len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
    else:
        input_ids = tokenize_fn.convert_tokens_to_ids(
            ['[CLS]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
    input_mask = None
    return input_ids, input_mask, segment_ids


def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=5)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument("--init_checkpoint", default='mt_dnn_models/bert_model_base.pt', type=str)
    parser.add_argument('--data_dir', default='data/canonical_data/bert_uncased_lower')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument('--train_datasets', default='mnli')
    parser.add_argument('--test_datasets', default='mnli_mised,mnli_matched')
    parser.add_argument('--glue_format_on', action='store_true')
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=False,
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str)
    parser.add_argument("--resume", action='store_true')

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    # fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    return parser


parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')
pprint(args)

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

task_defs = TaskDefs(args.task_def)
encoder_type = task_defs.encoderType
args.encoder_type = encoder_type

tokenizer = test_collater = model = None


def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 3:
        opt_v = max_opt
    return opt_v


def main():
    global tokenizer, test_collater, model
    logger.info('Launching the MT-DNN training')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size
    tasks = {}
    tasks_class = {}
    nclass_list = []
    decoder_opts = []
    task_types = []
    dropout_list = []
    loss_types = []
    kd_loss_types = []

    #train_datasets = []
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks: continue
        assert prefix in task_defs.n_class_map
        assert prefix in task_defs.data_type_map
        data_type = task_defs.data_type_map[prefix]
        nclass = task_defs.n_class_map[prefix]
        task_id = len(tasks)
        if args.mtl_opt > 0:
            task_id = tasks_class[nclass] if nclass in tasks_class else len(tasks_class)

        task_type = task_defs.task_type_map[prefix]

        dopt = generate_decoder_opt(task_defs.enable_san_map[prefix], opt['answer_opt'])
        if task_id < len(decoder_opts):
            decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
        else:
            decoder_opts.append(dopt)
        task_types.append(task_type)
        loss_types.append(task_defs.loss_map[prefix])
        kd_loss_types.append(task_defs.kd_loss_map[prefix])

        if prefix not in tasks:
            tasks[prefix] = len(tasks)
            if args.mtl_opt < 1: nclass_list.append(nclass)

        if (nclass not in tasks_class):
            tasks_class[nclass] = len(tasks_class)
            if args.mtl_opt > 0: nclass_list.append(nclass)

        dropout_p = task_defs.dropout_p_map.get(prefix, args.dropout_p)
        dropout_list.append(dropout_p)

        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        logger.info('Loading {} as task {}'.format(train_path, task_id))
        # train_data_set = SingleTaskDataset(train_path, True, maxlen=args.max_seq_len, task_id=task_id,
        #                                    task_type=task_type, data_type=data_type)
        # train_datasets.append(train_data_set)
    #train_collater = Collater(dropout_w=args.dropout_w, encoder_type=encoder_type)
    # multi_task_train_dataset = MultiTaskDataset(train_datasets)
    # multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio)
    # multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler,
    #                                    collate_fn=train_collater.collate_fn, pin_memory=args.cuda)

    opt['answer_opt'] = decoder_opts
    opt['task_types'] = task_types
    opt['tasks_dropout_p'] = dropout_list
    opt['loss_types'] = loss_types
    opt['kd_loss_types'] = kd_loss_types

    args.label_size = ','.join([str(l) for l in nclass_list])
    logger.info(args.label_size)
    dev_data_list = []
    test_data_list = []
    test_collater = Collater(is_train=False, encoder_type=encoder_type)

    logger.info('#' * 20)
    logger.info(opt)
    logger.info('#' * 20)

    bert_model_path = 'checkpoints/my_mnli/model_0.pt'
    state_dict = None

    if encoder_type == EncoderModelType.BERT:
        if os.path.exists(bert_model_path):
            state_dict = torch.load(bert_model_path, map_location=torch.device('cpu'))
            config = state_dict['config']
            config['attention_probs_dropout_prob'] = args.bert_dropout_p
            config['hidden_dropout_prob'] = args.bert_dropout_p
            config['multi_gpu_on'] = opt["multi_gpu_on"]
            opt.update(config)
        else:
            logger.error('#' * 20)
            logger.error('Could not find the init model!\n The parameters will be initialized randomly!')
            logger.error('#' * 20)
            config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
            config['multi_gpu_on'] = opt["multi_gpu_on"]
            opt.update(config)
    elif encoder_type == EncoderModelType.ROBERTA:
        bert_model_path = '{}/model.pt'.format(bert_model_path)
        if os.path.exists(bert_model_path):
            new_state_dict = {}
            state_dict = torch.load(bert_model_path)
            for key, val in state_dict['model'].items():
                if key.startswith('decoder.sentence_encoder'):
                    key = 'bert.model.{}'.format(key)
                    new_state_dict[key] = val
                elif key.startswith('classification_heads'):
                    key = 'bert.model.{}'.format(key)
                    new_state_dict[key] = val
            state_dict = {'state': new_state_dict}

    model = MTDNNModel(opt, state_dict=state_dict)
    if args.resume and args.model_ckpt:
        logger.info('loading model from {}'.format(args.model_ckpt))
        model.load(args.model_ckpt)

    #### model meta str
    headline = '############# Model Arch of MT-DNN #############'
    ### print network
    logger.info('\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))

    logger.info("Total number of params: {}".format(model.total_param))

    # # tensorboard
    # if args.tensorboard:
    #     args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
    #     tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def cluster_sentence(premise, hypothesis, tokenizer, test_collater, model):
    # print('Enter Sentence 1:')
    # premise = input()
    # print('Enter Sentence 2:')
    # hypothesis = input()

    prefix = 'mnli'
    input_ids, _, type_ids = bert_feature_extractor(premise, hypothesis, max_seq_length=64, tokenize_fn=tokenizer)
    features = [{'uid': '0', 'label': '0', 'token_id': input_ids, 'type_id': type_ids}]
    dev_data_set = RunTimeDataset(features, None, False, maxlen=args.max_seq_len, task_id=0,
                                  task_type=TaskType.Classification,
                                  data_type=DataFormat.PremiseAndOneHypothesis)
    dev_data = DataLoader(dev_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn,
                          pin_memory=args.cuda)
    # dev_data_list = [dev_data]

    args.cuda = False
    with torch.no_grad():
        label_dict = task_defs.global_map.get('mnli', None)
        pred = eval_model(model,
                          dev_data,
                          metric_meta=task_defs.metric_meta_map[
                              prefix],
                          use_cuda=args.cuda,
                          label_mapper=label_dict,
                          with_label=False,
                          task_type=task_defs.task_type_map[
                              prefix])
        return '{"label": ' + pred + '}'


from flask import Flask, request, jsonify, Response, render_template

app = Flask(__name__)


@app.route("/sentence_similarity_1", methods=["POST", "GET"])
def sentence_similarity():
    signal = snippet = None
    matches = []
    if 'signal' in request.args:
        signal = request.args['signal']
    if 'snippet' in request.args:
        snippet = request.args['snippet']
    else:
        use_threshold = None
    global tokenizer, test_collater, model
    matches = cluster_sentence(snippet, signal, tokenizer, test_collater, model)
    resp = Response(matches, mimetype='application/json')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    main()
    app.run(host="0.0.0.0", port="9998", debug=True)
