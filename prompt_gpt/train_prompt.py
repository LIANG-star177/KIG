# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: train.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:28
"""
    文件说明：
    通过新闻正文生成新闻标题的GPT2模型的训练文件
"""

import torch
import os
import random
import numpy as np
import argparse
import logging
# from distill_tuning import Distill_Tuning
from dialog_tuning import Distill_Tuning
from transformers import BertTokenizer
from dataset import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config
from tqdm import tqdm, trange
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, train_data, test_data, args):
    tb_write = SummaryWriter()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    # 获取模型所有参数
    params = [{'params': model.prompt_encoder.parameters(), 'lr':args.prompt_lr}]
    if args.use_lm_finetune:
        params.append({'params': model.model.parameters(), 'lr': args.finetune_lr})

    optimizer = AdamW(params, eps=args.adam_epsilon, weight_decay= 0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 清空cuda缓存
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    model.train()
    title_id = train_data.title_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            claim_label2 = batch["claim_label2"].to(device)
            # 获取训练结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,\
                                    title_id=title_id, claim_label2 = claim_label2)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                global_step += 1
                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss-logging_loss) /
                                        (args.logging_steps*args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss
                # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, device, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    model.train()
        # 每个epoch进行完，则保存模型
        prompt_output_dir = os.path.join(args.prompt_output_dir, "checkpoint-{}".format(global_step))
        torch.save(model.prompt_encoder.state_dict(),prompt_output_dir)
        if args.use_lm_finetune:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            model.model.save_pretrained(output_dir)
        # model_to_save = model.module if hasattr(model, "module") else model
        # model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        # torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, device, test_data, args):
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    title_id = test_data.title_id
    total_loss, total = 0.0, 0.0
    # 进行测试
    for step, batch in enumerate(iter_bar):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            claim_label2 = batch["claim_label2"].to(device)
            # 获取预测结果
            outputs = model.forward(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids,\
                                    title_id=title_id, claim_label2 = claim_label2)
            loss = outputs[0]
            loss = loss.item()
            # 对loss进行累加
            total_loss += loss*len(batch["input_ids"])
            total += len(batch["input_ids"])
    # 计算最终测试集的loss结果
    test_loss = total_loss / total
    return test_loss


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--config_path', default='new_gpt/config.json', type=str, help='模型参数配置信息')
    parser.add_argument('--vocab_path', default='new_gpt/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--train_file_path', default='data/new_split/train.json', type=str, help='新闻标题生成的训练数据')
    parser.add_argument('--test_file_path', default='data/new_split/test.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--pretrained_model_path', default="log/checkpoint-41680", type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--data_dir', default='data/new_split', type=str, help='生成缓存数据的存放路径')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='模型训练的轮数')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练时每个batch的大小')
    parser.add_argument('--test_batch_size', default=8, type=int, help='测试时每个batch的大小')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--eval_steps', default=1000, type=int, help='训练时，多少步进行一次测试')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--seed', type=int, default=2022, help='随机种子')
    parser.add_argument('--max_len', type=int, default=900, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--title_max_len', type=int, default=300, help='生成标题的最大长度，要比max_len小')

    #prompt args
    parser.add_argument("--multi_prompt", type=bool, default=True, help='是否多维prompt')
    parser.add_argument("--context_aware", type=bool, default=True, help='是否注意上下文')
    parser.add_argument("--use_lm_finetune", type=bool, default=False, help='是否finetune')
    parser.add_argument('--prompt_lr', default=1e-4, type=float, required=False, help='学习率')
    parser.add_argument('--finetune_lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--template_len', default=5, type=int, required=False,help='prompt长度')
    parser.add_argument("--pseudo_token", type=str, default='##🔥')
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument('--output_dir', default='log/ca_p_finetune_saved_model', type=str, help='模型输出路径')
    parser.add_argument('--prompt_output_dir', default='log/ca_multi_mean_prompt_saved_model', type=str, help='模型输出路径')
    return parser.parse_args()


def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda:0" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    args.device = device
    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config
    model = Distill_Tuning(args)
    print(count_parameters(model))
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.prompt_output_dir):
        os.mkdir(args.prompt_output_dir)
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    test_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练
    train(model, device, train_data, test_data, args)


if __name__ == '__main__':
    main()

