"""
@Time   :   2021-01-12 15:23:56
@File   :   main.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import os
import torch
from transformers import BertTokenizer
from src.dataset import get_corrector_loader
from src.models import SoftMaskedBertModel
from src.data_processor import preproc
from src.utils import get_abs_path
from src.trainer import Trainer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='gpu', type=str, help="硬件，cpu or cuda")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="是否加载训练保存的权重, one of [t,f]")
    parser.add_argument('--bert_checkpoint', default='bert-base-chinese', type=str)
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=16, type=int, help='批大小')
    parser.add_argument('--warmup_epochs', default=8, type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--accumulate_grad_batches',
                        default=16,
                        type=int,
                        help='梯度累加的batch数')
    parser.add_argument('--mode', default='train', type=str,
                        help='代码运行模式，以此来控制训练测试或数据预处理，one of [train, test, preproc]')
    parser.add_argument('--loss_weight', default=0.8, type=float, help='论文中的lambda，即correction loss的权重')
    arguments = parser.parse_args()
    if arguments.hard_device == 'cpu' and torch.cuda.is_available() is False:
        arguments.device = torch.device(arguments.hard_device)
    else:
        arguments.device = torch.device(f'cuda:{arguments.cuda_devices[0]}')
    if not 0 <= arguments.loss_weight <= 1:
        raise ValueError(f"The loss weight must be in [0, 1], but get{arguments.loss_weight}")
    print(arguments)
    return arguments


def main():
    args = parse_args()
    if args.mode == 'preproc':
        print('preprocessing...')
        preproc()
        return
    # torch.backends.cudnn.enabled = False
    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)

    print("Building SoftMaskedBert model")
    model = SoftMaskedBertModel(args,tokenizer.mask_token_id).to(args.device)
    print("Creating train dataloader")
    train_loader = get_corrector_loader(get_abs_path('data', 'train.json'),
                                        tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4)
    print("Creating valid dataloader")
    valid_loader = get_corrector_loader(get_abs_path('data', 'dev.json'),
                                        tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)
    print("Creating test dataloader")
    test_loader = get_corrector_loader(get_abs_path('data', 'test.json'),
                                       tokenizer,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)
    
    print("Loading Bert pretrain model")
    model.load_from_transformers_state_dict(get_abs_path('checkpoint', 'pytorch_model.bin'))
    if args.load_checkpoint:
        print("Loading checkpoint")
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'),
                                         map_location=args.hard_device))
    print("Creating SoftMaskBert Trainer")
    if args.mode == 'test':
        model.load_state_dict(
            torch.load(get_abs_path('checkpoint', 'softmaskedBert_model.bin'), map_location=args.hard_device)) 
    trainer = Trainer(model, tokenizer, train_dataloader=train_loader, test_dataloader=test_loader,
                        lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                        with_cuda=False if args.hard_device == 'cpu' else True,
                        cuda_devices=None if args.hard_device == 'cpu' else args.cuda_devices,
                        log_freq=100)
    if args.mode == 'train':
        for epoch in range(args.epochs):
            trainer.train(epoch)
            trainer.save(epoch, os.path.join(args.model_save_path, 'softmaskedBert_model.bin'))
            print('model saved.')

            if test_loader is not None:
                trainer.test(test_loader)
    else:
        
        trainer.test(test_loader)


if __name__ == '__main__':
    main()
