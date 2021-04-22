import torch
import operator
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from .models import SoftMaskedBertModel
from .optim_schedule import ScheduledOptim
from .utils import compute_corrector_prf, compute_sentence_level_prf
import tqdm


class Trainer:
    """
    Trainer is used for SoftMaskBert.

    """

    def __init__(self, model_spec:SoftMaskedBertModel, tokenizer, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for SoftMaskBERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:"+str(cuda_devices[0]) if cuda_condition else "cpu")

        # Setting the SoftMaskBert training model
        self.model = model_spec

        # Setting tokenizer
        self.tokenizer = tokenizer
        

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for SoftMaskedBERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        total_detect_correct = 0

        total_results = []
        total_det_acc_labels = []
        total_cor_acc_labels = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            ori_text, cor_text, det_labels = data
            encoded_ori_texts = self.tokenizer(ori_text, padding=True, return_tensors='pt')
            encoded_ori_texts.to(self.device)
            encoded_cor_text = self.tokenizer(cor_text, padding=True, return_tensors='pt')
            encoded_cor_text = encoded_cor_text.to(self.device)
            det_labels = det_labels.to(self.device)
            # print('encoded_ori_texts size is ', encoded_ori_texts['input_ids'].size())
            # print('encoded_cor_text size is ', encoded_cor_text['input_ids'].size())
            # print('det_labels size is ', det_labels.size())


            # 1. forward the SoftMaskedBert model
            # print("encoded_ori_texts device:", encoded_ori_texts['input_ids'].device)
            # print("encoded_cor_text device:", encoded_cor_text['input_ids'].device)
            # print("det_labels device:", det_labels.device)
            outputs = self.model.forward(encoded_ori_texts, encoded_cor_text['input_ids'], det_labels)

            # 2. calculate loss 
            loss = 0.8 * outputs[1] + 0.2 * outputs[0]

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
        
            # output of detector model 
            det_y_hat = (outputs[2] > 0.5).long()
            cor_y_hat = torch.argmax((outputs[3]), dim=-1)
            cor_y = encoded_cor_text['input_ids']
            cor_y_hat *= encoded_cor_text['attention_mask']

            results = []
            det_acc_labels = []
            cor_acc_labels = []
            for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
                _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
                _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
                _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
                cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
                det_acc_labels.append(det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
                results.append((_src, _tgt, _predict,))

            # outputs: loss.cpu().item(), det_acc_labels, cor_acc_labels, results
            avg_loss += loss.item()
            total_correct += sum(cor_acc_labels)
            total_detect_correct += sum(det_acc_labels)
            total_element += len(ori_text)
            total_results += results
            total_det_acc_labels += det_acc_labels
            total_cor_acc_labels += cor_acc_labels
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_cor_acc": total_correct / total_element* 100,
                "avg_det_acc": total_detect_correct / total_element*100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter),
                "total_cor_acc=", total_correct* 100.0 / total_element,
                "total_detect_acc=", total_detect_corerct* 100.0 / total_element)
        if train is False:
            compute_corrector_prf(total_results)
            compute_sentence_level_prf(total_results)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
