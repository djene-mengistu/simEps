import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # specify which GPU(s) to be used

from datetime import datetime
from distutils.dir_util import copy_tree
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle

# import torch.backends.cudnn as cudnn
# import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from utilities.dataloaders import*
# from new_dataloaders import*
# from loss import loss_sup, loss_adversarial_1, loss_adversarial_2, make_Dis_label, gt_label, loss_diff
from utilities.metrics import*
from utilities.losses_1 import*
from utilities.losses_2 import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from ICT_model import model, ema_model #, model2 #, Critic_model
from utilities.utilities import get_logger, create_dir


import os
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30250, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
 

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')

parser.add_argument('--ict_alpha', type=int, default=0.1,
                    help='ict_alpha')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU id's, GPU id's start from 0.


epochs = 600
# batchsize = 16 
# CE = torch.nn.BCELoss()
# criterion_1 = torch.nn.BCELoss()
num_classes = args.num_classes

ce_loss = CrossEntropyLoss()
# dice_loss = 1 - mDice(pred_mask, mask)
base_lr = args.base_lr
max_iterations = args.max_iterations

iter_per_epoch = 60

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model = model
        self.ema_model = ema_model 
        self._init_logger()

    # def _init_configure(self):
    #     with open('configs/config.yml') as fp:
    #         self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = '/.../model_weights/NEU_seg/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        # self.image_save_path_1 = log_dir + "/saved_images_1"
        # self.image_save_path_2 = log_dir + "/saved_images_2"

        # create_dir(self.image_save_path_1)
        # create_dir(self.image_save_path_2)

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        # print('Generator Learning Rate: {} Critic Learning Rate:{}'.format(lr_gen,lr_dis))
        # print('Generator Learning Rate: {}'.format(lr_gen))

        # self.model_1.to(device)
        # self.model_2.to(device)
        self.model.to(device)
        # self.model_2.to(device)

        # params = self.model_1.parameters() #+ list(self.model_2.parameters())
        # dis_params = self.critic.parameters() #Turn off to train w/o the discriminator
        # optimizer = torch.optim.SGD(params, lr=lr_gen, momentum=momentum)
        # optimizer_1 = optim.SGD(self.model.parameters(), lr=base_lr,
        #                   momentum=0.9, weight_decay=0.0001)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        # optimizer_2 = torch.optim.Adam(self.model2.parameters(), lr=base_lr)
        # # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=20, verbose=True)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.0000001, patience=40, verbose=True)
        # optimizer_2 = optim.SGD(self.model_2.parameters(), lr=base_lr,
        #                   momentum=0.9, weight_decay=0.0001)
        # optimizer = torch.optim.Adam(params, lr=lr_gen)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=20, verbose=True)
      
        
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {}".format(len(train_loader),
                                                                       len(unlabeled_loader),
                                                                       len(val_loader)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(1, epochs):

            running_ce_loss = 0.0
            running_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_iou_1 = 0.0
            running_train_dice_1 = 0.0
            running_consistency_loss = 0.0
            
            running_val_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_ce_loss = 0.0
                        
            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            running_val_accuracy_1 = 0.0
            
            optimizer_1.zero_grad()
            # optimizer_2.zero_grad()
            
            self.model.train()
            # self.model_2.train()

            semi_dataloader = iter(zip(cycle(train_loader), unlabeled_loader))
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                
                data = next(semi_dataloader)
                
                (inputs_S1, labels_S1), (inputs_U, labels_U) = data #data[0][0], data[0][1]


                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)

                inputs_U, labels_U = Variable(inputs_U), Variable(labels_U)
                inputs_U, labels_U = inputs_U.to(device), labels_U.to(device)
                
                # ICT mix factors
                ict_mix_factors = np.random.beta(args.ict_alpha, args.ict_alpha, size=(args.batch_size//2, 1, 1, 1))
                ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).cuda()
                unlabeled_batch_0 = inputs_U[0:args.batch_size//2, ...]
                unlabeled_batch_1 = inputs_U[args.batch_size//2:, ...]

                # Mix images
                batch_ux_mixed = unlabeled_batch_0 *(1.0 - ict_mix_factors) + unlabeled_batch_1 * ict_mix_factors #reduced batch size by halftmux
                
                # input_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)

                self.model.train()
                # self.model_2.train()

                # Train Model 1
                outputs_1 = self.model(inputs_S1)
                outputs_soft_1 = torch.softmax(outputs_1, dim=1)
                
                un_outputs_1 = self.model(batch_ux_mixed) #Use the mixed input instead of the original input_U
                un_outputs_soft_1 = torch.softmax(un_outputs_1, dim=1)

                # noise = torch.clamp(torch.randn_like(inputs_U) * 0.1, -0.2, 0.2)
                # ema_inputs = inputs_U + noise

                
                with torch.no_grad():
                    # ema_output = self.ema_model(ema_inputs)
                    ema_output_ux0 = torch.softmax(self.ema_model(unlabeled_batch_0), dim=1)
                    ema_output_ux1 = torch.softmax(self.ema_model(unlabeled_batch_1), dim=1)
                    batch_pred_mixed = ema_output_ux0 *(1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
                    
                    # ema_output_soft = torch.softmax(ema_output, dim=1)

                loss_ce_1 = ce_loss(outputs_1, labels_S1.long())
                loss_dice_1 = dice_loss(labels_S1.unsqueeze(1), outputs_1)
                
                loss_sup = 0.5 * (loss_dice_1 + loss_ce_1)

                consistency_weight = get_current_consistency_weight(iter_num//150)
                
                consistency_loss = torch.mean((un_outputs_soft_1 - batch_pred_mixed)**2) #MSE loss between teacher and student output
                                
                loss = loss_sup + consistency_weight * consistency_loss
             
                # seg_loss = seg_loss / self.accumulation_steps
                optimizer_1.zero_grad()
                
                loss.backward()

                # if (i + 1 ) % self.accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer_1.step()
                # optimizer_2.step()
                # optimizer.zero_grad()
                running_train_loss += loss.item()
                running_ce_loss += loss_ce_1.item()
                running_dice_loss += loss_dice_1.item()
                running_consistency_loss += consistency_loss.item()

                running_train_iou_1 += mIoU(outputs_1, labels_S1)
                running_train_dice_1 += mDice(outputs_1, labels_S1)

                update_ema_variables(self.model, self.ema_model, args.ema_decay, iter_num)

                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']

                
                iter_num = iter_num + 1
            
            epoch_loss = (running_train_loss) / (iter_per_epoch)
            epoch_ce_loss = (running_ce_loss) / (iter_per_epoch)
            epoch_dice_loss = (running_dice_loss) / (iter_per_epoch)
            epoch_train_iou = (running_train_iou_1) / (iter_per_epoch)
            epoch_train_dice = (running_train_dice_1) / (iter_per_epoch)
            epoch_consistency_loss = (running_consistency_loss) / (iter_per_epoch)
            # epoch_loss = running_loss / (len(train_loader))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train dice: {}'.format(epoch_train_dice))
            self.writer.add_scalar('Train/mDice', epoch_train_dice, epoch)
            self.logger.info('Train IoU: {}'.format(epoch_train_iou))
            self.writer.add_scalar('Train/mIoU', epoch_train_iou, epoch)

            self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
            self.writer.add_scalar('Train/consistency-Loss', epoch_consistency_loss, epoch)

            self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            torch.cuda.empty_cache()

            self.model.eval()
            # self.model_2.eval()
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    # images = Variable(images)
                    # gts = Variable(gts)
                    images = images.to(device)
                    gts = gts.to(device)
                    
                    prediction_1 = self.model(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)

                        

                # dice_coe_1 = dice_coef(prediction_1, gts)
                loss_ce_1 = ce_loss(prediction_1, gts.long())
                loss_dice_1 = 1 - mDice(prediction_1, gts)
                # loss_ce = loss_ce_1 + loss_ce_2
                # loss_dice = loss_dice_1 + loss_dice_2

                val_loss = 0.5 * (loss_dice_1 + loss_ce_1)

                running_val_loss += val_loss
                running_val_dice_loss += loss_dice_1
                running_val_ce_loss += loss_ce_1
                
                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)

                # running_val_iou_2 += mIoU(prediction_2, gts)
                # running_val_accuracy_2 += pixel_accuracy(prediction_2, gts)
                # running_val_dice_2 += mDice(prediction_2, gts)
                
                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_ce_loss_val = running_val_ce_loss / len(val_loader)
            epoch_dice_loss_val = running_val_dice_loss / len(val_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)
            scheduler_1.step(epoch_dice_val_1)

            # epoch_dice_val_2 = running_val_dice_2 / len(val_loader)
            # epoch_iou_val_2 = running_val_iou_2 / len(val_loader)
            # epoch_accuracy_val_2 = running_val_accuracy_2 / len(val_loader)
            # scheduler.step(epoch_dice_val_1)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)

            self.logger.info('Val CE-loss: {}'.format(epoch_ce_loss_val))
            self.writer.add_scalar('Val/CE-loss', epoch_ce_loss_val, epoch)

            self.logger.info('Val Dice-loss: {}'.format(epoch_dice_loss_val))
            self.writer.add_scalar('Val/Dice-loss', epoch_dice_loss_val, epoch)

            #model-1 perfromance
            self.logger.info('Val dice_1 : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Val/DSC-1', epoch_dice_val_1, epoch)

            self.logger.info('Val IoU_1 : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Val/IoU-1', epoch_iou_val_1, epoch)

            self.logger.info('Val Accuracy_1 : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Val/Accuracy-1', epoch_accuracy_val_1, epoch)
            
            self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            torch.cuda.empty_cache()

            
            
            mdice_coeff_1 =  epoch_dice_val_1
            # mdice_coeff_2 =  epoch_dice_val_2
            # mval_loss_1 = epoch_val_loss

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True

                # if not os.path.exists(self.image_save_path_1):
                #     os.makedirs(self.image_save_path_1)

                # copy_tree(self.image_save_path_1, self.save_path + '/best_model_predictions_1')
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1


                        
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                torch.save(state_1, Checkpoints_Path + '/ICT_10p.pth')
  
 
            
            
             
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))

            self.logger.info('current patience :{}'.format(self.patience))
            print('Current consistency weight:', consistency_weight)
            print('================================================================================================')
            print('================================================================================================')




if __name__ == '__main__':
    train_network = Network()
    train_network.run()