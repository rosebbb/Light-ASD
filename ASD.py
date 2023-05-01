import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from loss import lossAV, lossV
from model.Model import ASD_Model

class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()        
        self.model = ASD_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))
        if 'fixed_input_length' in kwargs:
            self.fixed_input_length = kwargs['fixed_input_length']
            
    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, lossV, lossAV, loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            if visualFeature.size(dim=1) == 0:
            # print('audioFeature.shape, visualFeature.shape:', audioFeature.shape, visualFeature.shape, labels.shape)
                print(f'{num} skipped because videoFeature size is 0')
                continue
            if audioFeature.size(dim=1) == 0:
            # print('audioFeature.shape, visualFeature.shape:', audioFeature.shape, visualFeature.shape, labels.shape)
                print(f'{num} skipped because audioFeature size is 0')
                continue
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())

            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsV = self.model.forward_visual_backend(visualEmbed)

            if self.fixed_input_length != 0:
                labels = labels[0][:, -5:]
                num_clips, frames_per_clip = labels.shape
                labels = labels.reshape((-1)).cuda() # Loss

                slice_index = torch.arange(20, 25).cuda()
                for i in range(1, num_clips):
                    slice_index = torch.hstack((slice_index, torch.arange((i+1)*25-5, (i+1)*25).cuda()))
                outsAV = torch.index_select(outsAV, 0, slice_index)
                outsV = torch.index_select(outsV, 0, slice_index)
            else:
                labels = labels[0].reshape((-1)).cuda() # Loss

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
            nlossV = self.lossV.forward(outsV, labels, r)
            nloss = nlossAV + 0.5 * nlossV

            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossV/(num), lossAV/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                # break
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss,
            }, path)
        
    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)['model_state_dict']
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)