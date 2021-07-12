import torch
import torch.nn as nn
import torch.nn.functional as F
import shap

from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import InputXGradient
from captum.attr import GuidedBackprop
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import FeatureAblation
from captum.attr import GuidedGradCam
from captum.attr import NoiseTunnel
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution

from torchcam.cams import GradCAMpp
from torchcam.cams import SmoothGradCAMpp

from torchvision.transforms.functional import resize

import numpy as np
import copy
import os,sys,time,datetime
import math
import random
import re

import argparse

def infoLine(message,infoType="info"):
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)

    if infoType == "ERROR":
        sys.exit()
    #
    sys.stdout.flush()
#

def getUniqueRunID():
    import hashlib,time,os
    mix = str(time.time()) + "|" + str(os.uname()) + "|" + str(os.getpid())
    #print(hashlib.md5(mix.encode()).hexdigest()[-8:])
    return hashlib.md5(mix.encode()).hexdigest()[-8:]
#


class Net(nn.Module):
    def __init__(self, paramHash):
        super(Net, self).__init__()
        # N*64*144*144
        self.conv1    = nn.Conv2d(1, 64, kernel_size=5, padding=2)       # N*64*144*144
        self.mp1      = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # N*64*72*72
        self.bn1      = nn.BatchNorm2d(64)
        self.relu1    = nn.ReLU()
        
        # N*64*72*72
        self.conv2    = nn.Conv2d(64, 128, kernel_size=5, padding=2)      # N*128*72*72
        self.mp2      = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # N*128*36*36
        self.bn2      = nn.BatchNorm2d(128)
        self.relu2    = nn.ReLU()
        
        # N*128*36*36
        self.conv3    = nn.Conv2d(128, 256, kernel_size=3, padding=1)     # N*256*36*36
        self.mp3      = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # N*256*18*18
        self.bn3      = nn.BatchNorm2d(256)
        self.relu3    = nn.ReLU()

        # N*256*18*18
        self.gap      = nn.AdaptiveAvgPool2d(1)                           # N*256*1*1
        self.bn       = nn.BatchNorm2d(256)                               # N*256*1*1
        self.relu     = nn.ReLU()
        
        self.fc       = nn.Linear(256*1*1, paramHash["data_class_num"])
        
        self.drop     = nn.Dropout2d(p=0.25, inplace=False)
        
        

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu1(self.bn1(self.mp1(self.conv1(x))))
        x = self.relu2(self.bn2(self.mp2(self.conv2(x))))
        x = self.relu3(self.bn3(self.mp3(self.conv3(x))))
        x = self.drop(x)
        x = self.relu(self.bn(self.gap(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        #print(x.size())
        return x
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
def readRawData(paramHash, dataType, maxline = -1):
    def onehot(classNum,index):
        datalist = np.zeros(classNum)
        datalist[index] = 1.0
        return datalist
    #

    classNum = paramHash["data_class_num"]
    geneNum  = paramHash["data_gene_num"]
    size     = math.ceil( math.sqrt(paramHash["data_gene_num"]) )
    
    # calculate offset for square
    while (size % 8) > 0:
        size = size  +1
    #
    offsetNum= size * size - geneNum
    offsetData = np.zeros(offsetNum)

    if paramHash["run_mode"] == "train" and paramHash["data_upsample"] and dataType == "train":
        # read data
        dataByLabelHash = {}
        with open(paramHash[dataType]["infile"], "rt") as fi:
            for line in fi:
                line = line.rstrip()
                row = line.split("\t")
                #label = onehot( classNum, int(row[1]) )
                label = int(row[1])
                
                data = np.array([np.concatenate( ( np.array([float(k) for k in row[2].split("|")]), offsetData) ).reshape(size,size),]) # shape: 1 x size x size

                if label not in dataByLabelHash:
                    dataByLabelHash[label] = []
                #
                dataByLabelHash[label].append(data)
        #
        
        # estimate up sampling
        maxNum = max( [ len(dataByLabelHash[label]) for label in dataByLabelHash ] )
        upsampleSize = maxNum * len( dataByLabelHash )
        paramHash["data_singleClassDataNum"] = maxNum
        
        # update param
        paramHash[dataType]["data_size"] = upsampleSize
        paramHash[dataType]["max_batch_num"] = math.ceil( paramHash[dataType]["data_size"] / paramHash["net_batch_size"] )
        paramHash[dataType]["current_index"] = -1
        
        # return data
        return dataByLabelHash

    else: ##### regular way to read data
        datalist = []
        with open(paramHash[dataType]["infile"], "rt") as fi:
            for line in fi:
                line = line.rstrip()
                row = line.split("\t")
                #label = onehot( classNum, int(row[1]) )
                label = int(row[1])

                data = np.array([np.concatenate( ( np.array([float(k) for k in row[2].split("|")]), offsetData) ).reshape(size,size),]) # shape: 1 x size x size

                datalist.append( (data, label) )
                maxline = maxline - 1
                if maxline == 0:
                    break
        #

        # update param
        paramHash[dataType]["data_size"] = len(datalist)
        paramHash[dataType]["max_batch_num"] = math.ceil( paramHash[dataType]["data_size"] / paramHash["net_batch_size"] )
        paramHash[dataType]["current_index"] = -1
    
        # return data    
        return datalist
#

def upsampleData(dataHash, paramHash, doUpsample = True):
    dataType = "train"
    if "original" not in dataHash:
        dataHash["original"] = copy.deepcopy( dataHash[dataType] )
    #
    
    datalist = []
    for label in dataHash["original"]:
        poolData = []
        poolData.extend( dataHash["original"][label] )
        
        if doUpsample:        
            if len(poolData) < paramHash["data_singleClassDataNum"]:
                poolData.extend( random.choices(dataHash["original"][label], k = paramHash["data_singleClassDataNum"] - len(poolData) ) )
        #
        
        for data in poolData: # size x size or 1 x size x size
            datalist.append( (data, label) )
    #
    # shuffle data
    random.shuffle( datalist )
    dataHash[dataType] = []
    dataHash[dataType] = datalist
    
    # update parameters
    paramHash[dataType]["data_size"] = len(dataHash[dataType])
    paramHash[dataType]["max_batch_num"] = math.ceil( paramHash[dataType]["data_size"] / paramHash["net_batch_size"] )
    paramHash[dataType]["current_index"] = -1
#

#
def getNextData(paramHash, dataHash, dataType, shuffle = True, addNoise = False):
    # shulffle data
    if paramHash[dataType]["current_index"] < 0 or paramHash[dataType]["current_index"] >= paramHash[dataType]["max_batch_num"]:
        if shuffle:
            random.shuffle(dataHash[dataType])
        paramHash[dataType]["current_index"] = 0
    #
    
    # calculate index
    batchSize = paramHash["net_batch_size"]
    index = paramHash[dataType]["current_index"] * batchSize
    
    # extract data
    data = np.array([ k[0] for k in dataHash[dataType][index:(index+batchSize)] ])
    label= np.array([ k[1] for k in dataHash[dataType][index:(index+batchSize)] ])
    #

    # move index
    paramHash[dataType]["current_index"] = paramHash[dataType]["current_index"] + 1
    
    # add noise
    if addNoise:
        noise = np.random.normal(0,paramHash["net_noise_sigma"], data.shape )
        p= np.random.random( data.shape[0] )
        p[np.where(p <  paramHash["net_data_noise"])] = 0.0
        p[np.where(p >= paramHash["net_data_noise"])] = 1.0
        p = p[:,None,None,None]
        data = np.absolute( data + noise * p )
    
    # return data
    return data, label
#


def train(model, paramHash, dataHash, optimizer, criterion):
    model.train()
    dataType = "train"
    running_loss_tmp = 0.0
    train_loss = 0.0
    for batch_idx in range(paramHash[dataType]["max_batch_num"]):
        noise = False
        if paramHash["net_data_noise"] > 0:
            noise = True
        #
        data,label = getNextData(paramHash, dataHash, dataType, shuffle = True, addNoise = noise)
        data  = torch.from_numpy(data).cuda()
        label = torch.from_numpy(label).cuda()
        #
        optimizer.zero_grad()
        digit = model(data)
        loss = criterion(digit, label)
        loss.backward()
        optimizer.step()
        running_loss_tmp += loss.data
    #
    train_loss = running_loss_tmp.cpu().numpy()
    return train_loss
#

def evaluate(model, paramHash, dataHash, dataType, criterion, prediction = False):
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    running_loss_tmp = 0.0
    
    predicted_save = np.array([])
    label_save = np.array([])
    digit_save = None
    
    with torch.no_grad():
        for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
            data,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
            data  = torch.from_numpy(data).cuda()
            label = torch.from_numpy(label).cuda()
            #
            digit = model(data)
            loss = criterion(digit, label)
            running_loss_tmp += loss.data
            _, predicted = torch.max(digit.data, 1)
            total += label.size(0)
            predicted_tmp = predicted.cpu().numpy()
            label_tmp = label.squeeze().data.cpu().numpy()
            
            correct += (predicted_tmp == label_tmp).sum()

            predicted_save = np.concatenate((predicted_save,predicted_tmp))
            label_save     = np.concatenate((label_save,    label_tmp))
            
            if prediction:
                digit_tmp = F.softmax(digit, dim=1).cpu().numpy()
                if digit_save is None:
                    digit_save = copy.deepcopy(digit_tmp)
                else:
                    digit_save     = np.concatenate((digit_save,    digit_tmp))
            #
    #
    valid_loss = running_loss_tmp.cpu().numpy()
    return label_save,predicted_save,valid_loss,correct,total,digit_save
#


def explain_shap(model, paramHash, dataHash, dataType, reference):
    if paramHash["net_explain_ref_shuffle"] == "yes":
        infoLine("Shuffling background data")
        random.shuffle(dataHash[reference])
    #

    attHash = {}
    labelHash = {}
    numSampling  = paramHash["net_explain_noiseTunnel"]
    
    if paramHash["net_explain_explainer"] == "DeepExplainerFactorize":
        for refIndex in range(paramHash["net_explain_ref_size"]):
            msg = "-------------- Reference: " + str(refIndex + 1) + "/" + str( paramHash["net_explain_ref_size"] ) + " -----------------"
            infoLine(msg)

            base_data = np.array( [ k[0] for k in dataHash[reference][refIndex:(refIndex+1)] ] )
            base_data = torch.from_numpy(base_data).cuda()
            explainer = shap.DeepExplainer(model, base_data)
            resetData(paramHash)
            
            # data index for each round explanation
            dataIndex = 0

            for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
                msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
                infoLine(msg)

                data_c,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
                labelList = list(label)

                if numSampling > 1:
                    tmpHash = {}
                    for ni in range(numSampling):
                        noise = np.random.normal(0,1.0, data_c.shape )
                        data = np.absolute( data_c + noise)
                        data = torch.from_numpy(data).cuda()
                        attScoreList = explainer.shap_values(data) # 82 * 4 * 1 * 144 * 144

                        for i in range(len(labelList)):
                            if i not in tmpHash:
                                tmpHash[i] = []
                            #
                            tmpHash[i].append( attScoreList[labelList[i]][i] )
                        #
                    #

                    for i in range(len(labelList)):
                        if dataIndex not in attHash:
                            label_value = labelList[i]
                            attHash[dataIndex] = []
                            labelHash[dataIndex]= str( label_value )
                        #
                        contrib = np.mean(np.array(tmpHash[i]), axis=0)
                        contrib = np.array( contrib.flatten() )
                        attHash[dataIndex].append( contrib )
                        dataIndex = dataIndex + 1
                else:
                    data  = torch.from_numpy(data_c).cuda()
                    attScoreList = explainer.shap_values(data)
                    attScoreList = [ list(k) for k in attScoreList ]

                    for i in range( len(labelList) ):
                        if dataIndex not in attHash:
                            label_value = labelList[i]
                            attHash[dataIndex] = []
                            labelHash[dataIndex]= str( label_value )
                        #

                        contrib = np.array( attScoreList[label_value][i].flatten() )
                        attHash[dataIndex].append( contrib )

                        #
                        dataIndex = dataIndex + 1
            #
    #
    if paramHash["net_explain_explainer"] == "DeepExplainer":
        base_data = np.array( [ k[0] for k in dataHash[reference][:paramHash["net_explain_ref_size"]] ] )
        print(base_data.shape)
        base_data = torch.from_numpy(base_data).cuda()
        explainer = shap.DeepExplainer(model, base_data)

        resetData(paramHash)
        # data index for each round explanation
        dataIndex = 0
        
        for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
            msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
            infoLine(msg)

            data_c,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
            labelList = list(label)

            if numSampling > 1:
                tmpHash = {}
                for ni in range(numSampling):
                    noise = np.random.normal(0,1.0, data_c.shape )
                    data = np.absolute( data_c + noise)
                    data = torch.from_numpy(data).cuda()
                    attScoreList = explainer.shap_values(data) # 82 * 4 * 1 * 144 * 144
                    for i in range(len(labelList)):
                        if i not in tmpHash:
                            tmpHash[i] = []
                        #
                        tmpHash[i].append( attScoreList[labelList[i]][i] )
                    #
                #

                for i in range(len(labelList)):
                    if dataIndex not in attHash:
                        label_value = labelList[i]
                        attHash[dataIndex] = []
                        labelHash[dataIndex]= str( label_value )
                    #
                    contrib = np.mean(np.array(tmpHash[i]), axis=0)
                    contrib = np.array( contrib.flatten() )
                    attHash[dataIndex].append( contrib )
                    dataIndex = dataIndex + 1
            else:
                data  = torch.from_numpy(data_c).cuda()
                attScoreList = explainer.shap_values(data)
                attScoreList = [ list(k) for k in attScoreList ]

                for i in range( len(labelList) ):
                    if dataIndex not in attHash:
                        label_value = labelList[i]
                        attHash[dataIndex] = []
                        labelHash[dataIndex]= str( label_value )
                    #

                    contrib = np.array( attScoreList[label_value][i].flatten() )
                    attHash[dataIndex].append( contrib )

                    #
                    dataIndex = dataIndex + 1
        #
    return attHash, labelHash
#

def explain_captum(model, paramHash, dataHash, dataType, reference):
    ## no need for reference
    if paramHash["net_explain_explainer"] == "Saliency":
        algo = Saliency(model)
    #
    if paramHash["net_explain_explainer"] == "InputXGradient":
        algo = InputXGradient(model)
    #
    if paramHash["net_explain_explainer"] == "GuidedBackprop":
        algo = GuidedBackprop(model)
    #
    
    ## need reference
    if paramHash["net_explain_explainer"] == "IntegratedGradients":
        algo = IntegratedGradients(model)
    #
    if paramHash["net_explain_explainer"] == "DeepLift":
        algo = DeepLift(model)
    #
    if paramHash["net_explain_explainer"] == "DeepLiftShap":
        algo = DeepLiftShap(model)
    #
    if paramHash["net_explain_explainer"] == "FeatureAblation":
        algo = FeatureAblation(model)
    #

    
    ### set Noise Tunnel
    needSampling = True
    numSampling  = 1
    if paramHash["net_explain_noiseTunnel"] > 1:
        explainer = NoiseTunnel(algo)
        numSampling  = paramHash["net_explain_noiseTunnel"]
    else:
        explainer = algo
        needSampling = False
    #
    
    needRef = True
    if paramHash["net_explain_explainer"] in ["Saliency", "InputXGradient", "GuidedBackprop", "GuidedGradCam"]:
        needRef = False
    #
    
    ##
    attHash  = {}
    labelHash= {}

    if paramHash["net_explain_ref_shuffle"] == "yes":
        infoLine("Shuffling reference data")
        random.shuffle(dataHash[reference])
    #
    
    
    for refIndex in range(paramHash["net_explain_ref_size"]):
        msg = "-------------- Reference: " + str(refIndex + 1) + "/" + str( paramHash["net_explain_ref_size"] ) + " -----------------"
        infoLine(msg)
            
        ref_data = np.array( [ k[0] for k in dataHash[reference][refIndex:(refIndex+1)] ] )
        base_data_c= None
        
        resetData(paramHash)
        
        # data index for each round explanation
        dataIndex = 0

        for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
            msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
            infoLine(msg)
            
            data,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
            label_c    = list(label)
            
            if base_data_c is None or len(data) != len(base_data_c):
                base_data_c = np.array( np.broadcast_to(ref_data, data.shape) )
                base_data = torch.from_numpy(base_data_c).cuda()
            #

            data = torch.from_numpy(data).cuda()
            label= torch.from_numpy(label).cuda()
            
            if needRef:
                if needSampling:
                    att = explainer.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label, baselines=base_data)
                else:
                    att = explainer.attribute(data, target=label, baselines=base_data)
                #
            else:
                if needSampling:
                    att = explainer.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label)
                else:
                    att = explainer.attribute(data, target=label)
                #
            #
            
            att = att.cpu().data.numpy()
            for i in range(len(label_c)):
                if dataIndex not in attHash:
                    attHash[dataIndex] = []
                    labelHash[dataIndex] = str(label_c[i])
                #
                attHash[dataIndex].append( att[i].flatten() )
                dataIndex = dataIndex + 1
            #
        #
        #
        if not needRef:
            break # some explaners do not support reference, so they can quit here
        #
    #
    
    return attHash, labelHash

#

def explain_GuidedGradCam(model, paramHash, dataHash, dataType, reference):
    # set explainer
    explainer      = GuidedGradCam(model,model.conv3)
    explainer_cam  = LayerGradCam(model,model.conv3)
    explainer_guide= GuidedBackprop(model)
    
    ### set Noise Tunnel
    needSampling = True
    numSampling  = 1
    if paramHash["net_explain_noiseTunnel"] > 1:
        explainer       = NoiseTunnel(explainer)
        explainer_cam   = NoiseTunnel(explainer_cam)
        explainer_guide = NoiseTunnel(explainer_guide)
        numSampling  = paramHash["net_explain_noiseTunnel"]
    else:
        needSampling = False
    #

    ##
    attHash  = {}
    labelHash= {}

    resetData(paramHash)

    # data index for each round explanation
    dataIndex = 0

    for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
        msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
        infoLine(msg)

        data,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
        label_c    = list(label)
        datashape  = data.shape[2:]
        
        data = torch.from_numpy(data).cuda()
        label= torch.from_numpy(label).cuda()
        
        if needSampling:
            #att = explainer.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label)
            att_cam   = explainer_cam.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label)
            att_guide = explainer_guide.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label)
        else:
            #att = explainer.attribute(data, target=label)
            att_cam   = explainer_cam.attribute(data, target=label)
            att_guide = explainer_guide.attribute(data, target=label)
        #
        att_cam = F.relu(att_cam)
        att_camup= LayerAttribution.interpolate(att_cam, datashape)
        
        #att      = att.cpu().data.numpy()
        att_cam  = att_cam.cpu().data.numpy()
        att_guide= att_guide.cpu().data.numpy()
        att_camup= att_camup.cpu().data.numpy()
        att = att_guide * att_camup

        for i in range(len(label_c)):
            if dataIndex not in attHash:
                attHash[dataIndex] = []
                labelHash[dataIndex] = str(label_c[i])
            #
            attHash[dataIndex].append( att[i].flatten() )
            dataIndex = dataIndex + 1
        #

    return attHash, labelHash
#

def explain_GuidedGradCamPP(model, paramHash, dataHash, dataType, reference):
    def gradCamPP(model, paramHash, dataHash, dataType, reference):
        model.eval()
        if paramHash["net_explain_noiseTunnel"] > 1:
            explainer = SmoothGradCAMpp(model, "conv3", num_samples = paramHash["net_explain_noiseTunnel"])
        else:
            explainer = GradCAMpp(model, "conv3")
        #

        resetData(paramHash)
        # data index for each round explanation
        dataIndex = 0
        cam_list = []
        
        for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
            msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
            infoLine(msg)

            data_list,label_list = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)

            for si in range(len(data_list)):
                data = np.array(data_list[si:si+1])
                label= np.array(label_list[si:si+1])

                data = torch.from_numpy(data).cuda()
                label= torch.from_numpy(label).cuda()

                logit = model(data)
                att = explainer(logit.squeeze(0).argmax().item(), logit)
                
                attcpu = att.cpu()
                attresize = resize(attcpu[None,:,:], (data.size(2),data.size(3))).data.numpy()
                
                cam_list.append(attresize)
        #
        
        return np.array(cam_list)
    #
    def guide(model, paramHash, dataHash, dataType, reference):
        # set explainer 
        algo = GuidedBackprop(model)

        ### set Noise Tunnel
        needSampling = True
        numSampling  = 1
        if paramHash["net_explain_noiseTunnel"] > 1:
            explainer = NoiseTunnel(algo)
            numSampling  = paramHash["net_explain_noiseTunnel"]
        else:
            explainer = algo
            needSampling = False
        #
        
        resetData(paramHash)
        
        guide_list = []
        labelHash  = {}
        dataIndex  = 0 

        for batch_idx in range(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])):
            msg = "-------------- batch: " + str(batch_idx + 1) + "/" + str(min(paramHash["net_eval_batch"],paramHash[dataType]["max_batch_num"])) + " -----------------"
            infoLine(msg)
            
            data,label = getNextData(paramHash, dataHash, dataType, shuffle = False, addNoise=False)
            label_c    = list(label)

            data = torch.from_numpy(data).cuda()
            label= torch.from_numpy(label).cuda()
            
            if needSampling:
                att = explainer.attribute(data, nt_type='smoothgrad', nt_samples=numSampling, target=label)
            else:
                att = explainer.attribute(data, target=label)
            #
            
            att = att.cpu().data.numpy()

            for i in range(len(label_c)):
                guide_list.append(att[i])
                if dataIndex not in labelHash:
                    labelHash[dataIndex] = str(label_c[i])
                #
                dataIndex = dataIndex + 1
            #
        #
        
        return np.array(guide_list), labelHash
    #
    
    attHash = {}
    infoLine("Prepare for Guides")
    guide, labelHash = guide(model, paramHash, dataHash, dataType, reference)
    
    infoLine("Prepare for CAMs")
    gradcampp = gradCamPP(model, paramHash, dataHash, dataType, reference)
    
    guidecampp = guide * gradcampp
    
    for i in range( len(guidecampp) ):
        attHash[i] = []
        tmp = guidecampp[i].flatten()
        attHash[i].append(tmp)
    #

    return attHash, labelHash
#


#
def explain(model, paramHash, dataHash, dataType, reference):
    ### Set explainer
    infoLine("Set explainer as " + paramHash["net_explain_explainer"] )
    
    if paramHash["net_explain_explainer"] in ["Saliency", "InputXGradient", "GuidedBackprop", "IntegratedGradients", "DeepLift", "DeepLiftShap", "FeatureAblation"]:
        attHash, labelHash = explain_captum(model, paramHash, dataHash, dataType, reference)
    #
    
    if paramHash["net_explain_explainer"] in ["GuidedGradCam",]:
        attHash, labelHash = explain_GuidedGradCam(model, paramHash, dataHash, dataType, reference)
    #
    
    if paramHash["net_explain_explainer"] in ["GuidedGradCamPP",]:
        attHash, labelHash = explain_GuidedGradCamPP(model, paramHash, dataHash, dataType, reference)
    #
    
    if paramHash["net_explain_explainer"] in ["DeepExplainer", "DeepExplainerFactorize"]:
        attHash, labelHash = explain_shap(model, paramHash, dataHash, dataType, reference)
    #
    
    if paramHash["net_explain_separate"] == "no":
        outDIR = paramHash["explain_dir"] + "/aggregate/"
        os.system("mkdir -p " + outDIR)
        if "specific" in paramHash["net_explain_ref_type"]:
            refType = "specific"
        else:
            refType = "universal"
        #
        runID = getUniqueRunID()
        
        outfile = outDIR + paramHash["net_explain_explainer"] + "."+ paramHash["net_explain_pool_post"] + ".ref-" + str(paramHash["net_explain_ref_size"]) + ".noiseTunnel-" + str(paramHash["net_explain_noiseTunnel"]) + "." + refType + "." + runID + ".dat"
        with open( outfile , "wt" ) as fo:
            for dataIndex in range( len( attHash ) ):
                attHash[dataIndex] = np.abs( np.array( attHash[dataIndex] ) )
                contrib = np.median( attHash[dataIndex] , axis= 0 )
                contrib = contrib.flatten()

                outline = str( labelHash[dataIndex] ) + "\t" + "|".join( [ str(v) for v in contrib[ : paramHash["data_gene_num"] ] ] )
                fo.write( outline + "\n" )
        #
    else:
        outDIR = paramHash["explain_dir"] + "/" + paramHash["net_explain_explainer"] + "/"
        os.system("mkdir -p " + outDIR)
        runID = getUniqueRunID()
        if "specific" in paramHash["net_explain_ref_type"]:
            refType = "specific"
        else:
            refType = "universal"
        #
        
        for refIndex in range(paramHash["net_explain_ref_size"]):
            outfile = outDIR + paramHash["net_explain_explainer"] + ".tissue-"+ paramHash["net_explain_pool_post"] + ".refIndex-" + str(refIndex + 1) + ".noiseTunnel-" + str(paramHash["net_explain_noiseTunnel"]) + "." + refType + "." + runID + ".dat"
            with open( outfile , "wt" ) as fo:
                for dataIndex in range( len( attHash ) ): 
                    contrib = np.abs( attHash[dataIndex][refIndex] )
                    contrib = contrib.flatten()

                    outline = str( labelHash[dataIndex] ) + "\t" + "|".join( [ str(v) for v in contrib[ : paramHash["data_gene_num"] ] ] )
                    fo.write( outline + "\n" )
            #
            
            if len(attHash[0]) < 2:
                break # some explaners do not support reference, so they can quit here
            #
        #
    #
#

def readMeta(paramHash):
    with open(paramHash["data_in_dir"] + "/meta.dat" , "rt") as fi:
        for line in fi:
            line = line.rstrip()
            row  = line.split("\t")
            if row[0] == "class":
                paramHash["data_class_num"] = int(row[1])
            #
            if row[0] == "gene":
                paramHash["data_gene_num"] = int(row[1])
            #
#

def resetData(paramHash):
    for dataType in ["train", "valid","test","pool"]:
        if dataType in paramHash:
            paramHash[dataType]["current_index"] = -1
    #
#



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 2.0.0 (CNN) \nDescription: predict sample classification by expression",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="inDIR",     type=str,   required=True,                  help="The directory of input data")
    parser.add_argument("-o", dest="outDIR",    type=str,   required=True,                  help="The directory of output data or model")
    parser.add_argument("-M", dest="runMode",   type=str,   required=False, default="train",help="Running mode: ", choices=['train', 'evaluate', 'predict', 'explain'])
    parser.add_argument("-b", dest="batchSize", type=int,   required=False, default=256,    help="Batch size")

    # training options
    parser.add_argument("-u", dest="upsample",  type=str,   required=False, default="yes",  help="Do up sampling: ", choices=['no', 'yes'])
    parser.add_argument("-R", dest="adjLR",     type=float, required=False, default=0.20,   help="Adjust learning rate")
    parser.add_argument("-p", dest="patience",  type=int,   required=False, default=5,      help="Max number of epochs with no improvement")
    parser.add_argument("-c", dest="metric",    type=str,   required=False, default="loss", help="Monitored metric: ", choices=['loss', 'accuracy'])
    parser.add_argument("-r", dest="learnRate", type=float, required=False, default=0.001,  help="Learning rate")
    parser.add_argument("-l", dest="l2Reg",     type=float, required=False, default=0.001,  help="L2 regularization")
    parser.add_argument("-e", dest="maxEpoch",  type=int,   required=False, default=200,    help="Max epoch number")
    parser.add_argument("-t", dest="evalBatch", type=int,   required=False, default=100,    help="Max evaluation batch")
    parser.add_argument("-N", dest="dataNoise", type=float, required=False, default=0.2,    help="Fraction of training data to be added noise while training the model.")
    parser.add_argument("-s", dest="noiseSigma",type=float, required=False, default=1.00,   help="Sigma for data noise, noise was generated by normal distribution N(0,sigma)")


    # explain options
    parser.add_argument("-B", dest="refType",   type=str,   required=False, default="specific", help="Type of reference samples", choices=['specific','universal','normal', 'zero'])
    parser.add_argument("-S", dest="refSize",   type=int,   required=False, default=64,         help="Number of reference samples to be used in the explanation step")
    parser.add_argument("-P", dest="postName",  type=int,   required=False,                     help="Tissue code, such as 0, 1, 2 and etc.")
    parser.add_argument("-y", dest="refShuffle", type=str,   required=False, default="yes",     help="Shuffle reference samples.", choices=['no', 'yes'])
    parser.add_argument("-k", dest="separateRef",type=str,  required=False, default="no",       help="Whether output each reference output separately", choices=['no', 'yes'])
    parser.add_argument("-E", dest="explainer", type=str,   required=False, default="DeepLiftShap", help="Type of model explainer", choices=['Saliency', 'IntegratedGradients', 'InputXGradient', 'GuidedBackprop', 'DeepLift', 'DeepLiftShap', 'FeatureAblation', "DeepExplainer", "DeepExplainerFactorize", "GuidedGradCam", "GuidedGradCamPP"])
    parser.add_argument("-D", dest="noiseTunnel",type=int,  required=False, default=1,          help="How many pseudo-samples to use by adding gaussian noise, 1: disable this option")
    
    
    
    args=parser.parse_args()
    
    # main parames
    paramHash = {}
    dataHash  = {}

    # param
    paramHash["data_in_dir"]           = args.inDIR # 
    paramHash["data_out_dir"]          = args.outDIR
    paramHash["run_mode"]              = args.runMode   # train
    paramHash["net_batch_size"]        = args.batchSize # 256
    paramHash["net_max_epoch"]         = args.maxEpoch  # 1000
    paramHash["net_learn_rate"]        = args.learnRate # 0.0001
    paramHash["net_adj_learn_rate"]    = args.adjLR     # 1.0
    paramHash["net_patience"]          = args.patience  # 10
    paramHash["net_metric"]            = args.metric    # loss
    paramHash["net_l2_regularization"] = args.l2Reg    # 0.0
    paramHash["net_eval_batch"]        = args.evalBatch # 50
    paramHash["net_data_noise"]        = args.dataNoise # 0.0
    paramHash["net_noise_sigma"]       = args.noiseSigma # 0.01
    paramHash["net_explain_ref_type"]   = args.refType # "norm"
    paramHash["net_explain_ref_size"]   = args.refSize # 16
    paramHash["net_explain_pool_post"] = args.postName #
    paramHash["net_explain_ref_shuffle"]= args.refShuffle # no
    
    paramHash["net_explain_separate"]  = args.separateRef # no
    paramHash["net_explain_explainer"] = args.explainer # 
    paramHash["net_explain_noiseTunnel"] = args.noiseTunnel # 
    
    
    
    
    if args.upsample == "yes":
        paramHash["data_upsample"] = True
        infoLine("Do up sampling")
    else:
        paramHash["data_upsample"] = False
        infoLine("Use vanilla data")
    #
    
    # modify reference type
    if paramHash["run_mode"] == "explain":
        if paramHash["net_explain_pool_post"] is None or paramHash["net_explain_pool_post"] < 0:
            infoLine("Invalid tissue code", "error")
        else:
            paramHash["net_explain_pool_post"] = str( paramHash["net_explain_pool_post"] )
            if paramHash["net_explain_ref_type"] == "specific":
                paramHash["net_explain_ref_type"] = paramHash["net_explain_ref_type"] +"."+ paramHash["net_explain_pool_post"]
            #

    #

    # read meta
    readMeta(paramHash)
    
    # model directory
    paramHash["model_dir"]   = paramHash["data_out_dir"] + "/saved_models"
    paramHash["evaluate_dir"]= paramHash["data_out_dir"] + "/evaluation"
    paramHash["predict_dir"] = paramHash["data_out_dir"] + "/prediction"
    paramHash["explain_dir"] = paramHash["data_out_dir"] + "/explanation"
    
    if paramHash["run_mode"] == "train":
        os.system("mkdir -p " + paramHash["model_dir"] )
    #
    if paramHash["run_mode"] == "evaluate":
        os.system("mkdir -p " + paramHash["evaluate_dir"] )
    #
    if paramHash["run_mode"] == "predict":
        os.system("mkdir -p " + paramHash["predict_dir"] )
    #
    if paramHash["run_mode"] == "explain":
        os.system("mkdir -p " + paramHash["explain_dir"] )
    #
    
    # read data
    if paramHash["run_mode"] == "train":
        # make directory
        paramHash["model_dir"] = paramHash["data_out_dir"] + "/saved_models"
        os.system("mkdir -p " + paramHash["model_dir"] )
        
        # read data
        for dataType in ["train", "valid"]:
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            dataHash[dataType] = readRawData(paramHash, dataType, maxline = -1)
        #
        paramHash["net_max_valid_batch"] = min( paramHash["train"]["max_batch_num"] , paramHash["valid"]["max_batch_num"] )
    #
    
    if paramHash["run_mode"] == "evaluate":
        # read data
        for dataType in ["train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            #
            
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            dataHash[dataType] = readRawData(paramHash, dataType, maxline = -1)
        #
    #
    
    if paramHash["run_mode"] == "predict":
        for dataType in ["pool","train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            #
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            dataHash[dataType] = readRawData(paramHash, dataType, maxline = -1)
        #
    #
    
    if paramHash["run_mode"] == "explain":
        for dataType in ["pool",paramHash["net_explain_ref_type"]]:
            infoLine("Reading "+ dataType +" data")
            paramHash[dataType] = {}
            if dataType == "pool":
                paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+"."+paramHash["net_explain_pool_post"]+".dat"
            else:
                paramHash[dataType]["infile"] = paramHash["data_in_dir"] + "/"+dataType+".dat"
            dataHash[dataType] = readRawData(paramHash, dataType, maxline = -1)
        #
    #
    
        
    infoLine("Report parameters")
    print("\n")
    print(paramHash)
    print("\n")
    
    # reset cache on gpu
    torch.cuda.empty_cache()
    
    infoLine("Set model")
    net = Net(paramHash)
    net = net.double()
    #
    
    infoLine("Copy network to GPU")
    print( "Number of parameters:" , count_parameters(net) )
    net.cuda()
    print(net)
    
    criterion = nn.CrossEntropyLoss()
    
    ############################################ train ############################################
    if paramHash["run_mode"] == "train":
        best_count = 0 # automatically stop when there is no improve for # epoches
        infoLine("Start trainning")
        optimizer = torch.optim.Adam(params=net.parameters(), lr=paramHash["net_learn_rate"], weight_decay = paramHash["net_l2_regularization"])
        best_performace = 0.0
        
        if paramHash["net_adj_learn_rate"] < 1.0:
            if paramHash["net_metric"] == "loss":
                best_performace = 1e20
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=paramHash["net_adj_learn_rate"], patience=paramHash["net_patience"], verbose=True)
            if paramHash["net_metric"] == "accuracy":
                best_performace = 0.0
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=paramHash["net_adj_learn_rate"], patience=paramHash["net_patience"], verbose=True)
        #
        
        for epochIndex in range(paramHash["net_max_epoch"]):
            #### train on dataset
            if paramHash["data_upsample"]:
                upsampleData(dataHash, paramHash, True)
            loss_train = train(net, paramHash, dataHash, optimizer, criterion)
            
            #### evaluate on dataset
            if paramHash["data_upsample"]:
                upsampleData(dataHash, paramHash, False)
            evaluation_trainData= evaluate(net, paramHash, dataHash, "train", criterion)
            evaluation_validData= evaluate(net, paramHash, dataHash, "valid", criterion)

            if paramHash["net_adj_learn_rate"] < 1.0:
                if paramHash["net_metric"] == "loss":
                    scheduler.step(evaluation_validData[2])
                if paramHash["net_metric"] == "accuracy":
                    scheduler.step(evaluation_validData[3]/evaluation_validData[4])
            #

            outline = "epoch:" + str(epochIndex + 1) + "\t" + "training_loss:" + "{:.6f}".format(loss_train)
            outline = outline + "\t" + "loss_train:" + "{:.6f}".format(evaluation_trainData[2]) + "\t" + "loss_valid:" + "{:.6f}".format(evaluation_validData[2])
            outline = outline + "\t" + "accuracy_train:" + str(evaluation_trainData[3]) + "/" + str(evaluation_trainData[4]) + "("+ "{:.2f}".format(evaluation_trainData[3] * 100.0 / evaluation_trainData[4]) +"%)"
            outline = outline + "\t" + "accuracy_valid:" + str(evaluation_validData[3]) + "/" + str(evaluation_validData[4]) + "("+ "{:.2f}".format(evaluation_validData[3] * 100.0 / evaluation_validData[4]) +"%)"
            infoLine(outline)
            resetData(paramHash)
            
            # check improvement
            improvement = False
            if paramHash["net_metric"] == "loss":
                if evaluation_validData[2] < best_performace:
                    best_performace = evaluation_validData[2]
                    improvement = True
            #
            if paramHash["net_metric"] == "accuracy":
                if evaluation_validData[3] * 1.0 / evaluation_validData[4] > best_performace:
                    best_performace = evaluation_validData[3] * 1.0 / evaluation_validData[4]
                    improvement = True
            #
            if improvement:
                modelfile = paramHash["model_dir"] + "/best.pt"
                torch.save(net.state_dict(), modelfile)
                best_count = 0
            else:
                best_count = best_count + 1
                if best_count > 3* paramHash["net_patience"]:
                    infoLine("There is no improve for " + str(best_count) + " continuous epoches, so we can stop training here." )
                    break
            #
    #
    if paramHash["run_mode"] == "evaluate":
        # load model
        infoLine("Loading trained model")
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/best.pt"))
        
        with open( paramHash["evaluate_dir"] + "/evalutation.tab", "wt" ) as fo:
            for dataType in ["train","valid","test"]:
                if dataType not in dataHash:
                    continue
                #
                infoLine("Make evaluation " + dataType)
                eval_retData= evaluate(net, paramHash, dataHash, dataType, criterion)
                outline = "accuracy on "+dataType+":" + str(eval_retData[3]) + "/" + str(eval_retData[4]) + "("+ "{:.2f}".format(eval_retData[3] * 100.0 / eval_retData[4]) +"%)"
                fo.write(outline + "\n")
                infoLine(outline)
    #
    
    if paramHash["run_mode"] == "predict":
        # load model
        infoLine("Loading trained model")
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/best.pt"))
        
        for dataType in ["pool","train","valid","test"]:
            if not os.path.exists( paramHash["data_in_dir"] + "/"+dataType+".dat" ):
                continue
            #
        
            infoLine("Make prediction on " + dataType)
            eval_retData= evaluate(net, paramHash, dataHash, dataType, criterion, prediction=True)
            outline = "accuracy_overall:" + str(eval_retData[3]) + "/" + str(eval_retData[4]) + "("+ "{:.2f}".format(eval_retData[3] * 100.0 / eval_retData[4]) +"%)"
            infoLine(outline)

            # output data # label_save,predicted_save,valid_loss,correct,total,digit_save
            labelList = list( eval_retData[0] )
            predList  = list( eval_retData[1] )
            digitList = [ list(k) for k in eval_retData[-1] ]

            with open( paramHash["predict_dir"] + "/prediction."+dataType+".tab", "wt" ) as fo:
                fo.write("label\tprediction\tscore\n")
                for i in range( len( labelList ) ):
                    outline = str(int( labelList[i] )) + "\t" + str(int( predList[i] ))+ "\t" + "|".join( [str(k) for k in digitList[i] ] )
                    fo.write( outline + "\n" )
            #
    #
    
    if paramHash["run_mode"] == "explain":
        # load model
        infoLine("Loading trained model")
        net.load_state_dict(torch.load(paramHash["model_dir"] + "/best.pt"))
        
        infoLine("Make explanation on whole data")
        explain(net, paramHash, dataHash, "pool",paramHash["net_explain_ref_type"])
        
    #
    ################################################
#
infoLine("Done!")