import numpy as np
import copy
import os,sys,time,datetime,gzip
import math
import random
import multiprocessing
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

def processDataset(datasetId, outputDIR, refNum, tempDIR):
    datasetDIR = outputDIR + "/" + datasetId
    ### prepare pool for each tissue
    filelist = os.listdir(datasetDIR)
    filelist = [ filename for filename in filelist if "temppool" in filename]
    for filename in filelist:
        tmp = filename.split(".")
        tissueCode = tmp[1]
        cmd = "cut -f 1 " + datasetDIR + "/" + filename + " > " + datasetDIR + "/sampleList."+ tissueCode +".dat"
        os.system(cmd)
        
        cmd = "cut -f 2- " + datasetDIR + "/" + filename + " > " + datasetDIR + "/pool."+ tissueCode +".dat"
        os.system(cmd)
    #
    
    
    ### prepare Tissue code
    cmd = "cp " + tempDIR + "/tissueCode.dat" + " " + datasetDIR + "/tissueCode.dat"
    os.system(cmd)
    
    
    ### merge pool
    filelist = [ datasetDIR + "/" + filename for filename in filelist ]
    cmd = "cat " + " ".join( filelist ) + " | shuf > " + datasetDIR + "/pool.tmp"
    os.system(cmd)
    
    cmd = "cut -f 1 " + datasetDIR + "/pool.tmp" + " > " + datasetDIR + "/sampleList.pool.dat"
    os.system(cmd)
    
    cmd = "cut -f 2- " + datasetDIR + "/pool.tmp" + " > " + datasetDIR + "/pool.dat"
    os.system(cmd)
    
    cmd = "rm -rf " + datasetDIR + "/pool.tmp"
    os.system(cmd)

    
    ### merge train
    filelist = os.listdir(datasetDIR)
    filelist = [ filename for filename in filelist if "temptrain" in filename]
    filelist = [ datasetDIR + "/" + filename for filename in filelist ]
    cmd = "cat " + " ".join( filelist ) + " | shuf > " + datasetDIR + "/train.dat"
    os.system(cmd)
    
    
    ### merge test
    filelist = os.listdir(datasetDIR)
    filelist = [ filename for filename in filelist if "temptest" in filename]
    filelist = [ datasetDIR + "/" + filename for filename in filelist ]
    cmd = "cat " + " ".join( filelist ) + " | shuf > " + datasetDIR + "/test.dat"
    os.system(cmd)
    
    
    ### merge valid
    filelist = os.listdir(datasetDIR)
    filelist = [ filename for filename in filelist if "tempvalid" in filename]
    filelist = [ datasetDIR + "/" + filename for filename in filelist ]
    cmd = "cat " + " ".join( filelist ) + " | shuf > " + datasetDIR + "/valid.dat"
    os.system(cmd)
    
    ### clean up
    cmd = "rm -rf " + datasetDIR + "/temp* "
    os.system(cmd)
    
    #
    infoLine(datasetId + " was done" )
#

def processDataByDataset(cpuNum, outputDIR, refNum, datasetNum, tempDIR):
    def loadTissueData(npy):
        tissueDataHash = np.load( npy, allow_pickle=True )
        tissueDataHash = tissueDataHash.item()
        return tissueDataHash
    #
    
    def loadTissueData(npy):
        tissueDataHash = np.load( npy, allow_pickle=True )
        tissueDataHash = tissueDataHash.item()
        return tissueDataHash
    #
    
    def truncated_norm(mu, sigma, refNum):
        sigma = sigma + 0.01
        a = mu - 1.0 * sigma
        b = mu + 1.0 * sigma
        while 1:
            x = np.random.normal(mu, sigma, size=int(refNum*3.0))
            x = x[ np.where( x >= a ) ]
            x = x[ np.where( x <= b ) ]
            if len(x) >= refNum:
                return x[:refNum]
        #
    #
    
    def makeUniversalRefData(refNum, tempDIR):
        refDataHash = {}
        
        ### calculate median expression value for each tissue
        medDataHash = {}
        npylist = os.listdir(tempDIR)
        npylist = [npy for npy in npylist if ".npy" in npy]
        for npy in npylist:
            infile = tempDIR + "/" + npy
            tissueDataHash = loadTissueData(infile)

            sampleList = list(tissueDataHash.keys())
            geneList = list(tissueDataHash[sampleList[0]].keys())
            
            for geneName in geneList:
                vlist = [ tissueDataHash[sample][geneName] for sample in sampleList ]
                if geneName not in medDataHash:
                    medDataHash[geneName] = []
                #
                medDataHash[geneName].append( np.median( vlist ) )
        #
        
        ### generate random expression values for each gene based on median scores
        
        for geneName in geneList:
            vlist = list(np.absolute(truncated_norm( np.mean(medDataHash[geneName]) , np.std(medDataHash[geneName]), refNum )))
            refDataHash[geneName] = [ "{:.2f}".format(k) for k in vlist ]
        return refDataHash
    #
    
    def saveUniversalReferenceDataByDataset(refDataHash, datasetDIR, refNum):
        geneFile = datasetDIR + "/orderedGeneList.dat"
        with open(geneFile, "rt") as fi:
            geneList = fi.readlines()
            geneList = [ geneName.rstrip() for geneName in geneList]
        #
        
        with open(datasetDIR + "/universal.dat", "wt" ) as fo:
            for k in range(refNum):
                outline = "universal\t9999\t" + "|".join([ refDataHash[geneName][k] for geneName in geneList ])
                fo.write(outline + "\n" )
        #
    #

    # prepare universal reference
    infoLine("Generating universal references for each dataset")
    refDataHash = makeUniversalRefData(refNum, tempDIR)
    for k in range(datasetNum):
        datasetDIR = outputDIR + "/dataset-" + str(k+1)
        saveUniversalReferenceDataByDataset(refDataHash, datasetDIR, refNum)
    #
    
    infoLine("Finishing data for each dataset")
    mpPool = multiprocessing.Pool(cpuNum)
    for k in range(datasetNum):
        datasetId  = "dataset-" + str(k+1)
        mpPool.apply_async( processDataset, args=(datasetId, outputDIR, refNum, tempDIR) )
    #
    mpPool.close()
    mpPool.join()
#

def processTissue(filename, outputDIR, refNum, datasetNum, tempDIR):
    def getTissue(filename,tempDIR):
        tmp = filename.split(".")
        tissueCode = tmp[1]
        code2tissueHash = {}
        with open(tempDIR + "/tissueCode.dat" , "rt" ) as fi:
            for line in fi:
                row = line.rstrip().split("\t")
                code2tissueHash[row[0]] = row[1]
        #
        return code2tissueHash[tissueCode] , tissueCode
    #
    
    def loadTissueData(npy):
        tissueDataHash = np.load( npy, allow_pickle=True )
        tissueDataHash = tissueDataHash.item()
        return tissueDataHash
    #
    
    def truncated_norm(mu, sigma, refNum):
        sigma = sigma + 0.01
        a = mu - 1.0 * sigma
        b = mu + 1.0 * sigma
        while 1:
            x = np.random.normal(mu, sigma, size=int(refNum*3.0))
            x = x[ np.where( x >= a ) ]
            x = x[ np.where( x <= b ) ]
            if len(x) >= refNum:
                return x[:refNum]
    #

    def makeSpecificRefData(tissueDataHash, refNum):
        sampleList = list(tissueDataHash.keys())
        geneList = list(tissueDataHash[sampleList[0]].keys())
        refDataHash = {}
        
        for geneName in geneList:
            vlist = [ tissueDataHash[sample][geneName] for sample in sampleList ]
            vlist = list(np.absolute(truncated_norm( np.mean(vlist) , np.std(vlist), refNum )))
            refDataHash[geneName] = [ "{:.2f}".format(k) for k in vlist ]
        return refDataHash
    #
    
    def readGeneList(infile):
        with open(infile, "rt") as fi:
            geneList = fi.readlines()
            geneList = [ geneName.rstrip() for geneName in geneList ]
            return geneList
    #
    
    def saveReferenceDataByTissue(prefix, geneList, refDataHash, refNum, outfile):
        with open(outfile, "wt") as fo:
            for i in range(refNum):
                vlist = [ refDataHash[geneName][i] for geneName in geneList ]
                outline = prefix + "\t" + "|".join( vlist )
                fo.write( outline  +"\n" )
    #
    
    def saveDataByTissue(prefix, geneList, tissueDataHash, outfile):
        poolData = []
        with open(outfile, "wt") as fo:
            sampleList = list(tissueDataHash.keys())
            sampleList.sort()
            for sample in sampleList:
                vlist = [ "{:.2f}".format(tissueDataHash[sample][geneName]) for geneName in geneList ]
                outline = sample + "\t" + prefix + "\t" + "|".join( vlist )
                fo.write( outline + "\n" )
                poolData.append( prefix + "\t" + "|".join( vlist ) )
        #
        return poolData
    #

    def saveTrainTestValidDataByTissue(poolData, train_file, test_file, valid_file):
        random.shuffle(poolData)

        if len(poolData) > 50:
            list_train = poolData[:int(len(poolData)*0.90) ]
            list_test  = poolData[ int(len(poolData)*0.90) : int(len(poolData)*0.95) ]
            list_valid = poolData[ int(len(poolData)*0.95) :  ]
        else:
            list_train = poolData[:-10]
            list_test  = poolData[-10:-5]
            list_valid = poolData[-5:]
        #

        with open( train_file, "wt" ) as fo:
            fo.write( "\n".join( list_train ) + "\n" )
        #

        with open( test_file, "wt" ) as fo:
            fo.write( "\n".join( list_test ) + "\n" )
        #

        with open( valid_file, "wt" ) as fo:
            fo.write( "\n".join( list_valid ) + "\n" )
    #
    
    tissue, tissueCode = getTissue(filename,tempDIR)
    
    infoLine("Loading data from " + tissue)
    tissueDataHash = loadTissueData(tempDIR + "/" + filename)
    
    infoLine("Generating specific references based on truncated normal distribution on the samples from " + tissue)
    refDataHash = makeSpecificRefData(tissueDataHash, refNum)
    
    infoLine("Preparing dataset for " + tissue)
    for k in range(datasetNum):
        datasetDIR = outputDIR + "/dataset-" + str(k+1)
        geneList = readGeneList( datasetDIR + "/orderedGeneList.dat" )
        
        # save reference data by tissue
        prefix  = "specific." + tissueCode
        outfile = datasetDIR + "/" + prefix + ".dat"
        saveReferenceDataByTissue(prefix + "\t9999", geneList, refDataHash, refNum, outfile)
        
        # save data by tissue 
        outfile = datasetDIR + "/temppool." + tissueCode + ".dat"
        poolData= saveDataByTissue( tissue + "\t" + tissueCode, geneList, tissueDataHash, outfile )
        
        # save train, test, valid data for the tissue
        train_file = datasetDIR + "/temptrain." + tissueCode + ".dat"
        test_file  = datasetDIR + "/temptest."  + tissueCode + ".dat"
        valid_file = datasetDIR + "/tempvalid." + tissueCode + ".dat"
        saveTrainTestValidDataByTissue(poolData, train_file, test_file, valid_file)
#


def processDataByTissue(cpuNum, outputDIR, refNum, datasetNum, tempDIR):
    filelist = os.listdir(tempDIR)
    filelist = [ myfile for myfile in filelist if "BYTISSUE" in myfile]
    filelist.sort()

    mpPool = multiprocessing.Pool(cpuNum)
    for filename in filelist:
        mpPool.apply_async( processTissue, args=(filename, outputDIR, refNum, datasetNum, tempDIR) )
    #
    mpPool.close()
    mpPool.join()
#

def skipPreprocessData(infile, outputDIR, refNum, datasetNum, tempDIR):
    def readGeneList(infile):
        with open(infile, "rt") as fi:
            geneList = fi.readlines()
            geneList = [ geneName.rstrip() for geneName in geneList ]
            return geneList
    #
    
    def prepareShuffledGeneList(geneList, outputDIR, datasetNum,shuffle=False):
        for i in range(datasetNum):
            datasetDIR = outputDIR + "/dataset-" + str(i+1)
            os.system("mkdir -p " + datasetDIR)

            geneList_s = copy.deepcopy(geneList)
            if shuffle:
                random.shuffle(geneList_s)

            with open(datasetDIR + "/orderedGeneList.dat", "wt" ) as fo:
                fo.write( "\n".join( geneList_s ) + "\n" )
    #
    def copyMetaData(outputDIR, tempDIR, datasetNum):
        for i in range(datasetNum):
            datasetDIR = outputDIR + "/dataset-" + str(i+1)
            os.system("mkdir -p " + datasetDIR)
            os.system("cp -rf " + tempDIR + "/meta.dat" + " " + datasetDIR )
    #
    
    # read gene list
    geneList = readGeneList( tempDIR + "/geneList.dat" )
    
    # prepare gene list
    infoLine("Shuffling gene order for each dataset")
    prepareShuffledGeneList(geneList, outputDIR, datasetNum,shuffle=True)
    
    # copy meta data
    infoLine("Copying meta data for each dataset")
    copyMetaData(outputDIR, tempDIR, datasetNum)
#


def preprocessData(infile, outputDIR, refNum, datasetNum, tempDIR):
    def saveMetaData(refNum, geneList, dataHash, sampleList, outputDIR, tempDIR):
        ### estimate train, test and valid
        train = 0
        test  = 0
        valid = 0

        for tissueCode in dataHash:
            num = len( dataHash[tissueCode] )
            if num <= 50:
                train = train + num - 10
                test = test + 5
                valid= valid + 5
            else:
                train = train + int(num * 0.9)
                test = test + int(num * 0.95) - int(num * 0.9)
                valid= valid + num - int(num * 0.95)
        #

        ### save data    
        for i in range(datasetNum):
            datasetDIR = outputDIR + "/dataset-" + str(i+1)
            os.system("mkdir -p " + datasetDIR)

            with open(datasetDIR + "/meta.dat", "wt" ) as fo:
                fo.write( "class\t"      + str( len( dataHash ) ) + "\n" )
                fo.write( "gene\t"      + str( len( geneList ) ) + "\n" )

                fo.write( "reference\t" + str( refNum ) + "\n" )
                fo.write( "pool\t"      + str( len( sampleList ) ) + "\n" )

                fo.write( "train\t" + str( train ) + "\n" )
                fo.write( "test\t"  + str( test ) + "\n" )
                fo.write( "valid\t" + str( valid ) + "\n" )
            #
            os.system("cp -rf " + datasetDIR + "/meta.dat" + " " + tempDIR )
        #
        
        ### save genelist to tempDIR
        with open(tempDIR + "/geneList.dat", "wt") as fo:
            fo.write( "\n".join(sorted(geneList)) + "\n" )
        #
    #

    def prepareShuffledGeneList(geneList, outputDIR, datasetNum,shuffle=False):
        for i in range(datasetNum):
            datasetDIR = outputDIR + "/dataset-" + str(i+1)
            os.system("mkdir -p " + datasetDIR)

            geneList_s = copy.deepcopy(geneList)
            if shuffle:
                random.shuffle(geneList_s)

            with open(datasetDIR + "/orderedGeneList.dat", "wt" ) as fo:
                fo.write( "\n".join( geneList_s ) + "\n" )
    #

    def saveTissueCode(tissueHash, tempDIR):
        with open(tempDIR + "/tissueCode.dat", "wt" ) as fo:
            for tissue in tissueHash:
                fo.write( tissueHash[tissue] + "\t" + tissue + "\n" )
    #

    def splitDataByTissue(dataHash, tempDIR):
        for tissueCode in dataHash:
            npy = tempDIR + "/BYTISSUE." + tissueCode + ".npy"
            np.save( npy, dataHash[tissueCode] )
    #

    def readData(infile):
        geneList = []
        sampleList = []
        tissueCodeList = []
        tissueHash = {}
        dataHash = {}

        infoLine("Loading data")
        if infile.endswith(".gz"):
            fi = gzip.open(infile, "rt")
        else:
            fi = open(infile, "rt")
        #
        content = fi.readlines()
        #
        fi.close()


        infoLine("Processing header")
        # header
        if len(content) > 0:
            row = content[0].rstrip().split("\t")
            tmpHash = {}
            sampleList = np.array(row[1:])

            for sample in sampleList:
                tmp = sample.split("|")
                tissue = tmp[0]
                tmpHash[tissue] = ""
            #

            tmpList = list(tmpHash.keys())
            tmpList.sort()
            for i in range( len( tmpList ) ):
                tissue = tmpList[i]
                tissueHash[tissue] = str(i)
            #

            for tissue in tissueHash:
                if tissueHash[tissue] not in dataHash:
                    dataHash[tissueHash[tissue]] = {}
                #
            #

            for i in range(len(sampleList)):
                sample = sampleList[i]
                tmp = sample.split("|")
                tissue = tmp[0]
                tissueCode = tissueHash[tissue]

                if sample not in dataHash[tissueCode]:
                    dataHash[tissueCode][sample] = {}
                #
                tissueCodeList.append( tissueCode )
            #
        #

        # process data
        infoLine("Processing data")
        content = content[1:]
        for line in content:
            row = line.rstrip().split("\t")
            geneName = row[0]

            geneList.append(geneName)

            vlist = np.array( [float(k) for k in row[1:]] )
            if np.min(vlist) < 0:
                infoLine("Expression score should be >=0", "error")
            #
            vlist = np.log2(vlist + 1.0)

            for i in range( len( vlist ) ):
                dataHash[ tissueCodeList[i] ][ sampleList[i] ][geneName] = vlist[i]
            #
        #

        geneList = sorted(geneList)

        ### return data
        return geneList, sampleList, tissueHash, dataHash
    #
    
    infoLine("Reading data")
    geneList, sampleList, tissueHash, dataHash = readData(infile)
    
    infoLine("Summary: " + str(len(geneList)) + " genes, " + str(len(tissueHash)) + " tissues, " + str(len(sampleList)) + " samples." )

    infoLine("Spliting data by tissue")
    splitDataByTissue(dataHash, tempDIR)
    
    infoLine("Saving tissue code")
    saveTissueCode(tissueHash, tempDIR)
    
    infoLine("Shuffling gene order for each dataset")
    #prepareShuffledGeneList(geneList, outputDIR, datasetNum, shuffle=False)
    prepareShuffledGeneList(geneList, outputDIR, datasetNum, shuffle=True)
    
    infoLine("Saving meta data")
    saveMetaData(refNum, geneList, dataHash, sampleList, outputDIR)
    
    #
    del geneList, sampleList, tissueHash, dataHash
#



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 2.0.0 \nDescription: prepare datasets for model interpretability",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="infile",    type=str,   required=True,  help="The path of input data")
    parser.add_argument("-o", dest="outDIR",    type=str,   required=True,  help="The directory of output data")
    parser.add_argument("-m", dest="datasetNum",type=int,   required=False, help="The number of dataset to be made.", default=1)
    parser.add_argument("-t", dest="cpuNum",    type=int,   required=False, help="Thread number. Default: all CPUs in the system.")
    parser.add_argument("-r", dest="refNum",    type=int,   required=False, help="The number of random references to be generated.", default=128)
    parser.add_argument("-k", dest="useTemp",   type=str,   required=False, default="no",  help="Begin with temp data.", choices=['no', 'yes'])

    args=parser.parse_args()
    
    infile     = args.infile # 
    outputDIR  = args.outDIR #
    cpuNum     = args.cpuNum #
    datasetNum = args.datasetNum #
    refNum     = args.refNum #
    useTempData= args.useTemp #
    
    if cpuNum is None:
        cpuNum = os.cpu_count()
    #

    tempDIR = outputDIR + "/tempDIR"
    os.system("mkdir -p " + tempDIR)

    infoLine("*** (1/3) Preprocessing data ***")
    if not useTempData:
        preprocessData(infile, outputDIR, refNum, datasetNum, tempDIR)
    else:
        skipPreprocessData(infile, outputDIR, refNum, datasetNum, tempDIR)
    #

    infoLine("*** (2/3) Processing data by tissue ***")
    processDataByTissue(cpuNum, outputDIR, refNum, datasetNum, tempDIR)


    infoLine("*** (3/3) Finalizing dataset ***")
    processDataByDataset(cpuNum, outputDIR, refNum, datasetNum, tempDIR)

    if not useTempData:
        os.system("rm -rf " + tempDIR)
    #
    infoLine("Done!")

