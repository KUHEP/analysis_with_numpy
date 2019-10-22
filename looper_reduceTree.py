#!/usr/bin/env pythoin

import subprocess as sp
import os
import threading as thr
import time

def runScript(script, inFile, oFile):
    FNULL = open(os.devnull, 'w')
    sp.call([script, inFile, oFile], stdout=FNULL)

def doLoop(script, inDir, inFileList, outDir):

    f = open(inFileList, 'r')
    fLines = f.readlines()
    
    threads = [thr.Thread(target=runScript, args=(script, str(inDir+line.replace('\n','')), str(outDir+line.replace('\n','')))) for line in fLines] 
    maxThreads=False
    for it, t in enumerate(threads):

        t.start()
        print 'job ', it, ' started'
        if thr.active_count() >= 11: maxThreads = True
        while maxThreads:
            time.sleep(2)
            if thr.active_count() < 11: maxThreads = False
            

if __name__ == "__main__":

    doLoop('./condor/reduceTree.py', './samples_latest/', './samples_latest_700.txt', './samples_reduced/samples_latest_cutflow/')
    #doLoop('./condor/reduceTree_JERUp.py', './samples_latest/JERUp/', './samples_latest_700.txt', './samples_reduced/samples_latest_JERUp/')
    #doLoop('./condor/reduceTree_JERDown.py', './samples_latest/JERDown/', './samples_latest_700.txt', './samples_reduced/samples_latest_JERDown/')
    #doLoop('./condor/reduceTree_JESUp.py', './samples_latest/JESUp/', './samples_latest_700.txt', './samples_reduced/samples_latest_JESUp/')
    #doLoop('./condor/reduceTree_JESDown.py', './samples_latest/JESDown/', './samples_latest_700.txt', './samples_reduced/samples_latest_JESDown/')
    #doLoop('./condor/reduceTree_JMRUp.py', './samples_latest/JMRUp/', './samples_latest_700.txt', './samples_reduced/samples_latest_JMRUp/')
    #doLoop('./condor/reduceTree_JMRDown.py', './samples_latest/JMRDown/', './samples_latest_700.txt', './samples_reduced/samples_latest_JMRDown/')
