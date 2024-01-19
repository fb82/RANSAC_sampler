#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sampler_numpy(n,k,m):
    p = np.floor(np.random.rand(m,k) * np.arange(n,n-k,-1)[np.newaxis, :]).astype(int)  
    for i in range(1,k):
        # equivalent to an iteration of insertion sort (lazy implementation)
        p[:, 0:i] = np.sort(p[:, 0:i])

        for j in range(i):
            p[:, i] = p[:, i] + (p[:, i] >= p[:, j])

    return p


def sampler_torch(n, k, m):

    idx = (torch.rand((m, k), device=device) * torch.arange(n, n-k, -1, device=device).unsqueeze(0).repeat(m,1)).type(torch.long)
    
    for k in range(1,k):
        # equivalent to an iteration of insertion sort (lazy implementation)
        idx[:, 0:k] = torch.sort(idx[:, 0:k])[0]

        for kk in range(k):
            idx[:, k] = idx[:, k] + (idx[:, k] >= idx[:, kk])

    return idx


if __name__ == '__main__':

    # number of objects
    n = 8000
    # RANSAC common k values are 3, 4, 7, 8
    k = 8
    # how many "batch" permutation to return 
    m = 500    
    # iteration to test    
    it = 50    
    print(f'n={n}, k={k}, m={m}')

    # numpy
    #
    start = time.time()
    for t in range(it):
        tmp = sampler_numpy(n,k,m)        
    end = time.time()
    print("The O(k^2) numpy implementation - Elapsed = %s s" % ((end - start)/t))    
    # print(tmp)

    start = time.time()
    for t in range(it):
        tmp = np.zeros((m,k)).astype(int)
        for i in range(m):
            tmp[i] = np.random.choice(n, size=k, replace=False)         
    end = time.time()
    print("Base O(n) numpy implementation - Elapsed = %s s" % ((end - start)/t))
    # print(tmp)


    # pytorch
    #
    start = time.time()
    for t in range(it):
        tmp = sampler_torch(n,k,m)        
    end = time.time()
    print("The O(k^2) pytorch implementation - Elapsed = %s s" % ((end - start)/t))    
    # print(tmp)
    
    start = time.time()
    for t in range(it):
        tmp = torch.zeros((m,k), device=device, dtype=torch.long)
        for i in range(m):
            tmp[i] = torch.randperm(n)[:k].to(device)      
    end = time.time()
    print("Base O(n) pytorch implementation - Elapsed = %s s" % ((end - start)/t))
    # print(tmp)
