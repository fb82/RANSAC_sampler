# Selection of m samples of k elements with no repetition from n indexes

O(k^2) implementation faster than the base O(n) when k<<n as in case of RANSAC model selection.

note: returned samples are ordered with the exception of the last element

-
RANSAC_sampler.py base test output:
```
n=8000, k=8, m=500
The O(k^2) numpy implementation - Elapsed = 0.00021175948940977757 s
Base O(n) numpy implementation - Elapsed = 0.04190991362746881 s
The O(k^2) pytorch implementation - Elapsed = 0.0014830316816057479 s
Base O(n) pytorch implementation - Elapsed = 0.05331532809199119 s
```
