# Selection of $m$ samples of $k$ random elements with no repetitions from a population of $n$

This is a $O(k^2)$ implementation, faster than the base $O(n)$ when $k$ $<<$ $n$, as in the case of RANSAC model selection. Can be used with Pytorch batches.

_Note: returned samples are ordered with the exception of the last element._


Output example (just run `RANSAC_sampler.py`):
```
*** Example output ***
n=23, k=4, m=5
-- Numpy --
[[ 3 11 18 12]
 [13 14 19  7]
 [ 0  2 19  5]
 [ 9 11 18  5]
 [11 17 21  7]]

-- Pytorch --
tensor([[ 5, 13, 20,  3],
        [ 4,  7, 18, 11],
        [13, 18, 21, 14],
        [ 0, 13, 17,  3],
        [ 7, 16, 17,  5]], device='cuda:0')

*** Running times ***
n=8000, k=8, m=500
- Base O(n) numpy implementation - Elapsed = 0.047170400619506836 s
- This O(k^2) numpy implementation - Elapsed = 0.00021838168708645568 s
- Base O(n) pytorch implementation - Elapsed = 0.03589857354456065 s
- This O(k^2) pytorch implementation - Elapsed = 0.0012787361534274354 s
```
