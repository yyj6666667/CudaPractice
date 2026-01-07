## select top-k element from a large array

### version:
 
* naive: k次kernel遍历， 每次找到一个最大值 O(k * N)
* v1   : 1次遍历， 维护一个最小堆, len = k