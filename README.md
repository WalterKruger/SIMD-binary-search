# Algorithm description
### S-Tree
A static B-Tree is constructed by dividing the search region into ‘n+1’ sub-regions, with ‘n’ being the number of parallel comparisons the SIMD instruction set is capable of performing. 
For example, if the search region has a minimum of ‘1’ and max of ‘100’ with n=4, then the head node will have the keys `[20, 40, 60, 80]`. 
A node's keys mark the boundaries of their search region, with their child nodes split in the same way, but with these current keys acting as the new boundaries.
### Searching the S-Tree using SIMD
At each node, a SIMD compare for equality is performed first. When a match is found, the function returns the index of where the marching value is in the original array.

Otherwise, a parallel compare less than is used to find the child node to traverse too next. This child node is analogous to the next region to search. 
For example, if the algorithm finds that the first key greater than the value to find is `40` from the node’s `[20, 40, 60, 80]`, 
then the child node would have the keys `[24, 28, 32, 36]` with is analogous for the region (20, 40).

# Performance
During my testing, **the traditional binary search was either slightly or significantly faster**. This could be due to the worse locality of reference the S-Tree has compared to an array, or that the additional overhead of loading values into a SIMD vector and extracting being more expensive than its advantages.
### Theoretical advantages
The reason why this approach is expected to perform better is that at each step, the current region-to-search is divided into the number of parallel SIMD comparisons + 1, where a traditional binary search only halves the region (Since it only performs a single comparison at a time).

