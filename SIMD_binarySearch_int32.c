# include <stdio.h>
# include <stdlib.h>    // Dynamic mem alloc
# include <immintrin.h> // SIMD
# include <stdint.h>    // So SIMD has aligned types
# include <stdbool.h>
# include <time.h>      // Peformance messuring

// Memory allocation with NULL return check
void* malloc_safe(const size_t size) {
    void *memoryBlock = malloc(size);
    if (memoryBlock == NULL) {
        perror("Memory allocation failed!"); abort();
    }
    return memoryBlock;
}

void* calloc_safe(const size_t elements, const size_t typeSize) {
    void *allocatedArray = calloc(elements, typeSize);
    if (allocatedArray == NULL) {
        perror("Memory allocation failed!"); abort();
    }
    return allocatedArray;
}


struct STreeNode {
    int32_t *keys;
    uint32_t *values;

    struct STreeNode **branches;
};

struct STreeNode* createSTreeNode(const uint8_t parallelCompares) {
    struct STreeNode* newNode = (struct STreeNode*)malloc_safe(sizeof(struct STreeNode));

    newNode->keys = calloc_safe(parallelCompares, sizeof(int32_t));
    newNode->values = calloc_safe(parallelCompares, sizeof(uint32_t));
    newNode->branches = calloc_safe(parallelCompares + 1, sizeof(struct STreeNode*));

    return newNode;
}

struct STree {
    uint8_t parallel_cmps;
    struct STreeNode *head;
};

struct STree* createSTree(const uint8_t parallelCompares) {
    struct STree* newSTree = (struct STree*)malloc_safe(sizeof(struct STree));

    newSTree->parallel_cmps = parallelCompares;
    newSTree->head = createSTreeNode(parallelCompares);

    return newSTree;
}

struct binSearchStep {
    size_t min;
    size_t max;

    struct STreeNode *node;
};


struct STree* create_binSearch_STree(const uint8_t parallelCompares, int32_t *sortedArray, const size_t arrLen) {
    struct STree *binSearch_STree = createSTree(parallelCompares);

    struct binSearchStep *regionsToAdd = calloc_safe(arrLen, sizeof(struct binSearchStep));

    // Set first value
    regionsToAdd[0].min = 0;
    regionsToAdd[0].max = arrLen;   // One greater than largest index
    regionsToAdd[0].node = binSearch_STree->head;

    size_t countOfRegions = 1;
    while (countOfRegions != 0) {
        // Take from anywhere, order doesn't matter
        struct binSearchStep curRegion = regionsToAdd[--countOfRegions];
        
        // Fill min and max of subregion helper array
        size_t subregionBounds[parallelCompares + 2];
        subregionBounds[0] = curRegion.min;
        subregionBounds[parallelCompares+1] = curRegion.max;

        double subregionWidth = (double)(curRegion.max - curRegion.min) / (parallelCompares + 1);
        //printf("min: %llu, max: %llu, step: %.2lf\n  [", curRegion.min, curRegion.max, subregionWidth);

        // Fill the STree node from the helper array
        for (size_t step=1; step<parallelCompares+1; step++) {
            subregionBounds[step] = (size_t)(curRegion.min + step*subregionWidth);
            curRegion.node->keys[step - 1] = sortedArray[ subregionBounds[step] ];
            curRegion.node->values[step - 1] = subregionBounds[step]; // "Value" is equal to the key's index

            //printf("%d, ", subregionBounds[step]);
        }
        //printf("\b\b]\n");

        // Current region has included all adjacent indexes
        if (curRegion.max - curRegion.min <= parallelCompares) continue;

        for (size_t i=0; i < parallelCompares+1; i++ ) {
            curRegion.node->branches[i] = createSTreeNode(parallelCompares);

            regionsToAdd[countOfRegions].min = subregionBounds[i];
            regionsToAdd[countOfRegions].max = subregionBounds[i+1];
            regionsToAdd[countOfRegions++].node = curRegion.node->branches[i];
        }
            
    }
    free(regionsToAdd);
    return binSearch_STree;
}


void printKeys(const struct STreeNode *nodeToPrint, const size_t keys) {
    printf("[");
    for (size_t i=0; i<keys; i++) printf("%d, ", nodeToPrint->keys[i]);
    printf("\b\b]\n");
}

// Return the largest key, less than or equal to the input
int32_t binSIMD_closestLssEql_SSE(const int32_t keyToFind, const struct STree *treeToSearch) {
    struct STreeNode *curNode = treeToSearch->head;
    //printf("\n");

    // Fill the register with copies of the key
    __m128i packedKeyToFind = _mm_set1_epi32(keyToFind);
    
    while (1) {
        // Load all the keys in the node into a vector
        __m128i searchKeys = _mm_load_si128( (__m128i*)curNode->keys );

        // Check for equality with any in node
        int32_t eqlMask = _mm_movemask_epi8(_mm_cmpeq_epi32(packedKeyToFind, searchKeys));
        if (eqlMask != 0) {
            //printf("EQL AT INDEX: %d\n", __builtin_ctz(eqlMask) / sizeof(keyToFind));
            //printKeys(curNode, treeToSearch->parallel_cmps);
            return curNode->values[ __builtin_ctz(eqlMask) / sizeof(keyToFind) ];
        }

        // Check which branch to go to next
        __m128i lessThan = _mm_cmplt_epi32(packedKeyToFind, searchKeys);
        int lessThanMask = _mm_movemask_epi8(lessThan);

        // So trailing zeros will be at most greatest branch index
        lessThanMask |= 1 << treeToSearch->parallel_cmps * sizeof(keyToFind);

        size_t lessThanUpTo = __builtin_ctz(lessThanMask) / sizeof(keyToFind);
        //printf("Less than up to: %d (%d)\n", lessThanUpTo, lessThanMask);
        //printKeys(curNode, treeToSearch->parallel_cmps);

        // At leaf; cant go any lower
        if (curNode->branches[0] == NULL) {
            //printf("None!\n");
            // lessThanUpTo == 0 only if there is nothing less than the input
            return (lessThanUpTo != 0)? curNode->values[lessThanUpTo - 1] : -1;
        }

        curNode = curNode->branches[lessThanUpTo];
    }
}

// Return the largest key, less than or equal to the input
int32_t binSIMD_closestLssEql_AVX(const int32_t keyToFind, const struct STree *treeToSearch) {
    struct STreeNode *curNode = treeToSearch->head;
    //printf("\n");

    // Fill the register with copies of the key
    __m256i packedKeyToFind = _mm256_set1_epi32(keyToFind);
    
    while (1) {
        // Load all the keys in the node into a vector
        __m256i searchKeys = _mm256_load_si256( (__m256i*)curNode->keys );

        // Check for equality with any in node
        __m256i equalVector = _mm256_cmpeq_epi32(packedKeyToFind, searchKeys);  // AVX2
        int equalMask = _mm256_movemask_epi8(equalVector);

        if (equalMask != 0) {
            //printf("EQL TO AN ELEMENT\n");
            //printKeys(curNode, treeToSearch->parallel_cmps);
            return curNode->values[ __builtin_ctz(equalMask) / sizeof(keyToFind) ];
        }
        
        // Check which branch to go to next
        __m256i lessThan = _mm256_cmpgt_epi32(searchKeys, packedKeyToFind); //AVX2
        int32_t lessThanMask = _mm256_movemask_epi8(lessThan);          //AVX2

        size_t lessThanUpTo = __builtin_ctz(lessThanMask) / sizeof(keyToFind);
        //printf("Less than up to: %d (%d)\n", lessThanUpTo, lessThanMask);
        //printKeys(curNode, treeToSearch->parallel_cmps);

        // At leaf; cant go any lower
        if (curNode->branches[0] == NULL) {
            //printf("None!\n");
            // lessThanUpTo == 0 only if there is nothing less than the input
            return (lessThanUpTo != 0)? curNode->values[lessThanUpTo - 1] : -1;
        }

        curNode = curNode->branches[lessThanUpTo];
        
    }
}

// Search S-Tree without special SIMD instructions
int32_t binSIMD_linear(const int32_t keyToFind, const struct STree *treeToSearch) {
    struct STreeNode *curNode = treeToSearch->head;
    //printf("\n");

    
    while (1) {
        // Find the next branch to go to
        size_t lessThanUpTo = 0;
        for (; lessThanUpTo < treeToSearch->parallel_cmps; lessThanUpTo++) {
            if (curNode->keys[lessThanUpTo] >= keyToFind) break;
        }

        if (curNode->keys[lessThanUpTo] == keyToFind) {
            //printf("EQL TO AN ELEMENT\n");
            //printKeys(curNode, treeToSearch->parallel_cmps);
            return curNode->values[lessThanUpTo];
        }

        //printf("Less than up to: %d\n", lessThanUpTo);
        //printKeys(curNode, treeToSearch->parallel_cmps);

        // At leaf; cant go any lower
        if (curNode->branches[0] == NULL) {
            //printf("None!\n");
            // lessThanUpTo == 0 only if there is nothing less than the input
            return (lessThanUpTo != 0)? curNode->values[lessThanUpTo - 1] : -1;
        }

        curNode = curNode->branches[lessThanUpTo];
        
    }
}


void printNodeBranches(struct STreeNode *nodeToPrint, const size_t elementsPerNode, const size_t DEPTH) {
    if (nodeToPrint == NULL) return;

    for (size_t i=0; i < DEPTH; i++) printf("\t");
    printKeys(nodeToPrint, elementsPerNode);

    for (size_t i=0; i < elementsPerNode+1; i++)
        printNodeBranches(nodeToPrint->branches[i], elementsPerNode, DEPTH+1);
}

void printSTree(const struct STree *treeToPrint) {
    printNodeBranches(treeToPrint->head, treeToPrint->parallel_cmps, 0);
}

// Traditional method to compare with
int32_t binarySearch(int32_t *array, int32_t x, int32_t high) {
    int32_t low = 0;

    // Repeat until the pointers low and high meet each other
    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (array[mid] == x)
        return mid;

        if (array[mid] < x)
        low = mid + 1;

        else
        high = mid - 1;
    }

    return high;
}


unsigned rndNumber(const unsigned lowerBound, const unsigned upperBound) {
    const unsigned PRIME = 65537;
    return (rand()*PRIME) % (upperBound - lowerBound + 1) + lowerBound;
}

int main() {
    // Change so size of SIMD vector is filled with type (SSE: 4, AVX: 8)
    const uint8_t parralelCompares = 8;
    size_t elements = 100000;

    // Example array (has to be sorted)
    //int32_t orderedNums[elements];
    //for (size_t i=0; i<elements; i++) orderedNums[i] = i;

    //int16_t orderedNums[] = {2,5,8,12,16,23,38,56,72,91};
    
    // Create array full of equally spaced values from min to max
    int32_t *orderedNums = calloc_safe(elements, sizeof(int32_t));
    orderedNums[0] = -((int64_t)1 << 32-1);

    int32_t running_count = -((int64_t)1 << 32-1); // 32-bit min
    for (size_t i=1; i < elements; i++) {
        running_count += ((size_t)1 << 32) / elements;
        orderedNums[i] = running_count;
    }


    struct STree *newSTree = create_binSearch_STree(parralelCompares, orderedNums, elements);


    // Print the created STree
    //printf("\n\n");
    //printSTree(newSTree);

    
    //printf("%d\n", orderedNums[ binSIMD_linear(99, newSTree) ]);
    //for (size_t i=0; i<elements; i++)
    //    printf("%d vs %d\n", i, orderedNums[ binSIMD_linear(i, newSTree) ]);
    
    // Performance messuring
    clock_t start_t = clock();

    int32_t foundNum;
    int32_t numToFind;
    for (size_t i=0; i< 50000000; i++) {
        numToFind = ((size_t)1 << 32)* rand() / RAND_MAX;
        // binarySearch(orderedNums ,numToFind, elements)
        //printf("%d closest? %d\n", numToFind, orderedNums[ binSIMD_closestLssEql_SSE(numToFind, newSTree) ]);
        
        foundNum = binSIMD_closestLssEql_AVX(numToFind, newSTree);
        //foundNum = binarySearch(orderedNums ,numToFind, elements);
    }
    clock_t end_t = clock();

    printf("%d\nSeconds to finish: %.2f\n", foundNum, (float)(end_t-start_t) / CLOCKS_PER_SEC);
    
    
}