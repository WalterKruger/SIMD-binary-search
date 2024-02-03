

class STreeNode:
    def __init__(self):
        self.keys = []
        self.branches = []


# For creating the S-Tree
class binSearchStep:
    def __init__(self, min:int, max:int, node:STreeNode):
        self.min = min
        self.max = max

        self.node = node

    def __str__(self): return f"[{self.min}, {self.middle}, {self.max}]"
    __repr__ = __str__

def main():
    # Example array
    array = [i for i in range(1000)]

    # Number of SIMD comparisons
    parallel_cmps = 8


    print(f"Cur region: {[i for i in range(len(array))]}")
    STree_head = STreeNode()

    # NOTE: max = One greater than largest index
    regionsToAdd = [ binSearchStep(min=0, max=len(array), node=STree_head) ]

    while regionsToAdd:
        curRegion = regionsToAdd.pop(-1)    # Take from anywhere, it doesn't matter

        subregionWidth = (curRegion.max - curRegion.min) / (parallel_cmps + 1)
        subregionBounds = [int(curRegion.min + step*subregionWidth) for step in range(1, parallel_cmps+1) ]
        subregionBounds = [curRegion.min] + subregionBounds + [curRegion.max]

        print(f"min: {curRegion.min}, max: {curRegion.max}, step: {subregionWidth:.2}")
        print(subregionBounds)

        # Min/max is implyed when searching
        curRegion.node.keys = [array[bound] for bound in subregionBounds[1 : -1] ]

        # Current region has included all adjacent indexes
        if subregionWidth <= parallel_cmps/(parallel_cmps+1): continue

        curRegion.node.branches = [ STreeNode() for loops in range( parallel_cmps + 1 ) ]

        for i in range(0, parallel_cmps+1):
            regionsToAdd.append( binSearchStep(subregionBounds[i], subregionBounds[i+1], curRegion.node.branches[i]) )
        
        print("")


    # Print the S-Tree
    print("\n")
    print(STree_head.keys)
    for branch in STree_head.branches:
        print(f"\t{branch.keys}")

        for subBranch in branch.branches:
            print(f"\t\t{subBranch.keys}")

            for subSubBranch in subBranch.branches:
                print(f"\t\t\t{subSubBranch.keys}")





if __name__ == "__main__": main()