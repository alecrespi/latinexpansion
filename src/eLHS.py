import numpy as np
from typing import List
from sortedcontainers import SortedSet
from utils import high_precision_difference as hpdiff
from scipy.stats.qmc import LatinHypercube as LHSampler
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from math import floor
from pprint import pp
import random


class Range: 
    def __init__(self, low: float, up: float):
        if not 0 <= low < up <= 1:
            raise ValueError("Range must be in [0, 1] and low < up")
        self.low = low
        self.up = up

    def get(self):
        return (self.low, self.up)
    
    def width(self):
        return hpdiff(self.up, self.low)
    
    def isRangeable(x: object):
        try:
            return \
                (type(x) == list or type(x) == np.ndarray) \
                and len(x) == 2 \
                and Range(x[0], x[1]) is not None
        except (ValueError, TypeError):
            return False

    def compare(self, range: object) -> int:
        r = None
        if type(range) == Range:
            r = range
        elif Range.isRangeable(range):
            r = Range(range[0], range[1])
        else:
            raise ValueError("Invalid range parameter, must be a Range object or a list of two elements [a, b] with 0 <= a < b <= 1.")
        
        diff = np.array(self.get()) - np.array(r.get())
        if diff[0] != 0:
            return diff[0]
        else:
            return diff[1]
        
    def __hash__(self) -> int:
        return hash((self.low, self.up))

    def __eq__(self, r: object) -> bool:
        return self.compare(r) == 0
    
    def __neq__(self, r: object) -> bool:
        return not self.__eq__(r)
    
    def __lt__ (self, r: object) -> bool:
        return self.compare(r) < 0

    def __le__ (self, r: object) -> bool:
        return self.compare(r) <= 0

    def __gt__ (self, r: object) -> bool:
        return self.compare(r) > 0
    
    def __ge__ (self, r: object) -> bool:
        return self.compare(r) >= 0
    
    def __str__(self) -> str:
        return self.get().__str__()
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __contains__(self, x: float) -> bool:
        return self.low <= x <= self.up
    
    def contains_list(self, x: object):
        if type(x) is list:
            return [self.__contains__(item) for item in x]
        else:
            return self.__contains__(x)
    
class RangeGroup:
    def __init__(self, ranges: List[object] = []):
        if not all([type(r) == Range or Range.isRangeable(r) for r in ranges]):
            raise ValueError("Invalid range parameter, must be a Range object or a list of two elements [low, up] with 0 <= low < up <= 1.")
        self.ranges = SortedSet(ranges)

    def __str__(self):
        return str(tuple(self.ranges))

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.ranges)
    
    def __iter__(self):
        return self.ranges.__iter__()
    
    def __next__(self) -> Range:
        return next(self.ranges)

    def __contains__(self, x: object) -> bool:
        return any([x in range for range in self.ranges])
        
    def __getitem__(self, i: int) -> Range:
        if type(i) == slice:
            return RangeGroup(list(self.ranges)[i])
        else:
            return list(self.ranges)[i]
        
    def contains_list(self, x: object):
        if type(x) is list:
            return [self.__contains__(item) for item in x]
        else:
            return self.__contains__(x)
        
    def remove(self, r: object):
        if type(r) == Range:
            self.ranges.discard(r)
        elif Range.isRangeable(r):
            self.ranges.discard(Range(r[0], r[1]))
        else:
            raise ValueError("Invalid range parameter, must be a Range object or a list of two elements [low, up] with 0 <= low < up <= 1.")
    
    def add(self, r: object):
        if type(r) == Range:
            self.ranges.add(r)
        elif Range.isRangeable(r):
            self.ranges.add(Range(r[0], r[1]))
        else:
            raise ValueError("Invalid range parameter, must be a Range object or a list of two elements [low, up] with 0 <= low < up <= 1.")
    

    class Generator:
        def regular(N: int) -> 'RangeGroup':
            return RangeGroup([Range(i/N, (i+1)/N) for i in range(N)])

class BinningGrid:
    def __init__(self, bins: List[RangeGroup]):
        self.dimensions = len(bins)
        self.bins : List[RangeGroup] = bins
    
    def __getitem__(self, i: int):
        if type(i) == slice:
            return BinningGrid(self.bins[i])
        else:
            return self.bins[i]
    
    def __str__(self) -> str:
        return "BinningGrid{ P = " + str(self.dimensions) + ", bins = " + self.bins.__str__() + " }"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class RegularBinningGrid(BinningGrid):
    def __init__(self, size: int, dimensions: int):
        self.dimensions = dimensions
        self.size = size
        self.bins : List[RangeGroup] = [RangeGroup.Generator.regular(size) for _ in range(dimensions)]

class SampleSet:
    def __init__(self, binning: RegularBinningGrid):
        self.binning = binning
        self.nsamples = 0
        self.samples = None
        self.span = None
        self.__p_grade = None
    
    def fill(self, samples: np.ndarray = None):
        if samples is None:
            self.samples = LHSampler(d = self.binning.dimensions).random(n = self.binning.size)
            self.nsamples = self.binning.size
        else:
            self.samples = samples
            self.nsamples = len(samples)
        
        self.span = self.distances().min()
        self.__p_grade = self.grade(self.binning)

    def grade(self, binning: BinningGrid = None, mode = 1) -> float:
        if binning is None:
            return self.__p_grade
        if binning.dimensions != self.binning.dimensions:
            raise ValueError("Binning dimensions must match the sample set dimensions.")

        N, Q, P = self.nsamples, binning.size, binning.dimensions

       
        if mode == 1:   # hard way: O(p * n^3)
            return \
                np.sum([
                        np.sum([
                            1 if any(
                                [
                                    self.samples[i, j] in binning.bins[j][q]
                                    for i in range(N)
                                ]
                            ) else 0
                            for q in range(Q)
                        ])
                    for j in range(P)
                ]) / (Q * P)
        elif mode == 2:
            pass

    def expanded_grade(self, expansion_binning: BinningGrid) -> float:
        if expansion_binning.dimensions != self.binning.dimensions:
            raise ValueError("Binning dimensions must match.")
        N, M = self.binning.size, expansion_binning.size
        if M == N:
            return self.__p_grade
        elif M < N:
            raise ValueError("Expansion binning must be greater than initial binning.")
        else:
            return self.grade(expansion_binning) + (M - N)/M

    def distances(self):
        return pdist(self.samples)
    
    def plot(
            self, 
            binning: BinningGrid = None, 
            grid: bool = True, 
            highlight: bool = True, 
            save: bool = False, 
            filepath: str = None):
        lhs = self.samples
        if binning is None:
            binning = self.binning
        if binning.dimensions != 2:
            return
        
        
        # scattering sample set
        plt.figure()
        plt.scatter(lhs[:, 0], lhs[:, 1], marker='o', c='r', s=2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        N = binning.size
        # drawing grid
        if grid:
            for q in range(1, N):
                plt.axvline(x = binning.bins[0][q].low, color='black', linestyle='--', linewidth=0.3)
                plt.axhline(y = binning.bins[1][q].low, color='black', linestyle='--', linewidth=0.3)
        
        # highlighting bins
        if highlight:
            for q in range(N):
                for sample in lhs:
                    if sample[0] in binning.bins[0][q]:
                        plt.axvspan(
                            binning.bins[0][q].low, 
                            binning.bins[0][q].up, 
                            facecolor='blue', 
                            alpha=0.15)
                    if sample[1] in binning.bins[1][q]:
                        plt.axhspan(
                            binning.bins[1][q].low, 
                            binning.bins[1][q].up, 
                            facecolor='blue', 
                            alpha=0.15)
            
        # plot or save 
        if save and filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

    def __extract_voids(samples: np.ndarray, binning: BinningGrid) -> BinningGrid:
        voids : List[RangeGroup] = []
        for j in range(binning.dimensions):
            void = RangeGroup()
            for q in range(len(binning[j])):
                if not any([sample[j] in binning.bins[j][q] for sample in samples]):
                    void.add(binning.bins[j][q])
            voids.append(void)
        return BinningGrid(voids)

    def __select_voids(voids_binning: BinningGrid, M: int, samples: np.ndarray = None) -> BinningGrid:
        voids = voids_binning.bins
        P = len(voids)
        selected = np.empty((P), dtype = RangeGroup)
        for j in range(P):
            if len(voids[j]) < M:
                raise ValueError("Not enough voids to fill the new samples.")
            elif len(voids[j]) == M:
                selected[j] = RangeGroup(voids[j].ranges)
            else:
                selected[j] = RangeGroup(random.sample(voids[j].ranges, M))
        return BinningGrid(selected)

    def __gen_local_lhs(M: int, binning: BinningGrid) -> np.ndarray:
        """
            Using the formulae: x_ij = l_ij + u_ij/M;
            where: 
                u_ij ~ U(0,1)
                l_ij = permutation(binning.bins[j][q].low) --> random permutation of lower bounds
        """
        P = binning.dimensions
        samples = np.empty((M, P))
        for j in range(P):
            bounds: RangeGroup = np.copy(binning.bins[j].ranges)
            random.shuffle(bounds)
            for q in range(M):
                samples[q, j] = bounds[q].low + random.random() * bounds[q].width()
            
        return samples

    # eLHS algorithm (not implemented for any BinningGrid)
    def expand(self, exp_binning: RegularBinningGrid, new_samples: int = None) -> "SampleSet":
        P = self.binning.dimensions
        N, M = self.binning.size, new_samples

        if new_samples is None:
            new_samples = exp_binning.size - N
        elif exp_binning.dimensions != P:
            raise ValueError("Binning dimensions must match.")
        if new_samples <= 0:
            raise ValueError("New samples must be grater than zero.")
        
        N, M = self.binning.size, new_samples

        voids: BinningGrid = SampleSet.__extract_voids(self.samples, exp_binning)
        genesys: BinningGrid = SampleSet.__select_voids(voids, M, self.samples)

        # expansion sample set
        expansion: np.ndarray = SampleSet.__gen_local_lhs(M, genesys)

        # final expanded set 
        expandedSS = SampleSet(RegularBinningGrid(N + M, P))   # Regular Binning Grid is temporary
        expandedSS.fill(np.concatenate((self.samples, expansion), axis=0))
        return expandedSS

#Test
B1 = RegularBinningGrid(370, 2)
B2 = RegularBinningGrid(20, 2)
B3 = RegularBinningGrid(530, 2)


BS, BF = B1, B3
err = 1e-12

ss = SampleSet(BS)
ss.fill()

elhs = ss.expand(BF)

# ss.plot(save=True, filepath="./data/initial.png")
elhs.plot(save=True, filepath="./data/eLHS.png")


print("GRADE PREDICTION")
print("Initial Sample Set: ", ss.grade())
print("Predicted: ", pr := ss.expanded_grade(BF), " | ", "✅" if abs(pr - elhs.grade()) <= err else "❌")

