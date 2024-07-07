from matplotlib import pyplot as plt
import numpy as np
from math import floor
from multipledispatch import dispatch


class SampleSetPlot:
    def __init__(self,
        points: np.ndarray, c: str = "red", s: float = 3.0, m: str = "o",
        highlight: bool = False, highlight_c: str = "blue", highlight_a: float = .15,
        overlaps: bool = False, overlaps_c: str = "red", overlaps_a: float = .3,
        voids: bool = False, voids_c: str = "grey", voids_a: float = .15
    ):
        self.points = points
        self.N, self.P = points.shape
        self.c, self.s, self.m = c, s, m
        self.highlight, self.highlight_c, self.highlight_a = highlight, highlight_c, highlight_a,
        self.overlaps, self.overlaps_c, self.overlaps_a = overlaps, overlaps_c, overlaps_a
        self.voids, self.voids_c, self.voids_a = voids, voids_c, voids_a
        self._ovs = None
        if self.P != 2:
            raise ValueError("Invalid bidimensional sample set dimension. points.shape[1] must be equal to 2.")
        

    def interval_pairs_gen(self, Ntot=None):
        if Ntot is None:
            Ntot = self.points.shape[0]
        ovs = np.zeros((Ntot, 2), dtype=int)
        timestep = 1/Ntot
        for i in range(self.N):
            qv = floor(self.points[i, 0]/timestep)
            qh = floor(self.points[i, 1]/timestep)
            if 0 <= qv < Ntot:
                ovs[qv][0] += 1
            if 0 <= qh < Ntot:
                ovs[qh][1] += 1
            yield qv, qh
        
        self._ovs = ovs
    

    def trace_overlaps(self, Ntot = None):
        if self._ovs is not None:
            return self._ovs
        else:
            list(self.interval_pairs_gen(Ntot))
            return self._ovs


@dispatch(
    np.ndarray, Ntot=int, c=str, s=float, m=str,
    grid=bool, grid_c=str,
    highlight=bool, highlight_c=str, highlight_a=float,
    interval_labels=bool, caption=str,
    overlaps=bool, overlaps_c=str, overlaps_a=float,
    voids=bool, voids_c=str,  voids_a=float,
    save=bool, filepath=str, exclusive_labels=bool)
def usePlotSampleSet(
    points: np.ndarray, Ntot: int = -1, c: str = "red", s: float = 3.0, m: str = "o",
    grid: bool = False, grid_c: str = "black",
    highlight: bool = False, highlight_c: str = "blue", highlight_a: float = .15,
    overlaps: bool = False, overlaps_c: str = "red", overlaps_a: float = .3,
    voids: bool = False, voids_c: str = "grey", voids_a: float = .15,
    interval_labels: bool = False, caption: str = None,
    save: bool = False, filepath: str = None, exclusive_labels:bool = False
):
    def plotSampleSetFacade( 
        points: np.ndarray = points, c: str = c, s: float = s, m: str = m,
        Ntot: int = Ntot,
        grid: bool = grid, grid_c: str = grid_c,
        highlight: bool = highlight, highlight_c: str = highlight_c, highlight_a: float = highlight_a,
        overlaps: bool = overlaps, overlaps_c: str = overlaps_c, overlaps_a: float = overlaps_a,
        voids: bool = voids, voids_c: str = voids_c, voids_a: float = voids_a,
        interval_labels: bool = interval_labels, caption: str = caption,
        save: bool = save, filepath: str = filepath, exclusive_labels:bool=exclusive_labels
    ):
        return usePlotSampleSet([SampleSetPlot(
            points, c, s, m, highlight, highlight_c, highlight_a, 
            overlaps, overlaps_c, overlaps_a,
            voids, voids_c, voids_a
        )], Ntot=(points.shape[0] if Ntot == -1 else Ntot), grid=grid, grid_c=grid_c,
            interval_labels=interval_labels, caption=caption,
            save=save, filepath=filepath, exclusive_labels=exclusive_labels)()

    return plotSampleSetFacade


@dispatch(
    list, Ntot=int, 
    grid=bool, grid_c=str,  
    interval_labels=bool, caption=str,
    save=bool, filepath=str, exclusive_labels=bool)
def usePlotSampleSet(
    sample_sets: list[SampleSetPlot], Ntot: int = -1,
    grid: bool = False, grid_c: str = "black",
    interval_labels: bool = False, caption: str = None,
    save: bool = False, filepath: str = None, exclusive_labels:bool = False
):
    def plotSampleSet(
        sample_sets: list[SampleSetPlot] = sample_sets,
        Ntot: int = Ntot,
        grid: bool = grid, grid_c: str = grid_c,
        interval_labels: bool = interval_labels, caption: str = caption,
        save: bool = save, filepath: str = filepath, exclusive_labels:bool = exclusive_labels
    ):
        if Ntot == -1:
            Ntot = np.sum([ss.points.shape[0] for ss in sample_sets])
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if grid:
            for q in range(0, Ntot):
                ax.axhline(y=q/Ntot, color=grid_c, linestyle='--', linewidth=0.3)
                ax.axvline(x=q/Ntot, color=grid_c, linestyle='--', linewidth=0.3)
        
        for ss in sample_sets:
            ax.scatter(ss.points[:, 0], ss.points[:, 1], marker=ss.m, c=ss.c, s=ss.s)
            if ss.highlight:
                for qv, qh in ss.interval_pairs_gen(Ntot):
                    ax.axhspan(qh/Ntot, (qh+1)/Ntot, facecolor=ss.highlight_c, alpha=ss.highlight_a)
                    ax.axvspan(qv/Ntot, (qv+1)/Ntot, facecolor=ss.highlight_c, alpha=ss.highlight_a)

            if ss.overlaps:
                ovs = ss.trace_overlaps(Ntot)
                for i in range(Ntot):
                    if ovs[i][0] == 2:
                        ax.axvspan(i/Ntot, (i+1)/Ntot, facecolor=ss.overlaps_c, alpha=ss.overlaps_a)
                    if ovs[i][1] == 2:
                        ax.axhspan(i/Ntot, (i+1)/Ntot, facecolor=ss.overlaps_c, alpha=ss.overlaps_a)

            if ss.voids:
                ovs = ss.trace_overlaps(Ntot)
                for i in range(Ntot):
                    if ovs[i][0] == 0:
                        ax.axvspan(i/Ntot, (i+1)/Ntot, facecolor=ss.voids_c, alpha=ss.voids_a)
                    if ovs[i][1] == 0:
                        ax.axhspan(i/Ntot, (i+1)/Ntot, facecolor=ss.voids_c, alpha=ss.voids_a)

        if interval_labels:
            # ticks are place in the middle of the interval
            xticks = yticks = np.arange(0, 1, 1/Ntot) + 1/(2*Ntot)
            xlabels = ylabels = np.arange(1, Ntot+1, 1)
            if exclusive_labels:
                ovs = ss.trace_overlaps(Ntot)
                yticks = yticks[ovs[:, 1] >= 1]
                ylabels = ylabels[ovs[:, 1] >= 1]
                xticks = xticks[ovs[:, 0] >= 1]
                xlabels = xlabels[ovs[:, 0] >= 1]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
            ax.tick_params(axis='both', which='both', length=0, width=0)

        if caption is not None or caption != "":
            fig.suptitle(caption, x=0.5, y=0.935, fontsize=12, fontweight='bold')

        if save and filepath is not None:
            fig.savefig(filepath, dpi=440, bbox_inches='tight')
        else:
            ax.plot()
    
    return plotSampleSet


# # no parameters dispatch
# @dispatch(Ntot=int, c=str, s=float, m=str,
#     grid=bool, grid_c=str, 
#     interval_labels=bool, caption=str,
#     save=bool, filepath=str)
# def usePlotSampleSet(
#     Ntot: int = -1, c: str = "red", s: float = 3.0, m: str = "o",
#     grid: bool = False, grid_c: str = "black",
#     highlight: bool = False, highlight_c: str = "blue", highlight_a: float = .15,
#     overlaps: bool = False, voids_c: str = "grey", overlaps_c: str = "red", overlaps_a: float = .3,
#     interval_labels: bool = False, caption: str = None,
#     save: bool = False, filepath: str = None
# ):
#     def plotSampleSetFacade( 
#         points: np.ndarray = np.empty((0, 2), dtype=int), 
#         c: str = c, s: float = s, m: str = m,
#         Ntot: int = Ntot,
#         grid: bool = grid, grid_c: str = grid_c,
#         highlight: bool = highlight, highlight_c: str = highlight_c, highlight_a: float = highlight_a,
#         overlaps: bool = overlaps, voids_c: str = voids_c, overlaps_c: str = overlaps_c, overlaps_a: float = overlaps_a,
#         interval_labels: bool = interval_labels, caption: str = caption,
#         save: bool = save, filepath: str = filepath
#     ):
#         return recursion_proxy([SampleSetPlot(
#             points, c, s, m, highlight, highlight_c, highlight_a, 
#             overlaps, voids_c, overlaps_c, overlaps_a
#         )],
#         (points.shape[0] if Ntot == -1 else Ntot), 
#         grid=grid, grid_c=grid_c,
#         interval_labels=interval_labels, caption=caption,
#         save=save, filepath=filepath)()
    
#     return plotSampleSetFacade

# def recursion_proxy(
#     sample_sets: list[SampleSetPlot], Ntot: int,
#     grid: bool = False, grid_c: str = "black",
#     interval_labels: bool = False, caption: str = None,
#     save: bool = False, filepath: str = None
# ):
#     return usePlotSampleSet(
#         sample_sets,
#         Ntot,
#         grid=grid, grid_c=grid_c,
#         interval_labels=interval_labels, caption=caption,
#         save=save, filepath=filepath)


# @dispatch(int, z=str)
# def pippo(a:int, z:str = "not set1"):
#     print(a, z, "pippo1")

# @dispatch(float, z=str)
# def pippo(a:int, z:str = "not set2"):
#     print(a, z, "pippo2")

# def pippo(z:str = "not set3"):
#     print(z, "pippo3!!!")

