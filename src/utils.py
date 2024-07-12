from scipy.stats.qmc import LatinHypercube as LHSampler
from matplotlib import cm, pyplot as plt
import numpy as np
from math import floor
import random as r
from decimal import Decimal
import json
from typing import Callable, Generator
import ipywidgets as ws
from IPython.display import display
from time import time
from IPython.display import clear_output

# 2D-only LHS plotter
def plotLHS_old(lhs: np.ndarray, grid: bool = False, highlight: bool = False, save: bool = False, filepath: str = None):
    N, P = lhs.shape
    if P != 2:
        return
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(lhs[:, 0], lhs[:, 1], marker='o', c='r', s=2)
    if grid:
        for q in range(0, N):
            plt.axhline(y=q/N, color='black', linestyle='--', linewidth=0.3)
            plt.axvline(x=q/N, color='black', linestyle='--', linewidth=0.3)
    # floor(coord/interval_size) = interval_index (starting from zero)
    if highlight:
        timestep = 1/N
        for i in range(N):
            try:
                if lhs[i, 0] is None or lhs[i, 0] is None or np.isnan(lhs[i, 0]) or np.isnan(lhs[i, 1]):
                    continue
                qh = floor(lhs[i, 0]/timestep)
                qv = floor(lhs[i, 1]/timestep)
                plt.axvspan(qh/N, (qh+1)/N, facecolor='blue', alpha=0.15)
                plt.axhspan(qv/N, (qv+1)/N, facecolor='blue', alpha=0.15)
            except:
                continue
    if save and filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

# other utilities 
def concat(a: np.ndarray, b:np.ndarray):
    return np.concatenate((a, b), axis=0)

def inner_coords(point, N):
    timestep = 1/N
    # increases rounding error
    return [ point[j] * N - floor(point[j]/timestep) for j in range(len(point))]

def rpop(v: np.ndarray):
    rindex = r.randint(0, len(v) - 1)
    return v[rindex], np.delete(v, rindex, 0)

# Approx Heaviside step function
def F(t, sharpness = 1000):
    return 0.5 * (1 + np.tanh(sharpness * t))

# Approx Heaviside step function
def H(x):
    return np.where(x >= 0, 1, 0)

# high precision difference
def high_precision_difference(a: float, b: float, rounding_error = 30):
    return round(
        float(Decimal(str(a)) - Decimal(str(b))), 
        rounding_error
    )


###################################

# minimum distance metric
def mindist(points):
    points = np.array(points)
    def dist_v_m(point:np.ndarray, pointset:np.ndarray):
        return np.min(np.linalg.norm((pointset - point).T, axis=0))
    def v_m(i:int):
        return points[i], np.vstack((points[:i], points[(i+1):]))
    return np.min([
        dist_v_m(*v_m(i)) for i in range(len(points))
    ])

#utility functions meant to save experiments
class Experiment:
    def __init__(self, id:str, data, description:str):
        self.id = id
        self.datalist = data
        self.description = description
    
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=True,
            indent=0)
    
    def instanceToJSON(obj):
        obj = Experiment(obj["id"], obj["datalist"], obj["description"])
        return obj.toJSON()

def expsave(id:str, data, mode:str, description: str = ""):
    if(mode == "w"):
        with open("./data/" + id + ".json", "w+") as f:
            experiment = Experiment(id, [data], description)
            f.write(experiment.toJSON())
    elif(mode == "a"):
        prevexp = expget(id)
        prevexp["datalist"].append(data)
        if description != "" and description is not None:
            prevexp["description"] = description
        with open("./data/" + id + ".json", "w+") as f:
            f.write(Experiment.instanceToJSON(prevexp))

def expget(id:str):
    f = open("./data/" + id + ".json", "r")
    return json.loads(f.read())

# monte-carlo simulation
def MCSim(f:Callable, xs: np.ndarray, V: float):
    return V * np.mean(np.apply_along_axis(f, 1, xs))

# numerical integration class for P functions
class midpointIntegration:
    def __init__(self, d:int):
        if d <= 0:
            raise ValueError("Dimension must be positive")
        self.d = d
    
    def __gen_indexes(self, Ns, borders = False) -> Generator[list, None, None]:
        index = [0] * self.d
        if not borders:
            Ns = (np.array(Ns) - 1).tolist()
        yield index
        for _ in range(np.prod(Ns)):
            for p in reversed(range(self.d)):
                if index[p] + 1 == Ns[p]:
                    if p == 0:
                        return
                    index[p] = 0
                else:
                    index[p] += 1
                    break
            yield index

    def __relindex(self, abs_index: int, Ns: list[int]):
        rel = np.zeros((self.d), dtype=int)
        r = abs_index
        for p in range(0, self.d):
            rel[p] = floor(r/Ns[p])
            r = r % Ns[p]
        rel[-1] = r
        return rel

    def __absindex(self, rel_index:list[int], Ns:list[int]):
        abs = 0
        for p in range(0, self.d - 1):
            abs += Ns[p] * rel_index[p]
        abs += rel_index[-1]
        return abs

    def __shift_on_axis(self, index, axis: int, Ns: list[int], increment: int = 1):
        if type(index) is int:
            index = self.__relindex(index, Ns)
        rtr = [*index]
        rtr[axis] += increment
        if type(index) is int:
            return self.__absindex(rtr)
        else:
            return rtr

    def midpoint(self, xs, index: list[int], Ns: list[int]):
        m = np.zeros((self.d))
        for p in range(self.d):
            shifted_index = self.__shift_on_axis(index, p, Ns)
            m[p] = (xs[*index, p] + xs[*shifted_index, p])/2
        return m

    # lbs = lower_boundaries
    # ubs = upper_boundaries
    def integrate(self, f:Callable, Ns:list[int], lbs: list[int], ubs: list[int], xs = None):
        if len(Ns) != self.d:
            raise ValueError("Ns must be a list of integers ", self.d, " long")
        
        if xs is None:
            xs = np.zeros((*Ns, self.d))
        for i in self.__gen_indexes(Ns, borders = True):
            xs[*i] = [((i[k]/Ns[k])*(ubs[k] - lbs[k]))+lbs[k] for k in range(self.d)]
            # xs[*i] = np.linspace(lbs[], ubs[k],)
         
        I = 0
        for i in self.__gen_indexes(Ns, borders = False):
            I += f(self.midpoint(xs, i, Ns))
        I *= 1/np.prod(Ns)
        return I

class MC:
    def __init__(self, P):
        self.P = P

    def random(self, N:int) -> np.ndarray:
        return np.random.rand(N, self.P)
    
def plot3d(
    X:np.ndarray, Y:np.ndarray, Z:np.ndarray, 
    elev=100, azim=30, color_bar=False, 
    save=False, fp=None):
    
    def save_plot(_):
        exp_id = fpInput.value if fp is None else fp
        if exp_id is not None and exp_id != "":
            fig = livePlot3d(
                elev=elevSlider.value, 
                azim=azimSlider.value
            )
            fig.savefig("./data/plots/" + exp_id + ".png", dpi=500)

    saveButton = ws.Button(description="SAVE")
    saveButton.on_click(save_plot)

    fpInput = ws.Text(
        value='',
        placeholder='ID ONLY, do not include file extension and path',
        description='Experiment ID: ',
        disabled=False
    )

    elevSlider = ws.IntSlider(
        value=elev,
        min=-90,
        max=90,
        step=1,
        description='Elevation:'
    )
    azimSlider = ws.IntSlider(
        value=azim,
        min=-90,
        max=90,
        step=1,
        description='Azimuth:'
    )
    
    def livePlot3d(elev, azim):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')   

        ax.view_init(elev=elev, azim=azim)
        if color_bar:
            fig.colorbar(surf, shrink=0.5, aspect=5)
        # Plot the surface.
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap=cm.coolwarm,
            linewidth=0, 
            antialiased=False,
            # rcount=len(X), ccount=len(Y),
            # rstride = 100, cstride = 100
        )
        return fig

    if save:
        display(saveButton)
        if fp is None:
            display(fpInput)
    
    ws.interact(livePlot3d, elev=elevSlider, azim=azimSlider)

