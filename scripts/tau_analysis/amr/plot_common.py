import base64
import re
import requests

import glob as glob
import matplotlib.figure as pltfig
import matplotlib.pyplot as plt

from io import BytesIO
from typing import Union

label_map = {
    "tau:AR1": "$FC_{CN}$",
    "tau:SR": "$BC_{NO}$",
    "tau:AR2": "$FD_{CO}$",
    "tau:AR3_UMBT": "$AG_{NO}$",
    "tau:AR3-AR3_UMBT": "$LB_{NO}$",
    "tau:AR3_LB": "$LB_{NO}$",
    "tau:AR3": "$AG+LB_{NO}$",
    "msgcnt:BoundaryComm": "MsgCount (BC)",
    "msgsz:BoundaryComm": "MsgSize (BC)",
    "npeer:BoundaryComm": "NumPeers (BC)",
    "rcnt:": "Load",
}

profile_label_map = {
    "profile10": "parth-baseline-1",
    "profile22": "parth-baseline-2",
    "profile23": "lpt-buggy-noderef",
    "profile24": "lpt-fixed-derefs",
    "profile25": "lpt-fixed-derefs",
    "profile26": "lpt-fixed-derefs",
    "profile28": "lpt-fixed-costs",
    "profile29": "lpt-lb-onlyB",
    "profile30": "contig-improved",
    "profile31": "parth-baseline-nomsglog",
    "profile32": "lpt-fixed-costs-nomsglog",
    "profile33": "contig-improved-nomsglog",
    "profile34": "parth-baseline-nomsglog-2",
    "profile35": "lpt-fixed-costs-nomsglog-2",
    "profile36": "contig-improved-nomsglog-2",
    "profile37": "parth-upd-noout-baseline",
    "profile38": "parth-upd-noout-lpt",
    "profile39": "parth-upd-noout-contig++",
    "profile40": "parth-upd-outnderef-baseline",
    "profile41": "parth-upd-outnderef-lpt",
    "profile42": "parth-upd-outnderef-contig++",
    "profile43": "parth-upd-outnderef-cppiter",
    "profile44": "parth-upd-outnderef-cppiter-lb1k",
    "burgers1": "parth-vibe-baseline",
    "burgers2": "parth-vibe-bl-packsz1",
    "athenapk1": "glxcool-baseline",
    "athenapk2": "glxcool-pack1",
    "athenapk4": "glxcool-p1-100k",
    "athenapk5": "glxcool-p1-thrott0",
    "athenapk7": "glxcool-postopt",
    "athenapk10": "glxcool-postopt-slot64",
    "athenapk11": "glxcool-postopt-slot64-2",
    "athenapk12": "glxcool-postopt-slot64-3",
    "athenapk13": "glxcool-postopt-slot64-4",
    "athenapk14": "glxcool-postopt-baseline",
    "athenapk15": "glxcool-postopt-lpt",
    "athenapk16": "glxcool-postopt-cpp",
    "stochsg2": "StochSG - Test",
}

prof_evt_map = {"0": "FluxDivergence", "1": "CalculateFluxes", "0+1": "FD+CF"}


def get_label(key: str):
    if key in label_map:
        return label_map[key]

    key = f"tau:{key}"
    return label_map[key]


def plot_init():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 26

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_init_big():
    SMALL_SIZE = 15
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("legend", fontsize=14)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_init_print():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc(
        "font", size=SMALL_SIZE
    )  # controls default text sizes plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y label
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_dir_latest() -> str:
    repo_root = "/users/ankushj/repos/amr/scripts/tau_analysis"
    glob_patt = f"{repo_root}/figures/202*"

    all_dirs = glob.glob(glob_patt)

    def get_key(x: str) -> int:
        mobj = re.search(r"202[0-9]+", x)
        if mobj:
            return int(mobj.group(0))
        else:
            return -1

    dir_latest = max(all_dirs, key=get_key)
    return dir_latest


class PlotSaver:
    @staticmethod
    def save(
        fig: pltfig.Figure,
        trpath: Union[str, None],
        fpath: Union[str, None],
        fname: str,
    ):
        PlotSaver._save_to_fpath(fig, trpath, fpath, fname, ext="png", show=False)
        # PlotSaver._send_to_server(fig)

    @staticmethod
    def _save_to_fpath(
        fig: pltfig.Figure,
        trpath: Union[str, None],
        fpath: Union[str, None],
        fname: str,
        ext="png",
        show=True,
    ):
        trpref = ""
        if trpath is not None:
            if "/" in trpath:
                trpref = trpath.split("/")[-1] + "_"
            elif len(trpath) > 0:
                trpref = f"{trpath}_"

        if fpath is None:
            fpath = plot_dir_latest()

        full_path = f"{fpath}/{trpref}{fname}.{ext}"

        if show:
            print(f"[PlotSaver] Displaying figure\n")
            fig.show()
        else:
            print(f"[PlotSaver] Writing to {full_path}\n")
            fig.savefig(full_path, dpi=300)

    @staticmethod
    def _send_to_server(fig: pltfig.Figure) -> None:
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        response = requests.post(
            "http://127.0.0.1:5000/update_plot",
            json={"plot_data": plot_data},
            proxies={"http": None, "https": None},
        )
        if response.status_code != 200:
            print(f"Failed: resp code {response.status_code}")

