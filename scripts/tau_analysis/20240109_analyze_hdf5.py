import glob
import os
import yt
import multiprocessing
import sys

# fp = "/mnt/ltio/parthenon-topo/blastwave01/run/sedov.out1.00031.phdf"
# ds = yt.load(fp)
# print(ds.field_list)
# ds.print_stats()
# ds.derived_field_list

#  s = yt.SlicePlot(ds, "z", ("parthenon", "advected_0"))
#  s.annotate_grids()
#  s.save()


def plot_frame(file_path: str, plt_dir: str, plot_vars: list[tuple]) -> None:
    if not os.path.exists(file_path):
        return

    file_name = file_path.split("/")[-1]
    file_name = file_name.replace(".phdf", "")
    print(f"Plotting {file_name}")

    dataset = yt.load(file_path)
    for v in plot_vars:
        s = yt.SlicePlot(dataset, "z", v)
        s.annotate_grids()
        #  s.annotate_cell_edges()

        plt_fname = f"{file_name}_{v[1]}"
        plt_fpath = f"{plt_dir}/{plt_fname}.png"
        s.save(plt_fpath)


def plot_frame_wrapper(args):
    plot_frame(*args)


def get_all_phdf(dir_path: str) -> list[str]:
    all_files = glob.glob(dir_path + "/*.phdf")
    return all_files


def run():
    exp_name = "stochsg6"
    #  exp_name = "sparse1"
    dir_path = f"/mnt/ltio/parthenon-topo/{exp_name}/run"
    all_files = get_all_phdf(dir_path)

    #  v = ("parthenon", "sparse_0")
    v = ("parthenon", "advected_10")
    plot_path = f"/users/ankushj/repos/amr/scripts/tau_analysis/figures/20240109/{exp_name}/{v[1]}"
    os.makedirs(plot_path, exist_ok=True)

    print("Input: ", dir_path)
    print("Output: ", plot_path)

    #  vv = ("parthenon", "advected_10")
    #  v = ("parthenon", "a")
    with multiprocessing.Pool(8) as p:
        jobs = [[f, plot_path, [v]] for f in all_files]
        print(jobs)
        p.map(plot_frame_wrapper, jobs)
    pass


if __name__ == "__main__":
    run()
