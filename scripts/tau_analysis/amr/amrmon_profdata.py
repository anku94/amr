import pandas as pd
import numpy as np

class ProfileData:
    def __init__(self, trace_dir: str):
        self.trace_dir: str = trace_dir
        lb_dfpath = f"{trace_dir}/lb_aggr.feather"
        self.lb_data: pd.DataFrame = pd.read_feather(lb_dfpath).astype({"id": int})
        self.lb_data["tsid"] = pd.factorize(self.lb_data["timestep"])[0]
        self.lb_data.set_index("id", inplace=True)

        names_dfpath = f"{trace_dir}/lb_names.feather"
        lb_names: pd.DataFrame = pd.read_feather(names_dfpath)
        self.name_map: dict[str, int] = lb_names.set_index("name")["id"].to_dict()

    def lookup_name(self, name: str) -> int:
        if name in self.name_map:
            return self.name_map[name]
        return -1

    def lookup_name_substr(self, name: str) -> int:
        name_match = [k for k in self.name_map.keys() if name in k]

        if len(name_match) == 0:
            return -1

        if len(name_match) > 1:
            print(f"Multiple matches for {name}: {name_match}")

        return self.name_map[name_match[0]]

    def get_matrix(self, name: str, approx: bool = False) -> np.ndarray:
        name_id = self.lookup_name(name)
        if approx and name_id == -1:
            name_id = self.lookup_name_substr(name)

        if name_id == -1:
            print(f"Name {name} not found")
            return np.array([])

        df_name = self.lb_data.loc[name_id]
        mat_df = df_name.pivot(index="tsid", columns="rank", values="time")
        mat = mat_df.to_numpy()

        return mat
