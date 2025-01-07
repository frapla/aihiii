import logging
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
from src._StandardNames import StandardNames
from src.utils.custom_log import init_logger

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class PrometheeSortingBased:
    """
    Calculation of PROMETHEE II Complete Pre-Order for scale 'lower is better' and ordinary criteria
    Algorithm adapted from:
    Calders, Toon; Assche, Dimitri van (2018):
    PROMETHEE is not quadratic: An O(qnlog(n)) algorithm. In: Omega 76, S. 63–69. DOI: 10.1016/j.omega.2017.04.003.
    """

    def __init__(self, filtered_crit=None, crit_renamer=None):
        # init
        self.alt_col_name = STR.alternatives
        self.crit_specs = pd.DataFrame()
        self.ranking: Dict[str, pd.Series] = {}
        self.alternative_names: List[str] = []
        self.net_flow_grouped: pd.Series = pd.Series(dtype="float64")

        self.net_str = "net_flow"

        self.filtered_crit = filtered_crit
        self.crit_renamer = crit_renamer

    def get_data(self, in_info: Union[List[pathlib.Path], pathlib.Path, pd.DataFrame]):
        # get data
        if isinstance(in_info, pathlib.Path):
            self.crit_specs = self.__get_assessed_alts_from_csv(csv_path=in_info)
        elif isinstance(in_info, list):
            self.crit_specs = self.__get_assessed_alts_from_txts(files=in_info)
        elif isinstance(in_info, pd.DataFrame):
            self.crit_specs = in_info
        else:
            LOG.critical("Unknown in data type - EXIT")
            sys.exit()

        LOG.info("Data\n%s", self.crit_specs)

        # filter and rename
        if self.crit_renamer:
            self.crit_specs = self.crit_specs.rename(index=self.crit_renamer)
        if self.filtered_crit:
            self.crit_specs = self.crit_specs.loc[self.filtered_crit]

        # to class namespace
        self.alternative_names = self.crit_specs.columns.to_list()

    def execute(self):
        # net flow
        net_flow = self.__calculate_net_flows()
        self.ranking[self.net_str] = pd.Series(net_flow).sort_values(ascending=False)
        self.net_flow_grouped = self.__with_indifferences()

    def __calculate_net_flows(self) -> dict:
        """
        Calculation of PROMETHEE II net flow for scale 'lower is better' and ordinary criteria
        Algorithm adapted from
        Calders, Toon; Assche, Dimitri van (2018):
        PROMETHEE is not quadratic: An O(qnlog(n)) algorithm. In: Omega 76, S. 63–69. DOI: 10.1016/j.omega.2017.04.003.
        :return: Net Flow for all alternatives
        """
        # get data local
        crit_specs: pd.DataFrame = self.crit_specs.copy()

        # set parameters
        num_alts = crit_specs.shape[1]
        numi = 1 / (num_alts - 1)
        weight = 1 / crit_specs.shape[0]

        # flows
        phi_ks = {}

        for criterion in crit_specs.index:
            LOG.debug(f"Process criterion {criterion}")
            # phi pos becomes phi neg if fk was multiplied by -1
            phi_k_pns = {}
            for pos_neg in [1, -1]:
                # # sort criterion
                fk_sort = crit_specs.loc[criterion] * pos_neg
                fk_sort = fk_sort.sort_values(ascending=False, kind="mergesort")
                LOG.debug("Sorted fk:\n%s", fk_sort)
                alts_sorted = fk_sort.index.to_list()
                fk_sorted = fk_sort.values.tolist()

                # init uni criterion flow (first object with the highest value has always no preference)
                phi_k_pn = [0]
                rights = fk_sorted.copy()

                # check preferences of next alternatives in the criterion
                for j in range(1, num_alts):
                    # initial uni criterion flow equal previous one - preferences added
                    phi_k_pn.append(phi_k_pn[j - 1])
                    while rights[0] > fk_sorted[j]:
                        # preference found
                        rights.pop(0)
                        phi_k_pn[j] += numi
                phi_k_pns[pos_neg] = dict(zip(alts_sorted, phi_k_pn))
                LOG.debug("Unicriterial phi%s %s:\n%s", "+" if pos_neg == 1 else "-", criterion, phi_k_pns[pos_neg])
            phi_ks[criterion] = {alt: phi_k_pns[1][alt] - phi_k_pns[-1][alt] for alt in alts_sorted}
            LOG.debug("Unicriterial phi %s:\n%s", criterion, phi_ks[criterion])
        phi = {}
        for alt in crit_specs:
            phi[alt] = sum([weight * phi_ks[crit][alt] for crit in crit_specs.index])

        return phi

    def __get_assessed_alts_from_txts(self, files: List[pathlib.Path]) -> pd.DataFrame:
        # init database
        crit_spec = defaultdict(list)

        LOG.info("Start parsing files")
        for f_path in files:
            # fill database
            crit_spec = self.__parse_alternative_file(f_path=f_path, dic=crit_spec)

        # to dataframe
        crit_spec = pd.DataFrame(crit_spec)
        LOG.info("All files parsed - %s alternatives, %s criteria", crit_spec.shape[0], crit_spec.shape[1])

        crit_spec = crit_spec.set_index(self.alt_col_name)
        crit_spec.columns.name = STR.criteria

        return crit_spec.T

    @staticmethod
    def __get_assessed_alts_from_csv(csv_path: pathlib.Path) -> pd.DataFrame:
        if csv_path.is_file():
            LOG.info("Read assessed alternatives from csv %s", csv_path)
            return pd.read_csv(csv_path, index_col=0)
        else:
            LOG.critical("No file %s - EXIT", csv_path)
            sys.exit()

    def printer(self):
        LOG.info("PROMETHEE II - Complete Preorder\n%s", self.ranking[self.net_str])

    def store(self, out_dr: pathlib.Path):
        LOG.info("Storing results in %s", out_dr)

        # store PROMETHEE I & II
        for case in self.ranking:
            f_path = out_dr / f"{case}.csv"
            LOG.info("Store %s in %s", case, f_path)
            self.ranking[case].to_csv(f_path)

        f_path = out_dr / "grouped_net_flow.csv"
        LOG.info("Store grouped_net_flow in %s", f_path)
        self.net_flow_grouped.to_csv(f_path)

        # store graphics
        self.plotter(f_path=out_dr / "flows_graph.jpg")

        LOG.info("Done storing results in %s", out_dr)

    def __with_indifferences(self, float_pnt_prec=5) -> pd.Series:
        """

        :param float_pnt_prec: Round on floating point precision
        :return: index in net flow, value is list with alternative names
        """
        net_flows = self.ranking[self.net_str].round(decimals=float_pnt_prec)

        phis = defaultdict(list)
        for idx in net_flows.index:
            phis[net_flows.loc[idx]].append(idx)
        phis = [(k, sorted(phis[k])) for k in sorted(phis.keys(), reverse=True)]
        phis = pd.Series(dict(phis))

        return phis

    def __parse_alternative_file(self, f_path: pathlib.Path, dic: DefaultDict[str, list]) -> DefaultDict[str, list]:
        LOG.info("Parse file %s", f_path)
        # read file
        with open(f_path) as f:
            content = f.readlines()

        # parse file
        for line in content:
            criterion, value = line.split("=")
            dic[criterion].append(float(value))

        # add alternative name
        dic[self.alt_col_name].append(f_path.stem)

        LOG.info("File parsed")

        return dic


def test():
    b_path = Path("src") / "mcdm" / "test_files"
    LOG.info("Directory %s - Exists: %s", b_path, b_path.is_dir())
    files = [x for x in b_path.glob("alt_*.txt") if int(x.stem.split("_")[-1]) < 4]

    LOG.info("Start")
    decider = PrometheeSortingBased()
    decider.get_data(in_info=files)
    decider.execute()
    decider.printer()
    LOG.info("Done")


if __name__ == "__main__":
    init_logger(log_lvl=logging.DEBUG)

    test()
