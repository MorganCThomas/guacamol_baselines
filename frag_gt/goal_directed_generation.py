from __future__ import print_function

import argparse
import ast
import random
from typing import Optional, List
from pathlib import Path

import numpy as np

from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum

from .frag_gt.frag_gt import FragGTGenerator
from ...common.utils import load_config, save_config


class FragGTGoalDirectedGenerator(FragGTGenerator):
    """ wrapper class to keep FragGT and GuacaMol independent """

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        return self.optimize(scoring_function=scoring_function,  # type: ignore
                             number_molecules=number_molecules,
                             starting_population=starting_population,
                             fixed_substructure_smarts=None,
                             job_name=None)

    # def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
    #                                  starting_population: Optional[List[str]] = None, job_name: Optional[str] = None,
    #                                  ) -> List[str]:
    #     """
    #     uncomment this version of fn to write intermediate results
    #     also switch guacamol base library to version on branch: https://github.com/BenevolentAI/guacamol/pull/21
    #     """
    #     return self.optimize(scoring_function=scoring_function,  # type: ignore
    #                          number_molecules=number_molecules,
    #                          starting_population=starting_population,
    #                          fixed_substructure_smarts=None,
    #                          job_name=job_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles", help="smiles file for initial population")
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument("--fragstore_path", type=str, default=str(Path(__file__).parent / "frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl"))
    parser.add_argument("--allow_unspecified_stereocenters", type=bool, default=True,
                        help="if false, unspecified stereocenters will be enumerated to specific stereoisomers")
    parser.add_argument("--scorer", type=str, default="counts", help="random|counts|ecfp4|afps")
    parser.add_argument("--operators", type=ast.literal_eval, default=None,
                        help="List of tuples of (operator, prob of applying) where probabilities must add to 1")
    parser.add_argument("--population_size", type=int, default=500)
    parser.add_argument("--n_mutations", type=int, default=500)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--mapelites", type=str, default=None, help="keep elites in discretized space for diversity: species|mwlogp")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--random_start", action="store_true", help="sample initial population instead of scoring and taking top k")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("-v", action="store_true", help="verbose")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    optimizer = FragGTGoalDirectedGenerator(smi_file=args.smiles_file,
                                            fragmentation_scheme="brics",
                                            fragstore_path=args.fragstore_path,
                                            allow_unspecified_stereo=args.allow_unspecified_stereocenters,
                                            scorer=args.scorer,
                                            operators=args.operators,
                                            population_size=args.population_size,
                                            n_mutations=args.n_mutations,
                                            generations=args.generations,
                                            map_elites=args.mapelites,
                                            random_start=args.random_start,
                                            patience=args.patience,
                                            n_jobs=args.n_jobs,
                                            # intermediate_results_dir=intermediate_results_dir,
                                            intermediate_results_dir=None,
                                            verbose=True,
    )
    
    # ---- Run using MolScore ----
    cfg = load_config(args.molscore_config)
    # Single mode
    if cfg.molscore_mode == "single":
        task = MolScore(
            model_name=cfg.model_name,
            task_config=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            add_run_dir=True,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(task.save_dir) / "args.yaml")
        save_config(cfg, Path(task.save_dir) / "molscore_args.yaml")
        with task as scoring_function:
            optimizer.generate_optimized_molecules(
                scoring_function = scoring_function,
                number_molecules = cfg.total_smiles,
            )
    # Benchmark mode
    if cfg.molscore_mode == "benchmark":
        MSB = MolScoreBenchmark(
            model_name=cfg.model_name,
            benchmark=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            add_benchmark_dir=True,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(MSB.save_dir) / "args.yaml")
        save_config(cfg, Path(MSB.save_dir) / "molscore_args.yaml")
        with MSB as benchmark:
            for task in benchmark:
                with task as scoring_function:
                    optimizer.generate_optimized_molecules(
                        scoring_function = scoring_function,
                        number_molecules = cfg.total_smiles,
                    )
    # Curriculum mode
    if cfg.molscore_mode == "curriculum":
        task = MolScoreCurriculum(
            model_name=cfg.model_name,
            benchmark=cfg.molscore_task,
            budget=cfg.total_smiles,
            output_dir=cfg.output_dir,
            **cfg.get("molscore_kwargs", {}),
        )
        # Save configs
        save_config(vars(args), Path(task.save_dir) / "args.yaml")
        save_config(cfg, Path(task.save_dir) / "molscore_args.yaml")
        with task as scoring_function:
            optimizer.generate_optimized_molecules(
                scoring_function = scoring_function,
                number_molecules = cfg.total_smiles,
            )


if __name__ == "__main__":
    main()
