import argparse
from pathlib import Path

from moleval.utils import read_smiles
from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum
from .optimizer import BestFromChemblOptimizer
from ...common.utils import load_config, save_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Goal-directed benchmark for best molecules from SMILES file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to run')

    args = parser.parse_args()
    smiles = read_smiles(args.smiles_file)
    optimizer = BestFromChemblOptimizer(smiles=smiles)
    
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
        with task as scorer:
            optimizer.generate_optimized_molecules(
                scoring_function = scorer,
                number_molecules = cfg.total_smiles,
                starting_population = read_smiles(scorer.starting_population) if scorer.starting_population else None
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
        save_config(vars(args), Path(MSB.output_dir) / "args.yaml")
        save_config(cfg, Path(MSB.output_dir) / "molscore_args.yaml")
        with MSB as benchmark:
            for task in benchmark:
                with task as scorer:
                    optimizer.generate_optimized_molecules(
                        scoring_function = scorer,
                        number_molecules = cfg.total_smiles,
                        starting_population = read_smiles(scorer.starting_population) if scorer.starting_population else None
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
        with task as scorer:
            optimizer.generate_optimized_molecules(
                scoring_function = scorer,
                number_molecules = cfg.total_smiles,
                starting_population = read_smiles(scorer.starting_population) if scorer.starting_population else None
            )
