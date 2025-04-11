import argparse
import os
from pathlib import Path

from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum

from .smiles_rnn_directed_generator import SmilesRnnDirectedGenerator
from ...common.utils import load_config, save_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Goal-directed generation benchmark for SMILES RNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default=None, help='Full path to the pre-trained SMILES RNN model')
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--number_repetitions', default=1, type=int, help='Number of re-training runs to average')
    parser.add_argument('--keep_top', default=512, type=int, help='Molecules kept each step')
    parser.add_argument('--n_epochs', default=20, type=int, help='Epochs to sample')
    parser.add_argument('--mols_to_sample', default=1024, type=int, help='Molecules sampled at each step')
    parser.add_argument('--optimize_batch_size', default=256, type=int, help='Batch size for the optimization')
    parser.add_argument('--optimize_n_epochs', default=2, type=int, help='Number of epochs for the optimization')
    parser.add_argument('--benchmark_num_samples', default=4096, type=int,
                        help='Number of molecules to generate from final model for the benchmark')
    parser.add_argument('--benchmark_trajectory', action='store_true',
                        help='Take molecules generated during re-training into account for the benchmark')
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()

    if args.model_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.model_path = os.path.join(dir_path, 'pretrained_model', 'model_final_0.473.pt')

    optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=args.model_path,
                                           n_epochs=args.n_epochs,
                                           mols_to_sample=args.mols_to_sample,
                                           keep_top=args.keep_top,
                                           optimize_n_epochs=args.optimize_n_epochs,
                                           max_len=args.max_len,
                                           optimize_batch_size=args.optimize_batch_size,
                                           number_final_samples=args.benchmark_num_samples,
                                           random_start=args.random_start,
                                           smi_file=args.smiles_file,
                                           n_jobs=args.n_jobs)
    
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
        save_config(vars(args), Path(MSB.output_dir) / "args.yaml")
        save_config(cfg, Path(MSB.output_dir) / "molscore_args.yaml")
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
