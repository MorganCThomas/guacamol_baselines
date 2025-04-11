from __future__ import print_function

import argparse
from pathlib import Path
import heapq
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np
from guacamol.utils.chemistry import canonicalize
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum
from moleval.utils import read_smiles

from . import crossover as co, mutate as mu
from ...common.utils import load_config, save_config


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population


class GB_GA_Generator:

    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, n_jobs=-1, random_start=False, patience=5):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience

    def load_smiles_from_file(self, smi_file):
        smiles = read_smiles(smi_file) # MODIFIED to read gzip
        return self.pool(delayed(canonicalize)(s.strip()) for s in smiles)

    def top_k(self, smiles, scoring_function, k):
        scores = scoring_function.score(smiles, score_only=True, flt=True)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        # MODIFIED
        #if number_molecules > self.population_size:
        #    self.population_size = number_molecules
        #    print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = list(np.random.choice(self.all_smiles, self.population_size))
            else:
                starting_population = self.top_k(self.all_smiles, scoring_function, self.population_size)

        # select initial population
        population_scores = scoring_function.score(starting_population, flt=True) # MODIFIED
        _ = sorted(zip(starting_population, population_scores), key=lambda x: x[1], reverse=True)[:self.population_size]
        population_smiles, population_scores = zip(*_)
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        #population_smiles = heapq.nlargest(self.population_size, starting_population, key=scoring_function.score)
        #population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        #population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)

        # evolution: go go go!!
        t0 = time()

        patience = 0
        generation = 0
        while not scoring_function.finished: # MODIFIED

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = self.pool(delayed(reproduce)(mating_pool, self.mutation_rate) for _ in range(self.population_size))

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            #population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
            population_scores = scoring_function.score([Chem.MolToSmiles(m) for m in population_mol], flt=True) # MODIFIED
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')
            generation += 1

        # finally
        return [Chem.MolToSmiles(m) for m in population_mol][:number_molecules]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--offspring_size', type=int, default=200)
    parser.add_argument('--mutation_rate', type=float, default=0.01)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    optimizer = GB_GA_Generator(smi_file=args.smiles_file,
                                population_size=args.population_size,
                                offspring_size=args.offspring_size,
                                generations=args.generations,
                                mutation_rate=args.mutation_rate,
                                n_jobs=args.n_jobs,
                                random_start=args.random_start,
                                patience=args.patience)
    
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


if __name__ == "__main__":
    main()
