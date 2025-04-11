from __future__ import print_function

import argparse
import copy
from pathlib import Path
from collections import namedtuple
from time import time
from typing import List, Optional

import joblib
import nltk
import numpy as np
from joblib import delayed
from rdkit import rdBase

from guacamol.utils.chemistry import canonicalize
from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum
from moleval.utils import read_smiles
from . import cfg_util, smiles_grammar
from ...common.utils import load_config, save_config

rdBase.DisableLog('rdApp.error')
GCFG = smiles_grammar.GCFG

Molecule = namedtuple('Molecule', ['score', 'smiles', 'genes'])


def cfg_to_gene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                     smiles_grammar.GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules


def select_parent(population, tournament_size=3):
    idx = np.random.randint(len(population), size=tournament_size)
    best = population[idx[0]]
    for i in idx[1:]:
        if population[i][0] > best[0]:
            best = population[i]
    return best


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def deduplicate(population):
    unique_smiles = set()
    unique_population = []
    for item in population:
        score, smiles, gene = item
        if smiles not in unique_smiles:
            unique_population.append(item)
        unique_smiles.add(smiles)
    return unique_population


def mutate(p_gene, scoring_function):
    c_gene = mutation(p_gene)
    c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    c_score = scoring_function.score(c_smiles)
    return Molecule(c_score, c_smiles, c_gene)


class ChemGEGenerator:

    def __init__(self, smi_file, population_size, n_mutations, gene_size, generations, n_jobs=-1, random_start=False, patience=5):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.gene_size = gene_size
        self.generations = generations
        self.random_start = random_start
        self.patience = patience

    def load_smiles_from_file(self, smi_file):
        smiles = read_smiles(smi_file) # MODIFIED for gz
        return self.pool(delayed(canonicalize)(s.strip()) for s in smiles)

    def top_k(self, smiles, scoring_function, k):
        #joblist = (delayed(scoring_function.score)(s) for s in smiles)
        #scores = self.pool(joblist)
        scores = scoring_function.score(smiles, flt=True)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        #if number_molecules > self.population_size:
        #    self.population_size = number_molecules
        #    print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            init_size = self.population_size #+ self.n_mutations
            all_smiles = copy.deepcopy(self.all_smiles)
            if self.random_start:
                starting_population = list(np.random.choice(all_smiles, init_size))
            else:
                starting_population = self.top_k(all_smiles, scoring_function, init_size)

        # The smiles GA cannot deal with '%' in SMILES strings (used for two-digit ring numbers).
        starting_population = [smiles for smiles in starting_population if '%' not in smiles]

        # calculate initial genes
        initial_genes = [cfg_to_gene(cfg_util.encode(s), max_len=self.gene_size)
                         for s in starting_population]

        # score initial population
        initial_scores = scoring_function.score(starting_population, flt=True)
        population = [Molecule(*m) for m in zip(initial_scores, starting_population, initial_genes)]
        population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
        population_scores = [p.score for p in population]

        # evolution: go go go!!
        t0 = time()

        patience = 0
        generation = 0

        while not scoring_function.finished:
        #for generation in range(self.generations):

            old_scores = population_scores
            # select random genes
            all_genes = [molecule.genes for molecule in population]
            choice_indices = np.random.choice(len(all_genes), self.n_mutations, replace=True)
            genes_to_mutate = [all_genes[i] for i in choice_indices]

            # evolve genes
            #joblist = (delayed(mutate)(g, scoring_function) for g in genes_to_mutate)
            #new_population = self.pool(joblist)
            # MODIFIED to prevent scoring a molecule at a time
            c_genes = self.pool(delayed(mutation)(g) for g in genes_to_mutate)
            c_smiles = self.pool(delayed(canonicalize)(cfg_util.decode(gene_to_cfg(g))) for g in c_genes)
            c_scores = scoring_function.score(c_smiles, flt=True)
            new_population = [Molecule(*m) for m in zip(c_scores, c_smiles, c_genes)]

            # join and dedup
            population += new_population
            population = deduplicate(population)

            # survival of the fittest
            population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

            # stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()

            population_scores = [p.score for p in population]

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
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')
            generation += 1

        # finally
        return [molecule.smiles for molecule in population[:number_molecules]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', default='data/guacamol_v1_all.smiles')
    parser.add_argument('--molscore_config', help='Path to the config file for the MolScore scoring function')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--n_mutations', type=int, default=200)
    parser.add_argument('--gene_size', type=int, default=300)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    np.random.seed(args.seed)

    optimizer = ChemGEGenerator(smi_file=args.smiles_file,
                                population_size=args.population_size,
                                n_mutations=args.n_mutations,
                                gene_size=args.gene_size,
                                generations=args.generations,
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
