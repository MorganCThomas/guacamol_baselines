from typing import List, Optional


class BestFromChemblOptimizer:
    """
    Goal-directed molecule generator that will simply look for the most adequate molecules present in a file.
    """

    def __init__(self, smiles: list) -> None:
        # get a list of all the smiles
        self.smiles = smiles

    def top_k(self, smiles, scoring_function, k):
        scores = scoring_function.score(smiles)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        """
        Will iterate through the reference set of SMILES strings and select the best molecules.
        """
        return self.top_k(self.smiles, scoring_function, number_molecules)
