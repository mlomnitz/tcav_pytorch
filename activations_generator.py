import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import CelebADataset


class ActivationsGenerator():
    def __init__(self, model, source_dir, source_df, acts_dir,
                 bottleneck_names, concepts=None, transform=None,
                 max_examples=10):

        assert os.path.isfile(source_df), 'Concepts dataframe is missing!'
        self.concepts_df = pd.read_csv(source_df)
        # First column is path
        if concepts is None:
            self.concepts = list(self.concepts_df.columns)[:1]
        else:
            self.concepts = concepts
        self.source_dir = source_dir
        self.acts_dir = acts_dir
        self.model = model
        self.bottleneck_names = bottleneck_names
        self.max_examples = max_examples
        self.transform = transform

    def generate_acts(self, batch_size=16, num_workers=8, verbose=False):
        """ Generates the concept activations from the loaded dataframe and
        saves them to the acts_dir
        """
        # check if acts_dir exists
        if not os.path.exists(self.acts_dir):
            os.makedirs(self.acts_dir)
        self.model.eval()
        for concept in self.concepts:
            sub_frame = self.concepts_df[self.concepts_df[concept] == 1]
            sub_frame.reset_index(drop=True, inplace=True)
            concept_dataset = CelebADataset(df=sub_frame,
                                            source_dir=self.source_dir,
                                            transform=self.transform)
            concept_loader = DataLoader(concept_dataset, batch_size=batch_size,
                                        num_workers=num_workers)
            acts = {}
            for idx, batch in enumerate(concept_loader):
                out_ = self.model(batch)
                for bottleneck in self.bottleneck_names:
                    acts[bottleneck].append(acts[bottleneck], self.model.
                                            bottlenecks_tensors[bottleneck].
                                            numpy())
                if verbose:
                    print("[{}/{}]".format(idx, len(concept_loader)))
                    
            for bottleneck in self.bottleneck_names:
                acts_path = os.join(self.acts_dir, 'acts_{}_{}'.format(concept, bottleneck))
                np.save(acts_path, acts[bottleneck])
                
                
                    
                    
