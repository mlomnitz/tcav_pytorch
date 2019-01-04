import time
from cav import CAV
from cav import get_or_train_cav
import numpy as np
import run_params
import utils


class TCAV(object):
    """TCAV object: runs TCAV for one target and a set of concepts.
    The static methods (get_direction_dir_sign, compute_tcav_score,
    get_directional_dir) invole getting directional derivatives and
    calculating TCAV scores. These are static because they might be
    useful independently, for instance, if you are developing a new
    interpretability method using CAVs.
    """

    @staticmethod
    def get_direction_dir_sign(mymodel, act, cav, concept, class_id):
        """Get the sign of directional derivative.
        
        Args:
            mymodel (nn.Module): a model class instance
            act: activations of one bottleneck to get gradient with respect to.
            cav: an instance of cav
            concept: one concept
            class_id: index of the class of interest (target) in logit layer.
        Returns:
            sign of the directional derivative
        """
        # negative one to get direction that decreases the probability
        grad = np.reshape(mymodel.get_gradient(act, [class_id], cav.bottleneck), -1)
        dot_prod = np.dot(grad, cav.get_direction(concept))
        return dot_prod < 0

    @staticmethod
    def compute_tcav_score(mymodel, target_class, concept, cav,
                           class_acts):
        """Compute TCAV score.
        
        Args:
            mymodel: a model class instance
            target_class: one target class
            concept: one concept
            cav: an instance of cav
            class_acts: activations of the images in the target class.
            
        Returns:
            TCAV score (i.e., ratio of pictures that returns negative dot
            product wrt loss).
        """
        count = 0
        class_id = mymodel.label_to_id(target_class)
        for i in range(len(class_acts)):
            act = np.expand_dims(class_acts[i], 0)
            if TCAV.get_direction_dir_sign(mymodel, act, cav, concept,
                                           class_id):
                count += 1
        return float(count) / float(len(class_acts))
    
    def __init__(self, target, concepts, bottlenecks, activation_generator,
                 alphas, random_counterpart, cav_dir, num_random_exp,
                 random_concepts):
        """Initialze tcav class.

        Args:          
            target: one target class
            concepts: one concept
            bottlenecks: the name of a bottleneck of interest.
            activation_generator: an ActivationGeneratorInterface instance to
                return activations.
            alphas: list of hyper parameters to run
            random_counterpart: the random concept to run against the concepts
                for statistical testing.
            cav_dir: the path to store CAVs
            num_random_exp: number of random experiments to compare against.
            random_concepts: A list of names of random concepts for the random
                experiments to draw from. Optional, if not provided, the names
                will be random500_{i} for i in num_random_exp.
        """       
        self.target = target
        self.concepts = concepts
        self.bottlenecks = bottlenecks
        self.activation_generator = activation_generator
        self.cav_dir = cav_dir
        self.alphas = alphas
        self.random_counterpart = random_counterpart
        self.mymodel = activation_generator.get_model()
        self.model_to_run = self.mymodel.model_name

        # make pairs to test.
        self._process_what_to_run_expand(num_random_exp=num_random_exp,
                                         random_concepts=random_concepts)
        # parameters
        self.params = self.get_params()
        
    def run(self):
        """Run TCAV for all parameters (concept and random), write results to html.
        Args:
          num_workers: number of workers to parallelize
          run_parallel: run this parallel.
        Returns:
          results: result dictionary.
        """
        # for random exp,  a machine with cpu = 30, ram = 300G, disk = 10G and
        # pool worker 50 seems to work.
        now = time.time()
        results = []
        for param in self.params:
            results.append(self._run_single_set(param))
        return results
    
    def _run_single_set(self, param):
        """Run TCAV with provided for one set of (target, concepts).
        Args:
          param: parameters to run
        Returns:
          a dictionary of results (panda frame)
        """
        bottleneck = param.bottleneck
        concepts = param.concepts
        target_class = param.target_class
        activation_generator = param.activation_generator
        alpha = param.alpha
        mymodel = param.model
        cav_dir = param.cav_dir
        
        # Get acts
        acts = activation_generator.process_and_load_activations(
            [bottleneck], concepts + [target_class])
        # Get CAVs
        cav_hparams = CAV.default_hparams()
        cav_hparams.alpha = alpha
        cav_instance = get_or_train_cav(
            concepts, bottleneck, acts, cav_dir=cav_dir,
            cav_hparams=cav_hparams)

        # clean up
        for c in concepts:
            del acts[c]

        # Hypo testing
        a_cav_key = CAV.cav_key(concepts, bottleneck, cav_hparams.model_type,
                                cav_hparams.alpha)
        target_class_for_compute_tcav_score = target_class

        for cav_concept in concepts:
            if cav_concept is self.random_counterpart or 'random' not in cav_concept:
                i_up = self.compute_tcav_score(
                    mymodel, target_class_for_compute_tcav_score, cav_concept,
                    cav_instance, acts[target_class][cav_instance.bottleneck])
                val_directional_dirs = self.get_directional_dir(
                    mymodel, target_class_for_compute_tcav_score, cav_concept,
                    cav_instance, acts[target_class][cav_instance.bottleneck])
                result = {'cav_key' : a_cav_key, 'cav_concept' : cav_concept,
                          'target_class' : target_class, 'i_up' : i_up,
                          'val_directional_dirs_abs_mean' :
                          np.mean(np.abs(val_directional_dirs)), 
                          'val_directional_dirs_mean' :
                          np.mean(val_directional_dirs),
                          'val_directional_dirs_std' :
                          np.std(val_directional_dirs),
                          'note' : 'alpha_%s ' % (alpha),
                          'alpha' : alpha,
                          'bottleneck' : bottleneck}
        del acts
        return result
    
#     def _process_what_to_run_expand(self, num_random_exp=100, random_concepts=None):
#         """Get tuples of parameters to run TCAV with.
#         TCAV builds random concept to conduct statistical significance testing
#         againts the concept. To do this, we build many concept vectors, and many
#         random vectors. This function prepares runs by expanding parameters.
#         Args:
#           num_random_exp: number of random experiments to run to compare.
#           random_concepts: A list of names of random concepts for the random experiments
#                        to draw from. Optional, if not provided, the names will be
#                        random500_{i} for i in num_random_exp.
#         """

#         target_concept_pairs = [(self.target, self.concepts)]

#         all_concepts_concepts, pairs_to_run_concepts = (
#             utils.process_what_to_run_expand(
#                 utils.process_what_to_run_concepts(target_concept_pairs),
#                 self.random_counterpart,
#                 num_random_exp=num_random_exp,
#                 random_concepts=random_concepts))
#         all_concepts_randoms, pairs_to_run_randoms = (
#             utils.process_what_to_run_expand(
#                 utils.process_what_to_run_randoms(target_concept_pairs,
#                                                   self.random_counterpart),
#                 self.random_counterpart,
#                 num_random_exp=num_random_exp,
#                 random_concepts=random_concepts))
#         self.all_concepts = list(set(all_concepts_concepts + all_concepts_randoms))
#         self.pairs_to_test = pairs_to_run_concepts + pairs_to_run_randoms

    def get_params(self):
        """Enumerate parameters for the run function.
        Returns:
          parameters
        """
        params = []
        for bottleneck in self.bottlenecks:
            for target_in_test, concepts_in_test in self.pairs_to_test:
                for alpha in self.alphas:
                    params.append(
                        run_params.RunParams(bottleneck, concepts_in_test,
                                             target_in_test,
                                             self.activation_generator,
                                             self.cav_dir, alpha,
                                             self.mymodel))
        return params
