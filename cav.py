from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


class CAV(object):
    
    @staticmethod
    def create_training_set(concepts, bottleneck, activations):
        return
    
    def __init__(self, concepts, bottleneck, hparams, save_path=None):
        self.concepts = concepts
        self.bottleneck = bottleneck
        self.hparams = hparams
        self.save_path = save_path

    def train(self, activations):
        x, labels = CAV.create_training_set(self.concepts, self.bottleneck,
                                            activations)
        
        if self.hparama.model_type == 'linear':
            lm = linear_model.SGDClassifier(alpha=self.hparams.alpha)
        elif self.hparams.model_type == 'logistic':
            lm = linear_model.LogisticRegression()
        else:
            raise ValueError('Invalid model type{}'.format(
                self.hparams.model_type))
        
        self.accuracies = self.train_lm(lm, x, labels)
        if len(lm.coef_) == 1:
            """
            If there were only two labels, the concept is assigned to label
            0 by
            default. So we flip the coef_ to reflect this.
            """
            self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
        else:
            self.cavs = [c for c in lm.coef_]
        self.save_cavs()
        
    def train_lm(self, lm, x, y):
        x_train, x_test, y_train, y_test, = train_test_split(x, y,
                                                             test_size=0.2, stratify=y)
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_test)
        #get accuracy for each class
        num_classes = max(y) +1
        acc = {}
        num_corrrect = 0
        for class_id in range(num_classes):
          # get indices of all test data that has this class.
            idx = (y_test == class_id)
            acc[labels2text[class_id]] = metrics.accuracy_score(
                y_pred[idx], y_test[idx])
            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[labels2text[class_id]])
        acc['overall'] = float(num_correct) / float(len(y_test))
        return acc 
    
    def _save_cavs(self):
        """Save a dictionary of this CAV to a pickle."""
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.bottleneck,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }
        if self.save_path is not None:
            with tf.gfile.Open(self.save_path, 'w') as pkl_file:
                pickle.dump(save_dict, pkl_file)
        else:
            tf.logging.info('save_path is None. Not saving anything')
