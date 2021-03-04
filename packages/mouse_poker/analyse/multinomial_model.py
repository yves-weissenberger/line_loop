import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from sklearn.model_selection import train_test_split
tfd = tfp.distributions


class multinomial_model(object):

    def __init__(self,X,y,n_classes=None,seed=0,CV_ratio=[.7,.3],eta=0.001,batch_size=50,prior_std=10):
        """ stuff lets call eta the learning rate"""

        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.n_dataPoints,self.dim = X.shape
        if n_classes is None:
            self.n_classes = len(np.unique(y))  #this is the number 
        else:
            self.n_classes = n_classes

        self.model = _BayesianMultinomialRegression(self.dim,self.n_classes,prior_std=prior_std)
        self._optimizer = tf.keras.optimizers.Adam(lr=eta,amsgrad=True,)

        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            X, y, test_size=0.4, random_state=self.seed)

        self._data_train = tf.data.Dataset.from_tensor_slices(
                    (self.X_train, self.y_train)).shuffle(10000).batch(batch_size)

    def fit(self,n_epochs=10,verbose=True,get_test_cost=True):
        # Fit the model

        test_cost = []
        for epoch in range(n_epochs):
            
            tmp = np.sum(self.model(self.X_test,sampling=False).log_prob(self.y_test).numpy())
            test_cost.append(tmp)

            if verbose:
                print('logP {}'.format(tmp))


            # Update weights each batch
            for x_data, y_data in self._data_train:
                self._train_step(self.model,x_data, y_data, self._optimizer)

        #if get_test_cost:
        #    return test_cost
        #else:
        #    return None
        return test_cost

    def predict(self,x):
        return self.model(x).logits.numpy()

    def test_set_poke_acc(self):
        """ This just returns the fraction of pokes that are correctly predicted.
            Not necessarily the most amazin"""
        tmp = []
        for _ in range(20):
            tmp.append(np.mean((np.argmax(self.predict(self.X_test),axis=1)==np.array([int(np.where(i)[0]) for i in self.y_test]))))
        return tmp

    @tf.function
    def _train_step(self,model,x_data, y_data,optimizer):
        with tf.GradientTape() as tape:
            log_prob = tf.reduce_mean(model(x_data).log_prob(y_data))
            kl_loss = model.losses/self.n_dataPoints
            elbo_loss = kl_loss - log_prob
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))





class _HBMR(tf.keras.Model):
    """ Hierarchical Bayesian Multinomial Regression model
        
    """

    def __init__(self, d,n_classes, name=None,prior_std=10):
        super(_BayesianMultinomialRegression, self).__init__(name=name)
        self.w_loc = tf.Variable(tf.random.normal([d, n_classes]), name='w_loc')
        self.w_std = tf.Variable(tf.random.normal([d, n_classes]), name='w_std')
        self.b_loc = tf.Variable(tf.random.normal([n_classes]), name='b_loc')
        self.b_std = tf.Variable(tf.random.normal([n_classes]), name='b_std')
        self.prior_std = prior_std

        
    
    @property
    def weight(self):
        """Variational posterior for the weight"""
        return tfd.Normal(self.w_loc, tf.exp(self.w_std))
    
    
    @property
    def subject_weight(self):
        return tfd.Normal(self.weight().sample())

    @property
    def bias(self):
        """Variational posterior for the bias"""
        return tfd.Normal(self.b_loc, tf.exp(self.b_std))



    
    def call(self, x, sampling=True):
        """Predict p(y|x)"""
        sample = lambda f: f.sample() if sampling else f.mean()
        logits = x @ sample(self.weight) + sample(self.bias)
        return tfd.Multinomial(1,logits=logits)#probs=tf.exp(logits)/tf.reduce_sum(tf.exp(logits),axis=0))#logits=logits)
    

    @property
    def losses(self):
        """Sum of KL divergences between posteriors and priors"""
        prior = tfd.Normal(0, self.prior_std)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))


class _BayesianMultinomialRegression(tf.keras.Model):
    """ Bayesian multinomial regression model. For now, is very simple
        However, provides template for creating a more useful hierarchical
        or learning based model, or even a structured model where you use
        can smartly distribute priors across animals.
     """

    def __init__(self, d,n_classes, name=None,prior_std=10):
        super(_BayesianMultinomialRegression, self).__init__(name=name)
        self.w_loc = tf.Variable(tf.random.normal([d, n_classes]), name='w_loc')
        self.w_std = tf.Variable(tf.random.normal([d, n_classes]), name='w_std')
        self.b_loc = tf.Variable(tf.random.normal([n_classes]), name='b_loc')
        self.b_std = tf.Variable(tf.random.normal([n_classes]), name='b_std')
        self.prior_std = prior_std

        
    
    @property
    def weight(self):
        """Variational posterior for the weight"""
        return tfd.Normal(self.w_loc, tf.exp(self.w_std))
    
    
    @property
    def bias(self):
        """Variational posterior for the bias"""
        return tfd.Normal(self.b_loc, tf.exp(self.b_std))



    
    def call(self, x, sampling=True):
        """Predict p(y|x)"""
        sample = lambda f: f.sample() if sampling else f.mean()
        logits = x @ sample(self.weight) + sample(self.bias)
        return tfd.Multinomial(1,logits=logits)#probs=tf.exp(logits)/tf.reduce_sum(tf.exp(logits),axis=0))#logits=logits)
    

    @property
    def losses(self):
        """Sum of KL divergences between posteriors and priors"""
        prior = tfd.Normal(0, self.prior_std)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))