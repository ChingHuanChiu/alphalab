'''
TODO: 1. add method to get model from mlflow artifacts
      2. hyperparameters tune 
      3. write DataLoader Class ,which is a generator
      4. add display such as cnf_matrix.png
'''

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple, List
from pathlib import Path


import sklearn
import tensorflow as tf
import mlflow

from mlflow.tracking import MlflowClient



class ITrainer(metaclass=ABCMeta):

    def __init__(self, experiment_name:str, reuse_exp_id: int) -> None:

        self._model = None
        DATA_SPLIT_DICT = self.train_test_data()
        self.X_train, self.y_train, self.X_test, self.y_test = \
            DATA_SPLIT_DICT['X_train'], DATA_SPLIT_DICT['y_train'],\
            DATA_SPLIT_DICT['X_test'], DATA_SPLIT_DICT['y_test'] 

        self.client =  MlflowClient()

        if not Path('mlruns').exists():
            self.experiment_id = 1

        elif experiment_name:
            self.experiment_id = self.client.create_experiment(experiment_name)
            # self.client.set_experiment_tag(self.experiment_id)
      
        if reuse_exp_id:
            self.experiment_id = reuse_exp_id
        

    @abstractmethod
    def train_test_data(self) -> Dict[str, Any]:
        """Make the train and test data for training and evaulate data.
    
        """
        raise NotImplementedError('you need to make your data ')


    @abstractmethod
    def run(self, **kwargs):
        """Train model and record by MLFlow"""
        
        raise NotImplementedError('Not Implemented')

    

    @abstractmethod
    def metrics(self) -> Dict[str, float]:
        raise NotImplementedError('Not Implemented')


    def hyper_space(self) -> Dict:
        pass

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, m):
        self._model = m
 

 

class SklearnTrainer(ITrainer):

    def __init__(self, experiment_name: str = None, reuse_exp_id: int = None) -> None:

        super().__init__(experiment_name, reuse_exp_id)
        self._model = None



    @abstractmethod
    def train_test_data(self) -> Dict[str, Any]:
        
        raise NotImplementedError('you need to make your data ')

    def run(self, **kwargs):
        mlflow.sklearn.autolog()

        with mlflow.start_run(experiment_id=self.experiment_id):
            self.train(**kwargs)


            for name, score in self.metrics().items():
                
                mlflow.log_metric(name, score)

    

    def train(self, **kwargs):

        
        if 'generator' in str(type(iter(self.X_train))):
            
            for step, (X, y) in enumerate(zip(self.X_train, self.y_train)):
                self.model.fit_partial(X, y)
            
        else:

            self.model.fit(self.X_train, self.y_train)

    @abstractmethod
    def metrics(self) -> Dict[str, float]:
        raise NotImplementedError('Not Implemented')
    

    def hyper_space(self):
        pass




class TFTrainer(ITrainer):

    def __init__(self, n_classes: int, experiment_name: str = None, reuse_exp_id: int = None) -> None:

        super().__init__(experiment_name, reuse_exp_id)

        self.n_classes = n_classes
        self._loss_fn: tf.keras.losses = None
        self._optimizer: tf.keras.optimizers = None
        self._callbacks: List = None
        self._batch_size: int = None
        self._epochs: int = None


    @abstractmethod
    def train_test_data(self) -> Dict[str, Any]:
     
        raise NotImplementedError('you need to make your data ')


    @abstractmethod
    def loss_fn(self) -> tf.keras.losses:
        ...

    @abstractmethod
    def optimizer(self) -> tf.keras.optimizers:
        ...

  
    @abstractmethod
    def callbacks(self) -> List:
        ...

    def run(self, **kwargs):
       

        mlflow.tensorflow.autolog()
        
        with mlflow.start_run(experiment_id=self.experiment_id):
            self.train(**kwargs)

            y_pred = self.model.predict(self.X_test)
            for name, score in self.metrics().items():
                score = score(self.y_test, y_pred).numpy()
                mlflow.log_metric(name, score)

    
    def train(self, **kwargs):

        if 'generator' in str(type(iter(self.X_train))):

            shape = (iter(self.X_train).__next__().shape[1], iter(self.X_train).__next__().shape[-1])
            
            if kwargs['batch_size'] is not None:
                print(f'Warning: your training data is generater , batch_size must be None, {self.batch_size} instead, \
                    it has been set None automatically!') 
            kwargs['batch_size'] = None

        else:
            shape = (self.X_train.shape[1], self.X_train.shape[-1])


        i = tf.keras.Input(shape=shape)
        out = self.model(i)
        self.model = tf.keras.Model(i, out)

        self.model.compile(loss=self.loss_fn(),
                           optimizer=self.optimizer(),
                           metrics=list(self.metrics().values()))

        # self.model.summary()

        self.model.fit(self.X_train, self.y_train, batch_size=kwargs['batch_size'], epochs=kwargs['epochs'], verbose=1, 
                    validation_split=0.1, callbacks=self.callbacks())

        # model.save(model_path, save_format='tf')

    def metrics(self) -> Dict:

        raise NotImplementedError('Not Implemented')

    def hyper_space(self):
        pass

 





