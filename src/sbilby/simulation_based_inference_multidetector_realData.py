import inspect
import pickle
import os
import bilby
import numpy as np
import sbi
import sbi.utils
import sbi.inference
import torch
import random
import matplotlib.pyplot as plt


from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger, check_directory_exists_and_if_not_mkdir
from bilby.core.prior.base import Constraint
from bilby.gw.detector import InterferometerList



class GenerateRealData(object):
    """
    A generic base class for data generator objects

    SimulateData instances generate data from an underlying model in a form
    suitable to pass into the SBI package

    Parameters
    ==========
    parameters: dictionary
        A dictionary of parameters for initialisation.
    call_parameter_key_list: list
        A list of keys corresponding to the ordering of the parameters to be
        passed to the call method of this class.
    """

    def __init__(self, parameters, call_parameter_key_list, simulation_number):
        self.parameters = parameters
        self.call_parameter_key_list = call_parameter_key_list
        self.simulation_numbers = torch.arange(0, simulation_number)
        logger.info(f"The maximum number of simulations you can do is {simulation_number}")
    def fix_parameter(self, key, val):
        self.parameters[key] = val
        self.call_parameter_key_list.pop(self.call_parameter_key_list.index(key))

    def get_data(parameters: dict, simulation_number):
        NotImplementedError("Method get_data() should be implemented by subclass")

    def __call__(self, new_parameter_list):
        if len(new_parameter_list) != len(self.call_parameter_key_list):
            raise ValueError(
                f"Instance of {self.__class__} called with parameter list of "
                f"length {len(new_parameter_list)}, but requires a list of "
                f"length {len(self.call_parameter_key_list)}"
            )

        for key, val in zip(self.call_parameter_key_list, new_parameter_list):
            self.parameters[key] = val
        logger.info(f"You are left with {self.simulation_numbers.numel()} pieces of training data")    
        if self.simulation_numbers.numel() == 0:  # Check if tensor is empty
            raise ValueError("No more elements to select!")

        # Choose a random index
        index = random.randint(0, self.simulation_numbers.numel() - 1)
        
        # Get the selected value
        sim_number = self.simulation_numbers[index].item()  # Convert tensor to Python int
        
        # Remove the selected value from the tensor
        self.simulation_numbers = torch.cat((self.simulation_numbers[:index], self.simulation_numbers[index+1:]))  # Reconstruct tensor

        return torch.as_tensor(self.get_data(self.parameters,sim_number))

class AdditiveSignalAndNoise_realData(GenerateRealData):
    def __init__(self, signal, noise, max_simulations):

        self.signal = signal
        self.noise = noise
        self.simulation_numbers=np.arange(0, max_simulations).tolist()
        
        # Extract the parameters and keys
        call_parameter_key_list = (
            self.signal.call_parameter_key_list + self.noise.call_parameter_key_list
        )
        joint_parameters = self.noise.parameters | self.signal.parameters
        parameters = {
            key: val
            for key, val in joint_parameters.items()
            if key in call_parameter_key_list
        }

        super(AdditiveSignalAndNoise_realData, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
            simulation_number=max_simulations
        )

    def get_data(self, parameters: dict, simulation_number):
        sparameters = {
            key: val
            for key, val in parameters.items()
            if key in self.signal.call_parameter_key_list
        }
        nparameters = {
            key: val
            for key, val in parameters.items()
            if key in self.noise.call_parameter_key_list
        }
        
        sdata = self.signal.get_data(sparameters,simulation_number)
        
        ndata = self.noise.get_data(nparameters, simulation_number)
        
        return sdata + ndata




class NLELikelihood_realData(Likelihood):
    def __init__(
        self,
        yobs,
        psd,
        generators,
        interferometers,
        bilby_priors,
        labels,
        num_simulations=1000,
        num_workers=1,
        density_estimator="maf",
        device="cpu",
        show_progress_bar=True,
        cache=True,
        cache_directory="likelihood_cache",
    ):
        """
        Neural likelihood estimated with SNLE

        Parameters
        ----------
        yobs: list of array_like
            TBD
        generators: instance of SimulateData
            An instance of the SimulateData class
        bilby_priors: bilby.core.prior.PriorDict
            The Bilby priors
        labels: list of str
            Unique strings used to cache the likelihoods
        num_simulations: int
            The number of simulations used to train the neural network
        num_workers: int
            The number of workers used to..FIX ME
        density_estimator: FIX ME
        device: FIX ME
        show_progress_bar: FIX ME
        cache: bool
            If true, write a copy of the likelihood to avoid retraining
        cache_directory: str
            The directory to store the likelihood cache
        """
        for i in range(len(InterferometerList(interferometers))):
            
            super().__init__(generators[i].parameters) #Careful on this for not using list

        self.yobs = yobs
        self.psd=psd
        self.generators = generators
        self.interferometers = InterferometerList(interferometers)
        self.bilby_priors = bilby_priors
        self.labels = labels
        self.num_simulations = num_simulations
        self.num_workers = num_workers
        self.density_estimator = density_estimator
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.cache = cache
        self.cache_directory = cache_directory
        self.fixed_parameters = [
            key for key in self.generators[0].call_parameter_key_list if bilby_priors[0][key].is_fixed
        ]   #Careful on this for not using list

        self.meta_data = dict(
            num_simulations=num_simulations,
            density_estimator=density_estimator,
            device=device
        )

    def init(self):
        # Initialise SBI elements
        
        self.sbi_generator=[]
        self.sbi_likelihood_estimator=[]
        self.sbi_potential_fn=[]
        self.sbi_prior=[]
        self.sbi_num_parameters=[]
        self.sbi_prior_returns_numpy=[]
        
        for self.ifo in range(len(self.interferometers)):
            self.init_prior()
            self.init_simulator()
            self.init_training()
            self.init_potential_fn()

    def update_parameters_from_prior(self):
        for key, val in self.bilby_prior.items():
            if val.is_fixed:
                self.parameters[key] = val.peak

    def init_prior(self):
        logger.info("Initialise the SBI prior")
        prior_min, prior_max = [], []
        for key in self.generators[self.ifo].call_parameter_key_list:
            if self.bilby_priors[self.ifo][key].is_fixed is False:
                prior_min.append(self.bilby_priors[self.ifo][key].minimum)
                prior_max.append(self.bilby_priors[self.ifo][key].maximum)

        torch_prior = sbi.utils.torchutils.BoxUniform(
            low=torch.as_tensor(prior_min),
            high=torch.as_tensor(prior_max))
        (
            sbi_prior,
            sbi_num_parameters,
            sbi_prior_returns_numpy,
        ) = sbi.utils.user_input_checks.process_prior(torch_prior)
        self.sbi_prior.append(sbi_prior)
        self.sbi_num_parameters.append(sbi_num_parameters)
        self.sbi_prior_returns_numpy.append(sbi_prior_returns_numpy)
        
    def init_simulator(self):
        logger.info("Initialise the SBI simulator")
        self.sbi_generator.append( sbi.utils.user_input_checks.process_simulator(
            self.generators[self.ifo], self.sbi_prior[self.ifo], self.sbi_prior_returns_numpy[self.ifo]
        ))
        sbi.utils.user_input_checks.check_sbi_inputs(
            self.sbi_generator[self.ifo], self.sbi_prior[self.ifo]
        )

    def init_training(self):
        if os.path.exists(self.cache_filename):
            self.load_trained_likelihood()
        else:
            self.train_likelihood()

    def load_trained_likelihood(self):
        logger.info(f"Loading in cached NLE {self.cache_filename}")
        with open(self.cache_filename, "rb") as file:
            self.sbi_likelihood_estimator.append(pickle.load(file))

    def train_likelihood(self):
        logger.info("Initialise training")
        inference = sbi.inference.SNLE(
            prior=self.sbi_prior[self.ifo],
            density_estimator=self.density_estimator,
            device=self.device,
            logging_level='WARNING',
            summary_writer=None,
            show_progress_bars=self.show_progress_bar,
        )
        simulated_params, simulated_yobs = sbi.inference.simulate_for_sbi(
            self.sbi_generator[self.ifo],
            proposal=self.sbi_prior[self.ifo],
            num_simulations=self.num_simulations,
            num_workers=self.num_workers,
            show_progress_bar=self.show_progress_bar,
        )
        logger.info(f"Number of produced simulations {len(simulated_yobs)}")
        inf_and_sims = inference.append_simulations(simulated_params, simulated_yobs)
        self.sbi_likelihood_estimator.append(inf_and_sims.train(show_train_summary=True))
        # Plot the training and validation loss as a function of the training epochs without using tensorboard.
        # plt.plot(inference._summary['validation_log_probs'],label = 'validation',c = 'k')
        # plt.plot(inference._summary['training_log_probs'],label = 'training',c= 'tab:green')
        # plt.ylabel('loss')
        # plt.xlabel('Epochs')
        # plt.legend(loc= 'best')
        #plt.savefig("/home/mattia.emma/public_html/NLE/sbilbi/glitchy_invetigations/Study_bias/Training_"+str(self.num_simulations)+"_"+str(self.ifo)+".png",dpi=100)
        if self.cache:
            logger.info(f"Writing the cached NLE to {self.cache_filename}")
            check_directory_exists_and_if_not_mkdir(self.cache_directory)
            with open(self.cache_filename, "wb") as file:
                pickle.dump(self.sbi_likelihood_estimator[self.ifo], file)

    @property
    def cache_filename(self):
        return f"{self.cache_directory}/{self.labels[self.ifo]}.pkl"

    def init_potential_fn(self):
        sbi_potential_fn, _ = sbi.inference.likelihood_estimator_based_potential(
            self.sbi_likelihood_estimator[self.ifo], self.sbi_prior[self.ifo], self.yobs[self.ifo]
        )
        self.sbi_potential_fn.append(sbi_potential_fn)

    def log_likelihood(self):
        logl=0
        for ifo in range(len(self.interferometers)):
            parameters = [
                np.float32(self.parameters[key])
                for key in self.generators[ifo].call_parameter_key_list
                if key not in self.fixed_parameters
            ]
            parameter_tensor = torch.as_tensor(parameters)
        
            logl += self.sbi_potential_fn[ifo](parameter_tensor)
        return float(logl)


class NLEResidualLikelihood_realData(NLELikelihood_realData):
    def __init__(
        self,
        yobs,
        psd,
        generators,
        interferometers,
        bilby_priors,
        labels,
        num_simulations=1000,
        num_workers=1,
        cache=True,
        cache_directory="likelihood_cache",
    ):
        """
        Neural likelihood estimated with SNLE

        Parameters
        ----------
        yobs: array_like
            TBD
        FIXME generate: instance of SimulateData
            An instance of the SimulateData class
        bilby_prior: bilby.core.prior.PriorDict
            The Bilby prior
        label: str
            A unique string used to cache the likelihood
        num_simulations: int
            The number of simulations used to train the neural network
        num_workers: int
            The number of workers used to..FIX ME
        cache: bool
            If true, write a copy of the likelihood to avoid retraining
        cache_directory: str
            The directory to store the likelihood cache
        """
        super().__init__(
            yobs=yobs,psd=psd, generators=list(map(lambda item: item.noise, generators)), interferometers=interferometers, bilby_priors=bilby_priors, labels=labels,
            num_simulations=num_simulations, num_workers=num_workers, cache=cache,
            cache_directory=cache_directory
        )
        
        self.noise_generators=[]
        self.signal_generators=[]
        self.sbi_generator=[]
        self.sbi_likelihood_estimator=[]
        self.sbi_prior=[]
        self.sbi_num_parameters=[]
        self.sbi_prior_returns_numpy=[]
        for generator in generators:
            self.noise_generators.append(generator.noise)
            self.signal_generators.append(generator.signal)
        self.interferometers = InterferometerList(interferometers)
        self.psd=psd
    def init(self):
        # Initialise SBI elements
        
        for self.ifo in range(len(self.interferometers)):
            self.init_prior()
            self.init_simulator()
            self.init_training()

    def init_potential_fn(self):
        sbi_potential_fn, _ = sbi.inference.likelihood_estimator_based_potential(
            self.sbi_likelihood_estimator[self.ifo], self.sbi_prior[self.ifo], self.yobs_residual
        )
        self.sbi_potential_fn.append(sbi_potential_fn)

    def log_likelihood(self):
        logl=0
        for self.ifo in range(len(self.interferometers)):
            parameters = [
                np.float32(self.parameters[key])
                for key in self.generators[self.ifo].call_parameter_key_list
                if key not in self.fixed_parameters
            ]
        
            self.sbi_potential_fn=[]
            signal_prediction = self.signal_generators[self.ifo].get_data(self.parameters,psd=self.psd[self.ifo])
           
            self.yobs_residual = self.yobs[self.ifo] - signal_prediction
            
            self.init_potential_fn()
            parameter_tensor = torch.as_tensor(parameters)
            logl += self.sbi_potential_fn[0](parameter_tensor)
        return float(logl)



class GenerateRealData_removal(object):
    """
    A generic base class for data generator objects

    SimulateData instances generate data from an underlying model in a form
    suitable to pass into the SBI package

    Parameters
    ==========
    parameters: dictionary
        A dictionary of parameters for initialisation.
    call_parameter_key_list: list
        A list of keys corresponding to the ordering of the parameters to be
        passed to the call method of this class.
    """

    def __init__(self, parameters, call_parameter_key_list, simulation_number, removal_mode=True):
        self.parameters = parameters
        self.call_parameter_key_list = call_parameter_key_list
        self.simulation_numbers = torch.arange(0, simulation_number)
        self.simulation_counter = {i: 0 for i in self.simulation_numbers.tolist()}
        logger.info(f"The maximum number of simulations you can do is {simulation_number}")
    def fix_parameter(self, key, val):
        self.parameters[key] = val
        self.call_parameter_key_list.pop(self.call_parameter_key_list.index(key))

    def get_data(parameters: dict, simulation_number):
        NotImplementedError("Method get_data() should be implemented by subclass")

    def __call__(self, new_parameter_list):
        if len(new_parameter_list) != len(self.call_parameter_key_list):
            raise ValueError(
                f"Instance of {self.__class__} called with parameter list of "
                f"length {len(new_parameter_list)}, but requires a list of "
                f"length {len(self.call_parameter_key_list)}"
            )
    
        for key, val in zip(self.call_parameter_key_list, new_parameter_list):
            self.parameters[key] = val
        
        if self.simulation_numbers.numel() == 0:
            raise ValueError("No more elements to select!")
    
        index = random.randint(0, self.simulation_numbers.numel() - 1)
        sim_number = self.simulation_numbers[index].item()
    
        if self.removal_mode:
            logger.info(f"You are left with {self.simulation_numbers.numel()} pieces of training data")    
            self.simulation_numbers = torch.cat((
                self.simulation_numbers[:index],
                self.simulation_numbers[index+1:]
            ))
        else:
            self.simulation_counter[sim_number] += 1
            logger.info(
                f"Simulation number {sim_number} selected. "
                f"Now used {self.simulation_counter[sim_number]} times."
            )
    
        return torch.as_tensor(self.get_data(self.parameters, sim_number))
