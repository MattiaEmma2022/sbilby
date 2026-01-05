import inspect
import pickle
import os
import bilby
import numpy as np
import sbi
import sbi.utils
import sbi.inference
import torch
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from bilby.core.likelihood import Likelihood  #removed .base from files for new bilby version
from bilby.core.utils import logger, check_directory_exists_and_if_not_mkdir
from bilby.core.prior.base import Constraint
from .simulation_based_inference import GenerateData
from .simulation_based_inference_multidetector_realData import GenerateRealData
from  bilby.gw.detector.networks import InterferometerList

class GenerateWhitenedIFONoise_fromGWF(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, noise_prior, use_mask, times):
        call_parameter_key_list = noise_prior.non_fixed_keys
        parameters = noise_prior.sample()

        super(GenerateWhitenedIFONoise_fromGWF, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.old_ifo=ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma_"+self.old_ifo.name.lower()]
        self.ifon = InterferometerList([self.old_ifo.name])
        self.ifon.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.old_ifo.sampling_frequency,
            duration=self.old_ifo.duration,
            start_time=self.old_ifo.start_time,
        )
        self.ifon.maximum_frequency=self.old_ifo.sampling_frequency/2
        self.ifo=self.ifon[0]
        noise=self.ifo.strain_data.time_domain_strain
        
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(noise), alpha=alpha)
        window_factor = np.mean(window ** 2)
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        
        hf = np.fft.rfft(noise*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        hf_whitened=hf/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        ht_whitened=(np.fft.irfft(hf_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        whitened_strain = ht_whitened * np.array(sigma)
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain


class GenerateWhitenedSignal_fromGWF(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior, use_mask, times):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal_fromGWF, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        frequencies=None
        if frequencies is None:
            frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
            mask = self.ifo.frequency_mask
        
        time_data=self.ifo.strain_data.time_domain_strain
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(time_data), alpha=alpha)
        window_factor = np.mean(window ** 2)
        
        frequency_data = np.fft.rfft(time_data*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        frequency_data_whitened=frequency_data/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        data_noise_whitened=(np.fft.irfft(frequency_data_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask#*window_factor
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= frequency_data+ signal_ifo
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde-data_noise_whitened
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened






class GenerateWhitenedSignal_fromPSD(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior, use_mask, times):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal_fromPSD, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        frequency_signal=self.ifo.strain_data.frequency_domain_strain
        time_whitened=self.ifo.whitened_time_domain_strain
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        frequencies=None
        if frequencies is None:
            frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
            mask = self.ifo.frequency_mask
        else:
            mask = np.ones(len(frequencies), dtype=bool)
        
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response =self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= frequency_signal+ signal_ifo
        frequency_window_factor = (
                np.sum(self.ifo.frequency_mask)
                / len(self.ifo.frequency_mask)
            )   
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (self.ifo.amplitude_spectral_density_array * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde-time_whitened
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened


class GenerateWhitenedIFONoise_fromPSD(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, use_mask, times):
        super(GenerateWhitenedIFONoise_fromPSD, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.ifo = ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma"]
        self.ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.ifo.sampling_frequency,
            duration=self.ifo.duration,
            start_time=self.ifo.start_time,
        )
        whitened_strain = self.ifo.whitened_time_domain_strain * np.array(sigma)
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain






class GenerateWhitenedIFONoise_fromGWF_multidetector(GenerateData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, use_mask, times):
        super(GenerateWhitenedIFONoise_fromGWF, self).__init__(
            parameters=dict(sigma=None),
            call_parameter_key_list=["sigma"],
        )
        self.old_ifo=ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        sigma = self.parameters["sigma"]
        self.ifon = InterferometerList([self.old_ifo[0].name])
        self.ifon.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.old_ifo.sampling_frequency,
            duration=self.old_ifo.duration,
            start_time=self.old_ifo.start_time,
        )
        self.ifon.maximum_frequency=self.old_ifo.sampling_frequency/2
        self.ifo=self.ifon[0]
        noise=self.ifo.strain_data.time_domain_strain
        
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(noise), alpha=alpha)
        window_factor = np.mean(window ** 2)
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        
        hf = np.fft.rfft(noise*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        hf_whitened=hf/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        ht_whitened=(np.fft.irfft(hf_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        whitened_strain = ht_whitened * np.array(sigma)
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain


class GenerateWhitenedSignal_fromGWF_multidetector(GenerateData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior, use_mask, times):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()

        super(GenerateWhitenedSignal_fromGWF, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
    def get_data(self, parameters: dict):
        self.parameters.update(parameters)
        parameters = self.parameters

        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        frequencies=None
        if frequencies is None:
            frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
            mask = self.ifo.frequency_mask
        
        time_data=self.ifo.strain_data.time_domain_strain
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(time_data), alpha=alpha)
        window_factor = np.mean(window ** 2)
        
        frequency_data = np.fft.rfft(time_data*window) / self.ifo.sampling_frequency*self.ifo.frequency_mask
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        frequency_data_whitened=frequency_data/ (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        data_noise_whitened=(np.fft.irfft(frequency_data_whitened)*np.sqrt(np.sum(self.ifo.frequency_mask))/frequency_window_factor)
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask#*window_factor
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= frequency_data+ signal_ifo
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (self.ifo.amplitude_spectral_density_array*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(self.ifo.frequency_mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde-data_noise_whitened
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened


##################################################  Real Data  #########################################################################################################################################

class GenerateWhitenedIFONoise_realData(GenerateRealData):
    """
    TBD
    Parameters
    ==========
    num_data:
    """

    def __init__(self, ifo, noise_prior, data, psd, use_mask, times, max_simulations):
        call_parameter_key_list = noise_prior.non_fixed_keys
        parameters = noise_prior.sample()
        
        super(GenerateWhitenedIFONoise_realData, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
            simulation_number=max_simulations,
        )
        self.ifo=ifo
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
        self.data=data
        self.psd=psd
    def get_data(self, parameters: dict, simulation_number):
        self.parameters.update(parameters)
        self.simulation_number=simulation_number
        sigma = self.parameters["sigma_"+self.ifo.name.lower()]
        psd=self.psd[self.ifo.name][self.simulation_number] #Start again here
        frequency_array=psd.frequencies
        frequency_mask = ((np.array(frequency_array) >= 20)&(np.array(frequency_array) <= self.ifo.sampling_frequency/2))
        noise=self.data[self.ifo.name][self.simulation_number]
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(len(noise), alpha=alpha)
        window_factor = np.mean(window ** 2)
        frequency_window_factor = (np.sum(frequency_mask)/ len(frequency_mask))
        
        hf = np.fft.rfft(noise*window) / self.ifo.sampling_frequency*frequency_mask   #Careful in using sampling frequency and duration from the ifo object
        hf_whitened=hf/ (np.sqrt(np.array(psd))*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4))
        ht_whitened=(np.fft.irfft(hf_whitened)*np.sqrt(np.sum(frequency_mask))/frequency_window_factor)
        whitened_strain = ht_whitened * np.array(sigma)
        whitened_strain.shift(self.ifo.start_time-np.array(whitened_strain.times[0]))
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            whitened_strain=whitened_strain[mask]
        return whitened_strain
    
class GenerateWhitenedSignal_realData(GenerateRealData):
    """
    TBD

    Parameters
    ==========
    ifo:

    waveform_generator:

    bilby_prior:
    """

    def __init__(self, ifo, waveform_generator, signal_prior,data, psd, use_mask, times, max_simulations):
        call_parameter_key_list = signal_prior.non_fixed_keys
        parameters = signal_prior.sample()
        
        super(GenerateWhitenedSignal_realData, self).__init__(
            parameters=parameters,
            call_parameter_key_list=call_parameter_key_list,
            simulation_number=max_simulations,
        )
        self.ifo = ifo
        self.waveform_generator = waveform_generator
        self.use_mask=use_mask
        self.time_lower=times[0]
        self.time_upper=times[1]
        self.data=data
        self.psd=psd

    
    def get_data(self, parameters: dict, simulation_number=0, psd=None):
        self.parameters.update(parameters)
        self.simulation_number=simulation_number
        parameters = self.parameters
        
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        if psd is None:
            psd=self.psd[self.ifo.name][self.simulation_number] #Start again here
        else:
            psd=psd
            
        roll_off=0.2
        alpha=2*roll_off/self.ifo.duration
        window = tukey(int(self.ifo.duration*self.ifo.sampling_frequency), alpha=alpha)
        window_factor = np.mean(window ** 2)
        frequencies = self.ifo.frequency_array[self.ifo.frequency_mask]
        mask = self.ifo.frequency_mask
        frequency_window_factor = (np.sum(self.ifo.frequency_mask)/ len(self.ifo.frequency_mask))
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.ifo.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)
        
            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask#*window_factor
        
        time_shift = self.ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])
        
        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.ifo.strain_data.start_time
        dt = dt_geocent + time_shift
        
        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)
        
        signal_ifo[mask] *= self.ifo.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.ifo.name), **parameters
        )
        frequency_domain_signal= signal_ifo
        
        ht_tilde = (
            np.fft.irfft(frequency_domain_signal / (np.sqrt(np.array(psd))*np.sqrt(window_factor) * np.sqrt(self.ifo.duration / 4)))
            * np.sqrt(np.sum(mask)) / frequency_window_factor
        )
        ht_whitened=ht_tilde
        #Taking only the piece necessary for the training
        if self.use_mask:
            window_start = self.ifo.start_time +(self.ifo.duration/2.)- self.time_lower
            window_end = self.ifo.start_time + (self.ifo.duration/2.) + self.time_upper
            mask = (self.ifo.time_array >= window_start) & (self.ifo.time_array <= window_end)
            ht_whitened=ht_whitened[mask]

        return ht_whitened



######################## Functions to work with real data ###################################

def create_ifo(data, ifo, trigger_time, inter,psd_yobs_use, waveform_generator, injection_parameters):
     psd_yobs=[]
     
     yobs_first=data[ifo+'_trigger']
     psd_yobs.append(psd_yobs_use)
     #Create ifo with the data from the detector
     ifon = bilby.gw.detector.get_empty_interferometer(ifo)
     ifon.strain_data.set_from_gwpy_timeseries(data[ifo+'_trigger'])
     ifon.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
         frequency_array=psd_yobs_use.frequencies.value, psd_array=psd_yobs_use.value
     )
     ifon.maximum_frequency=2048
     ifon.minimum_frequency = 20
     #Inject signal in ifo
     _ = ifon.inject_signal(
     waveform_generator=waveform_generator, parameters=injection_parameters
 )
     return ifon

def load_or_generate_data(trigger_time, interferometers, data_duration, psd_data_length=0, data_dir=".", label="data", cache=True):
    """
    Loads data from a pickle file if it exists, otherwise generates and saves it.

    Parameters:
        trigger_time (float): GPS trigger time
        interferometers (list): List of interferometers
        data_duration (int): Duration of the data segment
        psd_data_length (int): Extra duration if needed for PSD computation
        data_dir (str): Directory to store/load the data
        label (str): Label for the file name

    Returns:
        data (dict): The data dictionary (either loaded or newly generated)
    """
    total_duration = data_duration + psd_data_length
    file_name = f"{data_dir}/{label}_{trigger_time}_{total_duration}.pkl"

    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            print(f"Loading {label} from existing file: {file_name}")
            return pickle.load(f)
    else:
        print(f"{label} file not found. Generating new data...")
        data = get_data(trigger_time, interferometers, total_duration, cache)
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
        return data



def get_data(trigger_time, ifos,duration, cache=True):
 data={}
 start_before=trigger_time-10-duration
 end_before=trigger_time-10
 start_after=trigger_time+10
 end_after=trigger_time+10+duration
 for ifo in ifos:
     data[ifo+'_before']=TimeSeries.fetch_open_data(ifo, start_before, end_before, cache=cache)
     data[ifo+'_after']=TimeSeries.fetch_open_data(ifo, start_after, end_after, cache=cache)
     data[ifo+'_trigger']=TimeSeries.fetch_open_data(ifo, trigger_time-2, trigger_time+2, cache=cache)
 return data 






def slice_data_and_compute_psd(data, data_psd, interferometers, data_duration, sample_rate=4096):
    """
    Slice time-series data into 4s chunks and compute PSD from 16s chunks.
    
    Parameters:
        data (dict): Dictionary containing time-series data for each interferometer and segment ('before'/'after')
        data_psd (dict): Dictionary containing time-series data for PSD computation
        interferometers (list): List of interferometer names (e.g., ['H1', 'L1'])
        data_duration (int): Duration of the total data segment in seconds
        sample_rate (int): Sampling rate in Hz (default: 4096)
        
    Returns:
        data_4s (dict): Dictionary with 4s sliced data chunks per interferometer
        psd (dict): Dictionary with PSDs computed from 16s chunks per interferometer
    """
    data_4s = {}
    data_psd_16s = {}
    psd = {}

    for ifo in interferometers:
        data_4s[ifo] = []
        data_psd_16s[ifo] = []

        for name in ['before', 'after']:
            for i in range(data_duration - 4):
                start = int(i * sample_rate)
                end_4s = start + 4 * sample_rate
                end_16s = start + 16 * sample_rate

                data_4s[ifo].append(data[f"{ifo}_{name}"][start:end_4s])
                data_psd_16s[ifo].append(data_psd[f"{ifo}_{name}"][start:end_16s])

        # Compute PSD from each 16s segment
        psd[ifo] = [
            segment.psd(fftlength=4, overlap=None, window='hann', method='median')
            for segment in data_psd_16s[ifo]
        ]

    return data_4s, psd
