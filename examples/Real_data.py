#!/home/mattia.emma/.conda/envs/sbi/bin/python

import sys
import time
from collections import namedtuple
import json
import argparse
import os
import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import copy
import pickle
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey

import sbilby
from sbilby.simulation_based_inference_multidetector_realData import GenerateRealData
from sbilby.simulation_based_inference_multidetector_realData import AdditiveSignalAndNoise_realData
import matplotlib.pyplot as plt
from sbilby.data_generation import GenerateWhitenedIFONoise_realData
from sbilby.data_generation import GenerateWhitenedSignal_realData

########################################### Running sampler ######################################################
class SamplerLikelihood(object):
    def __init__(
        self,
        nrle_likelihood,
        priors,
        outdir,
        injection_parameters,
        training_time,
    ):
        self.nrle_likelihood = nrle_likelihood
        self.priors = priors
        self.outdir = outdir
        self.injection_parameters = injection_parameters
        self.statistics = dict(
            likelihood_class=nrle_likelihood.__class__.__name__,
            likelihood_metadata=nrle_likelihood.meta_data,
        )
        self.training_time=training_time

    def _time_likelihood(self, likelihood, n, name):
        eval_times = []
        for _ in range(n):
            likelihood.parameters.update(self.priors[0].sample())
            start = time.time()
            likelihood.log_likelihood()
            end = time.time()
            eval_times.append(end - start)
        self.statistics[f"likelihood_{name}_eval_time_mean"] = float(
            np.mean(eval_times)
        )
        self.statistics[f"likelihood_{name}_eval_time_std"] = float(np.std(eval_times))

    def benchmark_time(self, n=100):
        self._time_likelihood(self.rnle_likelihood, n, "rnle")
        self.statistics["likelihood_training_time"]=self.training_time
    def benchmark_posterior_sampling(self, run_sampler_kwargs=None):
        kwargs = dict(nlive=1000, sampler="dynesty", dlogz=0.5, check_point_delta_t=60)
        if run_sampler_kwargs is not None:
            kwargs.update(run_sampler_kwargs)

        
        result_nrle = bilby.run_sampler(
            likelihood=self.nrle_likelihood,
            priors=self.priors,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.nrle_likelihood.labels[0],
            #conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )

        for key, val in self.priors.items():
            if val.is_fixed is False:
                samplesA = result_nrle.posterior[key]

                fig, ax = plt.subplots()
                ax.hist(samplesA, bins=50, alpha=0.8, label="RNLE", density=True)
                ax.axvline(self.injection_parameters[key], color="k")
                ax.legend()
                plt.savefig(
                        f"{self.outdir}/{self.benchmark_likelihood.label}_1D_posterior_{key}.png"
                )




########################################## System commands #######################################################
print(f"Running command {' '.join(sys.argv)}")

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int,default=1)
parser.add_argument("--likelihood", type=str, default="RNLE")
parser.add_argument("--num-simulations", type=int, default=500)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--nlive", type=int, default=1500)
parser.add_argument("--dlogz", type=float, default=0.1)
parser.add_argument("--rseed", type=int, default=42)
parser.add_argument("--time_lower", type=float, default=0)
parser.add_argument("--time_upper", type=float, default=0)
parser.add_argument("--fs", type=float, default=4096)
args = parser.parse_args()

np.random.seed(args.rseed)
times=[args.time_lower, args.time_upper]
num_simulations = args.num_simulations

################################################## Set-up event info#############################################
interferometers=["H1","L1"]  
# Standard Bilby
injection_parameters = dict(
    chirp_mass=9.72,
    mass_ratio=0.68,
    a_1=0.56,
    a_2=0.45,
    tilt_1=1.14,
    tilt_2=1.28,
    phi_12=3.21,
    phi_jl=3.03,
    luminosity_distance=608.33,
    theta_jn=2.2,
    psi=1.59,
    phase=3.23,
    geocent_time=1259514944.0,
    ra=1.31,
    dec=-0.56,
)

signal_priors = bilby.gw.prior.BBHPriorDict()
signal_priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=5, maximum=15, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
#signal_priors['geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters["geocent_time"]-0.1,maximum=injection_parameters["geocent_time"]+0.1)
for key in [
    "a_1",
    "a_2",
    #"chirp_mass",
    "mass_ratio",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "phase",
    "geocent_time",
    "luminosity_distance",
    "theta_jn"
]:
    signal_priors[key] = injection_parameters[key]
noise_priors_h1 = bilby.core.prior.PriorDict(dict(sigma_h1=bilby.core.prior.Uniform(0, 2, 'sigma_h1')))
noise_priors_l1 = bilby.core.prior.PriorDict(dict(sigma_l1=bilby.core.prior.Uniform(0, 2, 'sigma_l1')))

duration = 4.0
sampling_frequency = 4096
minimum_frequency = 20
start_time = injection_parameters["geocent_time"] - duration / 2

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
    maximum_frequency=sampling_frequency/2,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)


ifos = bilby.gw.detector.InterferometerList(interferometers)
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)

# SBI setup
genA_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)



priors_h1 = noise_priors_h1 | signal_priors 
priors_h1 = bilby.core.prior.PriorDict(copy.deepcopy(priors_h1))
priors_h1.convert_floats_to_delta_functions()

priors_l1 = noise_priors_l1 | signal_priors 
priors_l1 = bilby.core.prior.PriorDict(copy.deepcopy(priors_l1))
priors_l1.convert_floats_to_delta_functions()

noise_priors= [noise_priors_h1, noise_priors_l1]
nrle_priors=[priors_h1, priors_l1]
priors= noise_priors_h1 | signal_priors | noise_priors_l1
priors = bilby.core.prior.PriorDict(copy.deepcopy(priors))
priors.convert_floats_to_delta_functions()

#################################################### Setup data####################################################################
def get_data(trigger_time, ifos,duration):
 data={}
 start_before=trigger_time-10-duration
 end_before=trigger_time-10
 start_after=trigger_time+10
 end_after=trigger_time+10+duration
 for ifo in ifos:
     data[ifo+'_before']=TimeSeries.fetch_open_data(ifo, start_before, end_before, cache=True)
     data[ifo+'_after']=TimeSeries.fetch_open_data(ifo, start_after, end_after, cache=True)
     data[ifo+'_trigger']=TimeSeries.fetch_open_data(ifo, trigger_time-2, trigger_time+2, cache=True)
 return data  

trigger_time=1259514944.0
data_duration=300
data= get_data(trigger_time, interferometers, data_duration)
psd_data_length=16
data_psd=get_data(trigger_time, interferometers, data_duration+psd_data_length)
# Dump dictionary to a pickle file
with open("/home/mattia.emma/public_html/NLE/sbilbi/data/data_"+str(trigger_time)+"_"+str(data_duration)+".pkl", "wb") as f:
    pickle.dump(data, f)
# Load dictionary from the pickle file
#with open("/home/mattia.emma/public_html/NLE/sbilbi/data/data_"+str(trigger_time)+"_"+str(data_duration)+".pkl", "rb") as f:
#    loaded_data = pickle.load(f)

data_4s={}
data_psd_16s={}
for ifo in interferometers:
    data_4s[ifo]=[]
    data_psd_16s[ifo]=[]
    for name in ['before', 'after']:
        for i in range(data_duration-4):
            data_4s[ifo].append(data[ifo+'_'+name][i*4096:i*4096+4096*4])
            data_psd_16s[ifo].append(data_psd[ifo+'_'+name][i*4096:i*4096+4096*16])

psd={}
for ifo in interferometers:
   psd[ifo]=[]
   for i in data_psd_16s[ifo]:
       psd[ifo].append(i.psd(fftlength=4, overlap=None, window='hann', method='median'))


######################################## Create observational data #############################################
use_mask=True
times=[0.8,0.2]
yobs=[]
psd_yobs=[]
for ifo in interferometers:
    yobs_first=data[ifo+'_trigger']
    psd_yobs_data=TimeSeries.fetch_open_data(ifo, trigger_time-18, trigger_time-2, cache=True)
    psd_yobs_use=psd_yobs_data.psd(fftlength=4, overlap=None, window='hann', method='median')
    psd_yobs.append(psd_yobs_use)
    frequency_array=psd_yobs_use.frequencies
    frequency_mask = ((np.array(frequency_array) >= 20)&(np.array(frequency_array) <= ifos[0].sampling_frequency/2))
    
    
    roll_off=0.2
    alpha=2*roll_off/ifos[0].duration
    window = tukey(len(yobs_first), alpha=alpha)
    window_factor = np.mean(window ** 2)
    frequency_window_factor = (np.sum(frequency_mask)/ len(frequency_mask))
    
    yobsf = np.fft.rfft(yobs_first*window) / ifos[0].sampling_frequency*frequency_mask   #Careful in using sampling frequency and duration from the ifo object
    yobsf_whitened=yobsf/ (np.sqrt(np.array(psd_yobs_use))*np.sqrt(window_factor) * np.sqrt(ifos[0].duration / 4))
    yobst_whitened=(np.fft.irfft(yobsf_whitened)*np.sqrt(np.sum(frequency_mask))/frequency_window_factor)
    window_start = ifos[0].start_time +(ifos[0].duration/2.)- times[0]
    window_end = ifos[0].start_time + (ifos[0].duration/2.) + times[1]
    mask = (ifos[0].time_array >= window_start) & (ifos[0].time_array <= window_end)
    yobst_cut = yobst_whitened[mask]
    yobs.append(yobst_cut)
plt.plot(yobs[0])
plt.savefig("yobs.png")
############################################### Feed it to sbi #################################################

noise=GenerateWhitenedIFONoise_realData(ifos[0],copy.deepcopy(noise_priors_h1),data_4s,psd,use_mask, times, len(data_4s['H1']))
signal=GenerateWhitenedSignal_realData(ifos[0], waveform_generator, copy.deepcopy(signal_priors),data_4s,psd, use_mask, times, len(data_4s['H1'])) 
signal_and_noise = [AdditiveSignalAndNoise_realData(signal, noise, len(data_4s['H1']))]   #Everything has to be in a list if we are using the multidetector code

label = f"N{num_simulations}_fs{sampling_frequency}_seed{args.rseed}_R{args.repeat}_D{args.dimensions}"
interferometer=['H1']
benchmark_likelihood = sbilby.simulation_based_inference_multidetector_realData.NLEResidualLikelihood_realData(
        yobs[0],
        psd_yobs[0],
        signal_and_noise,
        interferometer,
        bilby_priors=copy.deepcopy(nrle_priors),
        labels=label,
        num_simulations=num_simulations,
        cache_directory='likelihood_cache',
        cache=True,
        #show_progress_bar=True,
)


start=time.time()
benchmark_likelihood.init()
end=time.time()
training_time=end-start

# ######################################   Benchmark ##################################
outdir="Test"
bench = SamplerLikelihood(
    benchmark_likelihood,
    priors,
    outdir,
    injection_parameters=injection_parameters,
    training_time=training_time,
    
)
#bench.benchmark_time()
bench.benchmark_posterior_sampling(
    dict(
        sampler="dynesty",
        nlive=args.nlive,
        dlogz=args.dlogz,
        resume=args.resume,
        print_method="interval-10",
        npool=8,
        sample="acceptance-walk",
        check_point_delta_t=180,
    )
)