#!/home/mattia.emma/.conda/envs/sbi_bug_fix/bin/python

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
import random
import sbilby
from sbilby.simulation_based_inference_multidetector_realData import GenerateRealData
from sbilby.simulation_based_inference_multidetector_realData import AdditiveSignalAndNoise_realData
import matplotlib.pyplot as plt
from sbilby.data_generation import GenerateWhitenedIFONoise_realData
from sbilby.data_generation import GenerateWhitenedSignal_realData
from sbilby.data_generation import *
########################################### Running sampler ######################################################
class BenchmarkLikelihood(object):
    def __init__(
        self,
        nrle_likelihood,
        reference_likelihood,
        priors,
        reference_prior,
        outdir,
        injection_parameters,
        training_time,
    ):
        self.nrle_likelihood = nrle_likelihood
        self.reference_likelihood = reference_likelihood
        self.priors = priors
        self.reference_prior = reference_prior
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
            likelihood.parameters.update(self.priors.sample())
            start = time.time()
            likelihood.log_likelihood()
            end = time.time()
            eval_times.append(end - start)
        self.statistics[f"likelihood_{name}_eval_time_mean"] = float(
            np.mean(eval_times)
        )
        self.statistics[f"likelihood_{name}_eval_time_std"] = float(np.std(eval_times))

    def benchmark_time(self, n=100):
        self._time_likelihood(self.nrle_likelihood, n, "nrle")
        self._time_likelihood(self.reference_likelihood, n, "reference")
        self.statistics["likelihood_training_time"]=self.training_time
    def benchmark_posterior_sampling(self, run_sampler_kwargs=None):
        kwargs = dict(nlive=1000, sampler="dynesty", dlogz=0.5, check_point_delta_t=60)
        if run_sampler_kwargs is not None:
            kwargs.update(run_sampler_kwargs)

        result_reference = bilby.run_sampler(
            likelihood=self.reference_likelihood,
            priors=self.reference_prior,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.nrle_likelihood.labels[0] + "_REFERENCE",
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )
        
        result_nrle = bilby.run_sampler(
            likelihood=self.nrle_likelihood,
            priors=self.priors,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.nrle_likelihood.labels[0],
            #conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )

        for key, val in self.reference_prior.items():
            if val.is_fixed is False:
                samplesA = result_nrle.posterior[key]
                samplesB = result_reference.posterior[key]
                js = calculate_js(samplesA, samplesB)
                self.statistics[f"1D_posterior_JS_{key}_median"] = js.median
                self.statistics[f"1D_posterior_JS_{key}_plus"] = js.plus
                self.statistics[f"1D_posterior_JS_{key}_minus"] = js.minus

                fig, ax = plt.subplots()
                ax.hist(samplesA, bins=50, alpha=0.8, label="RNLE", density=True)
                ax.hist(samplesB, bins=50, alpha=0.8, label="Bilby", density=True)
                ax.axvline(self.injection_parameters[key], color="k")
                ax.set(xlabel=key, title=f"JS={js.median}")
                ax.legend()
                plt.savefig(
                        f"{self.outdir}/{self.nrle_likelihood.labels}_1D_posterior_{key}.png"
                )
                    

    def write_results(self):
        bilby.utils.check_directory_exists_and_if_not_mkdir("RESULTS_real_data_injection")
        with open(
            f"RESULTS_real_data_injection/result_nrle_{self.nrle_likelihood.labels}.json", "w"
        ) as file:
            json.dump(self.statistics, file, indent=4)
            
def calc_summary(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    summary = namedtuple("summary", ["median", "lower", "upper"])
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    return summary


def calculate_js(samplesA, samplesB, ntests=100, xsteps=100):
    js_array = np.zeros(ntests)
    for j in range(ntests):
        nsamples = min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)
          
        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf), 2))

    return calc_summary(js_array)
########################################## System commands #######################################################
print(f"Running command {' '.join(sys.argv)}")

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int,default=1)
parser.add_argument("--likelihood", type=str, default="RNLE")
parser.add_argument("--num-simulations", type=int, default=5)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--nlive", type=int, default=1500)
parser.add_argument("--dlogz", type=float, default=0.1)
parser.add_argument("--rseed", type=int, default=42)
parser.add_argument("--time_lower", type=float, default=0)
parser.add_argument("--time_upper", type=float, default=0)
parser.add_argument("--fs", type=float, default=4096)
parser.add_argument("--duration", type=int, default=300)
args = parser.parse_args()

np.random.seed(args.rseed)
times=[args.time_lower, args.time_upper]
num_simulations = args.num_simulations

################################################## Set-up event info#############################################
interferometers=["H1","L1"]  
# Standard Bilby
#chirp_mass_value=random.uniform(26, 30)
injection_parameters = dict(
    chirp_mass=28,
    mass_ratio=0.82,
    a_1=0.32,
    a_2=0.44,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=1400.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1249439118.0 ,
    ra=1.375,
    dec=-1.2108,
)


signal_priors = bilby.gw.prior.BBHPriorDict()
signal_priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=24, maximum=32, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
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

#######################Reference Priors ##################
reference_priors = copy.deepcopy(signal_priors)
reference_priors = bilby.core.prior.PriorDict(reference_priors)
reference_priors.convert_floats_to_delta_functions()


#################################################### Setup data####################################################################
 

trigger_time = 1249439118.0
data_duration = args.duration
psd_data_length = 16

# Load or generate main data
data = load_or_generate_data(trigger_time, interferometers, data_duration, psd_data_length=0, label="data", cache=False)

# Load or generate PSD data
data_psd = load_or_generate_data(trigger_time, interferometers, data_duration, psd_data_length=psd_data_length, label="psd_data", cache=False)


data_4s, psd = slice_data_and_compute_psd(data, data_psd, interferometers, data_duration)


######################################## Create observational data #############################################
ifo_list = bilby.gw.detector.InterferometerList([])
reference_ifo_list=bilby.gw.detector.InterferometerList([])





use_mask=True
times=[0.8,0.2]
yobs=[]
psd_yobs=[]
signals=[]
m=0
for ifo in interferometers:
    psd_yobs_data=TimeSeries.fetch_open_data(ifo, trigger_time-18, trigger_time-2, cache=True)
    psd_yobs_use=psd_yobs_data.psd(fftlength=4, overlap=None, window='hann', method='median')
    psd_yobs.append(psd_yobs_use)
    ifo_list.append(create_ifo(data, ifo, trigger_time, ifos[m],psd_yobs_use, waveform_generator, injection_parameters))
    window_start = ifos[m].start_time +(ifos[m].duration/2.)- times[0]
    window_end = ifos[m].start_time + (ifos[m].duration/2.) + times[1]
    mask = (ifos[m].time_array >= window_start) & (ifos[m].time_array <= window_end)
    sig=ifo_list[m].whitened_time_domain_strain[mask]
    yobs.append(sig)
    m+=1
###########################################Reference Likelihood ##############################
reference_ifo_list.append(ifo_list[0])
reference_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(reference_ifo_list, waveform_generator)
############################################### Feed it to sbi #################################################

noise=GenerateWhitenedIFONoise_realData(ifos[0],copy.deepcopy(noise_priors_h1),data_4s,psd,use_mask, times, len(data_4s['H1']))
signal=GenerateWhitenedSignal_realData(ifos[0], genA_waveform_generator, copy.deepcopy(signal_priors),data_4s,psd, use_mask, times, len(data_4s['H1'])) 
signal_and_noise = [AdditiveSignalAndNoise_realData(signal, noise, len(data_4s['H1']))]   #Everything has to be in a list if we are using the multidetector code

label = [f"N{num_simulations}_fs{sampling_frequency}_seed{args.rseed}_R{args.repeat}_D{args.dimensions}_d{args.duration}"]
interferometer=['H1']
benchmark_likelihood = sbilby.simulation_based_inference_multidetector_realData.NLEResidualLikelihood_realData(
        yobs,
        psd_yobs,
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

# ############################################# Plot the likelihood before starting to sample ################################
benchmark_likelihood.parameters.update(dict(sigma_h1=1.0))
reference_likelihood.parameters.update(injection_parameters)
benchmark_likelihood.log_likelihood()
x = "chirp_mass"
deltax = 8
xtrue = injection_parameters[x]
x_array = np.linspace(xtrue - deltax / 2, xtrue + deltax / 2, 500)

logl = []
logl_bilby=[]
for xval in x_array:
    benchmark_likelihood.parameters[x] = xval
    reference_likelihood.parameters[x] = xval
    logl.append(benchmark_likelihood.log_likelihood())
    logl_bilby.append(reference_likelihood.log_likelihood())

fig, ax = plt.subplots(figsize = (6.5, 4.5))    
ax.plot(x_array, np.exp(np.array(logl)/np.abs(np.max(logl))), label='RNLE', color='orange')
ax2 = ax.twinx()
ax2.plot(x_array, np.exp(np.array(logl_bilby)/np.abs(np.max(logl_bilby))),'--', label='Whittle', color="blue", alpha=0.4)

ax.set_xlabel(r"$\mathcal{M}/M_\odot$")
ax.set_ylabel(r'RNLE $\log{\mathcal{L}}$')
ax2.set_ylabel(r'Whittle $\log{\mathcal{L}}$')
ax.axvline(xtrue, color='black')
#ax.set_title('NLE '+str(num_simulations)+'sim')
#ax.set_xlim(29.7,30.3)
ax.legend()
ax2.legend(loc="upper left")
plt.tight_layout()
plt.savefig("RNLE_"+str(num_simulations)+"_"+str(args.rseed)+"likelihood.png", dpi=300)
plt.show()
benchmark_likelihood.labels[0]=f"N{num_simulations}_seed{args.rseed}_R{args.repeat}"
# # ######################################   Benchmark ##################################

outdir="real_data_injection_bug_fix_change_seed"
bench = BenchmarkLikelihood(
    benchmark_likelihood,
    reference_likelihood,
    priors,
    reference_priors,
    outdir,
    injection_parameters=injection_parameters,
    training_time=training_time,
    
)
bench.benchmark_time()
bench.benchmark_posterior_sampling(
    dict(
        sampler="dynesty",
        nlive=args.nlive,
        dlogz=args.dlogz,
        resume=args.resume,
        print_method="interval-10",
        npool=2,
        sample="acceptance-walk",
        check_point_delta_t=180,
    )
)
bench.write_results()
