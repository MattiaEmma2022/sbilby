#!/home/mattia.emma/.conda/envs/sbi/bin/python

import sys
import time
from collections import namedtuple
import json
import argparse

import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

#import bilby.core.likelihood.simulation_based_inference as sbibilby
import sbilby
import sbilby.simulation_based_inference as sbibilby

class BenchmarkLikelihood(object):
    def __init__(
        self,
        benchmark_likelihood,
        reference_likelihood,
        prior,
        outdir,
        simulation_parameters,
    ):
        self.benchmark_likelihood = benchmark_likelihood
        self.reference_likelihood = reference_likelihood
        self.prior = prior
        self.outdir = outdir
        self.simulation_parameters = simulation_parameters
        self.statistics = dict(
            likelihood_class=benchmark_likelihood.__class__.__name__,
            likelihood_metadata=benchmark_likelihood.meta_data,
        )

    def _time_likelihood(self, likelihood, n, name):
        eval_times = []
        for _ in range(n):
            likelihood.parameters.update(self.prior.sample())
            start = time.time()
            likelihood.log_likelihood()
            end = time.time()
            eval_times.append(end - start)
        self.statistics[f"likelihood_{name}_eval_time_mean"] = float(
            np.mean(eval_times)
        )
        self.statistics[f"likelihood_{name}_eval_time_std"] = float(np.std(eval_times))

    def benchmark_time(self, n=100):
        self._time_likelihood(self.benchmark_likelihood, n, "benchmark")
        self._time_likelihood(self.reference_likelihood, n, "reference")

    def benchmark_posterior_sampling(self, run_sampler_kwargs=None):
        kwargs = dict(nlive=1000, sampler="dynesty", dlogz=0.5)
        if run_sampler_kwargs is not None:
            kwargs.update(run_sampler_kwargs)

        result_reference = bilby.run_sampler(
            likelihood=self.reference_likelihood,
            priors=self.prior,
            outdir=self.outdir,
            label=self.benchmark_likelihood.label + "_REFERENCE",
            **kwargs,
        )

        result_benchmark = bilby.run_sampler(
            likelihood=self.benchmark_likelihood,
            priors=self.prior,
            outdir=self.outdir,
            label=self.benchmark_likelihood.label,
            **kwargs,
        )

        for key, val in self.prior.items():
            if val.is_fixed is False:
                samplesA = result_benchmark.posterior[key]
                samplesB = result_reference.posterior[key]
                js = calculate_js(samplesA, samplesB)
                self.statistics[f"1D_posterior_JS_{key}_median"] = js.median
                self.statistics[f"1D_posterior_JS_{key}_plus"] = js.plus
                self.statistics[f"1D_posterior_JS_{key}_minus"] = js.minus

                fig, ax = plt.subplots()
                ax.hist(samplesA, bins=50, alpha=0.8, label="Benchmark")
                ax.hist(samplesB, bins=50, alpha=0.8, label="Reference")
                ax.axvline(self.simulation_parameters[key], color="k")
                ax.set(xlabel=key, title=f"JS={js.median}")
                ax.legend()
                plt.savefig(
                    f"{self.outdir}/{self.benchmark_likelihood.label}_1D_posterior_{key}.png"
                )

    def write_results(self):
        bilby.utils.check_directory_exists_and_if_not_mkdir("RESULTS_second")
        with open(
            f"RESULTS_second/result_benchmark_{self.benchmark_likelihood.label}.json", "w"
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
        nsamples = 3000#min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)

        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf), 2))

    return calc_summary(js_array)


# Define a deterministic model
def sinegaussian(time_array, amplitude, frequency, alpha):
    return (
        amplitude
        * np.exp(-((time_array / alpha) ** 2))
        * np.sin(2 * np.pi * frequency * time_array)
    )


outdir = "outdir_benchmark_sine-gaussian"

print(f"Running command {' '.join(sys.argv)}")

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int, default=2)
parser.add_argument("--likelihood", type=str, default="RNLE")
parser.add_argument("--num-simulations", type=int, default=1000)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--nlive", type=int, default=1000)
parser.add_argument("--dlogz", type=float, default=0.1)
parser.add_argument("--rseed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.rseed)
num_simulations = args.num_simulations

# Construct the prior
prior = bilby.core.prior.PriorDict(
    dict(
        amplitude=bilby.core.prior.Uniform(9, 11, "amplitude"),
        frequency=bilby.core.prior.Uniform(1, 3, "frequency"),
        alpha=bilby.core.prior.Uniform(1, 3, "alpha"),
        sigma=bilby.core.prior.Uniform(0, 2, "amplitude"),
    )
)

# Define a set of simulation/injection parameters
simulation_parameters = prior.sample()

# Freeze to only n dimensions
for i in range(len(prior) - args.dimensions):
    key = list(prior.keys())[i]
    prior[key] = bilby.core.prior.DeltaFunction(simulation_parameters[key], key)


time_array = np.linspace(-10, 10, 250)
fixed_arguments_dict=dict(time_array=time_array)
for key, val in prior.items():
    if val.is_fixed:
        fixed_arguments_dict[key] = val.peak

generator = sbibilby.AdditiveWhiteGaussianNoise(
    model=sinegaussian,
    fixed_arguments_dict=fixed_arguments_dict,
    bilby_prior=prior,
)

yobs = generator.get_data(simulation_parameters)

reference_likelihood = bilby.core.likelihood.GaussianLikelihood(
    time_array, yobs, sinegaussian
)

label = f"SG_{args.likelihood}_D{args.dimensions}_N{num_simulations}_R{args.repeat}_plots"
if args.likelihood == "NLE":
    benchmark_likelihood = sbibilby.NLELikelihood(
        yobs=yobs,
        generator=generator,
        bilby_prior=prior,
        label=label,
        num_simulations=num_simulations,
        cache_directory=outdir,
    )
elif args.likelihood == "RNLE":
    benchmark_likelihood = sbibilby.NLEResidualLikelihood(
        yobs=yobs,
        generator=generator,
        bilby_prior=prior,
        label=label,
        num_simulations=num_simulations,
        cache_directory=outdir,
    )

benchmark_likelihood.init()

bench = BenchmarkLikelihood(
    benchmark_likelihood,
    reference_likelihood,
    prior,
    outdir,
    simulation_parameters=simulation_parameters,
)
bench.benchmark_time()
bench.benchmark_posterior_sampling(
    dict(
        nlive=args.nlive,
        dlogz=args.dlogz,
        resume=args.resume,
        print_method="interval-10",
    )
)
bench.write_results()
