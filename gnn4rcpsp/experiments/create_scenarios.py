from discrete_optimization.rcpsp.rcpsp_parser import parse_file
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, create_poisson_laws_duration, \
    create_poisson_laws, UncertainRCPSPModel, MethodBaseRobustification, MethodRobustification
import os
import pickle

def create_diff_scenario(base_rcpsp_model, nb_scenarios: int = 20):
    uniform_law = create_poisson_laws(
            base_rcpsp_model=base_rcpsp_model,
            range_around_mean_resource=1,
            range_around_mean_duration=3,
            do_uncertain_resource=False,
            do_uncertain_duration=True,
        )
    uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=base_rcpsp_model,
            poisson_laws={
                task: laws
                for task, laws in uniform_law.items()
                if task in base_rcpsp_model.mode_details
            },
            uniform_law=True,
        )
    sampled_scenario = [uncertain_rcpsp.create_rcpsp_model(MethodRobustification(MethodBaseRobustification.SAMPLE))
                        for j in range(nb_scenarios)]
    return sampled_scenario, base_rcpsp_model, uniform_law


def precompute_scenarios():
    # Create a big binary file storing many sampled scenarios.
    this_dir = os.path.dirname(__file__)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kobe_rcpsp_dir = os.path.join(root_dir, "../kobe-rcpsp/data/rcpsp")
    subfolders_j = [os.path.join(kobe_rcpsp_dir, f) for f in os.listdir(kobe_rcpsp_dir) if ".sm" in f]
    precomputed_scenarios = []
    # we restrict to the psplib instances.
    for subfolder in subfolders_j:
        print(subfolder)
        files = os.listdir(subfolder)
        for f in files:
            if ".sm" in f:
                base_rcpsp = parse_file(os.path.join(subfolder, f))
                scenarios = create_diff_scenario(base_rcpsp_model=base_rcpsp, nb_scenarios=10)
                precomputed_scenarios += [{"file": os.path.join(subfolder, f),
                                           "base_model": scenarios[1],
                                           "sampled_scenarios": scenarios[0],
                                           "uniform_range": scenarios[2]}]
    pickle.dump(precomputed_scenarios, open(os.path.join(this_dir, "experiments_precomputed_scenarios.pk"),
                                            "wb"))


def load_precompute_scenarios():
    this_dir = os.path.dirname(__file__)
    precomputed_scenarios = pickle.load(open(os.path.join(this_dir, "experiments_precomputed_scenarios.pk"), 'rb'))
    print(len(precomputed_scenarios))


if __name__ == "__main__":
    precompute_scenarios()