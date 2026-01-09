# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code show to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm
and run it under different environments
"""
import argparse
import numpy as np
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking, ProdEnv

from nvflare.app_common.aggregators import ModelAggregator
from nvflare.client import FLModel


class GWASMetaAggregator(ModelAggregator):
    """
    Collects GWAS summary statistics from clients and performs
    inverse-variance weighted meta-analysis during aggregation.
    """

    def __init__(self):
        super().__init__()
        self.client_betas = []
        self.client_ses = []
        self.received_params_type = None

    def accept_model(self, model: FLModel):
        """
        Called once per client.
        Expects model.params to contain GWAS summary statistics.
        """
        if self.received_params_type is None:
            self.received_params_type = model.params_type

        params = model.params
        beta = np.asarray(params["beta"])
        se = np.asarray(params["se"])

        self.client_betas.append(beta)
        self.client_ses.append(se)

    def aggregate_model(self) -> FLModel:
        """
        Perform inverse-variance weighted GWAS meta-analysis.
        """
        betas = np.stack(self.client_betas, axis=0)  # (K, P)
        ses = np.stack(self.client_ses, axis=0)      # (K, P)

        variances = ses ** 2
        weights = 1.0 / variances

        # Meta-analysis estimates
        meta_beta = np.sum(weights * betas, axis=0) / np.sum(weights, axis=0)
        meta_var = 1.0 / np.sum(weights, axis=0)
        meta_se = np.sqrt(meta_var)

        aggregated_params = {
            "beta": meta_beta,
            "se": meta_se,
        }

        return FLModel(
            params=aggregated_params,
            params_type=self.received_params_type,
        )

    def reset_stats(self):
        """
        Clear state between FL rounds.
        """
        self.client_betas = []
        self.client_ses = []
        self.received_params_type = None


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=1)

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds

    recipe = FedAvgRecipe(
        name="fed_gwas",
        min_clients=n_clients,
        num_rounds=num_rounds,
        train_script="client.py",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Send the regenie script to all clients
    recipe.job.to_clients("client_regenie.sh")

    # Simulation Environment (FL Server and clients on same NVIDIA Brev instance)
    # env = SimEnv(num_clients=n_clients)
    
    # Production Environment (FL Server on AWS and clients on NVIDIA Brev)
    env = ProdEnv(startup_kit_location="/home/ubuntu/hroth@nvidia.com", username="hroth@nvidia.com")
    
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in :", run.get_result())
    print()


if __name__ == "__main__":
    main()
