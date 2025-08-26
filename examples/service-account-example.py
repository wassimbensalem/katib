#!/usr/bin/env python3
"""
Example demonstrating the new service account feature in Katib.

This example shows how to create a Katib experiment with a custom service account
for running trials with specific permissions.
"""

from kubeflow.katib import KatibClient
from kubeflow.katib import search
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective_function(parameters):
    """
    Example objective function for hyperparameter optimization.
    
    Args:
        parameters: Dictionary containing hyperparameters
        
    Returns:
        float: The objective metric value
    """
    lr = parameters.get('lr', 0.01)
    batch_size = parameters.get('batch_size', 32)
    
    # Simulate training and return a metric
    # In a real scenario, this would be actual model training
    import random
    random.seed(42)  # For reproducible results
    
    # Simulate training loss based on hyperparameters
    loss = 1.0 / (1.0 + lr * 100) + batch_size / 1000.0 + random.uniform(0, 0.1)
    
    logger.info(f"Trial completed - lr: {lr}, batch_size: {batch_size}, loss: {loss}")
    return loss


def create_experiment_with_service_account():
    """
    Create a Katib experiment using a custom service account.
    """
    # Initialize Katib client
    client = KatibClient()
    
    # Define hyperparameter search space
    parameters = {
        'lr': search.double(min=0.001, max=0.1),
        'batch_size': search.int(min=16, max=128)
    }
    
    # Create experiment with custom service account
    experiment_name = "service-account-demo"
    service_account = "katib-trial-sa"  # Custom service account
    
    logger.info(f"Creating experiment '{experiment_name}' with service account '{service_account}'")
    
    try:
        client.tune(
            name=experiment_name,
            objective=objective_function,
            parameters=parameters,
            base_image="python:3.9",
            max_trial_count=5,
            parallel_trial_count=2,
            service_account=service_account  # New service account parameter
        )
        
        logger.info(f"Experiment '{experiment_name}' created successfully!")
        logger.info(
            f"Trials will run with service account: {service_account}"
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise


def create_experiment_directly():
    """
    Create a Katib experiment directly using the create_experiment method.
    """
    from kubeflow.katib import models
    
    # Initialize Katib client
    client = KatibClient()
    
    # Create experiment object
    experiment = models.V1beta1Experiment(
        metadata=models.V1ObjectMeta(
            name="direct-service-account-demo",
            namespace="default"
        ),
        spec=models.V1beta1ExperimentSpec(
            parameters=[
                models.V1beta1ParameterSpec(
                    name="lr",
                    parameter_type="double",
                    feasible_space=models.V1beta1FeasibleSpace(
                        min="0.001",
                        max="0.1"
                    )
                ),
                models.V1beta1ParameterSpec(
                    name="batch_size",
                    parameter_type="int",
                    feasible_space=models.V1beta1FeasibleSpace(
                        min="16",
                        max="128"
                    )
                )
            ],
            objective=models.V1beta1ObjectiveSpec(
                type="minimize",
                objective_metric_name="loss"
            ),
            algorithm=models.V1beta1AlgorithmSpec(
                algorithm_name="random"
            ),
            max_trial_count=3,
            parallel_trial_count=2,
            # Note: trial_template would need to be set for a complete experiment
        )
    )
    
    service_account = "katib-trial-sa"
    
    logger.info(f"Creating experiment directly with service account '{service_account}'")
    
    try:
        client.create_experiment(
            experiment=experiment,
            service_account=service_account  # New service account parameter
        )
        
        logger.info("Experiment created directly with service account!")
        
    except Exception as e:
        logger.error(f"Failed to create experiment directly: {e}")
        raise


def main():
    """
    Main function to demonstrate the service account feature.
    """
    logger.info("=== Katib Service Account Feature Demo ===")
    
    # Example 1: Using the tune method
    logger.info("\n1. Creating experiment using tune method...")
    try:
        create_experiment_with_service_account()
    except Exception as e:
        logger.warning(f"Tune method failed (this is expected if service account doesn't exist): {e}")
    
    # Example 2: Using create_experiment method
    logger.info("\n2. Creating experiment directly...")
    try:
        create_experiment_directly()
    except Exception as e:
        logger.warning(f"Direct creation failed (this is expected if service account doesn't exist): {e}")
    
    logger.info("\n=== Demo completed ===")
    logger.info(
        "Note: These examples will fail if the service account "
        "'katib-trial-sa' doesn't exist."
    )
    logger.info("To make them work, create the service account first:")
    logger.info(
        "kubectl create serviceaccount katib-trial-sa"
    )


if __name__ == "__main__":
    main()
