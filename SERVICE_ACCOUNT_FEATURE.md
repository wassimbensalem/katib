# Katib Service Account Feature

This document describes the new service account feature that allows users to specify a Kubernetes service account when creating Katib experiments.

## Overview

The service account feature enables users to run Katib trials with specific Kubernetes service accounts instead of the default service account. This provides better security, access control, and multi-tenancy capabilities.

## Backend Changes (Go)

### 1. Experiment CRD Modification

Added `serviceAccountName` field to `ExperimentSpec` in the Katib API types:

```go
// ServiceAccountName specifies the name of the Kubernetes service account to use for trial pods.
// If not specified, the default service account for the namespace will be used.
// +optional
ServiceAccountName string `json:"serviceAccountName,omitempty"`
```

### 2. Controller Logic

Modified the experiment controller to inject the specified service account into trial pod specifications. The controller now:

- Checks if a service account is specified in the experiment
- Injects the service account into different types of Kubernetes resources:
  - **Kubernetes Jobs**: `spec.template.spec.serviceAccountName`
  - **Training Jobs** (PyTorchJob/TFJob/MPIJob): Each replica type's pod template
  - **Generic Resources**: Common pod template paths

### 3. Service Account Injection

The injection logic handles various resource types:

```go
func (g *DefaultGenerator) injectServiceAccount(runSpec *unstructured.Unstructured, serviceAccountName string) error {
    kind := runSpec.GetKind()
    
    switch kind {
    case "Job":
        return g.injectServiceAccountToJob(runSpec, serviceAccountName)
    case "PyTorchJob", "TFJob", "MPIJob":
        return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
    case "Deployment", "StatefulSet":
        return g.injectServiceAccountToWorkload(runSpec, serviceAccountName)
    // ... other resource types
    }
}
```

### 4. Validation

Added webhook validation to ensure the specified service account exists in the same namespace:

```go
func (g *DefaultValidator) validateServiceAccount(instance *experimentsv1beta1.Experiment) field.ErrorList {
    // Check if the service account exists in the same namespace
    serviceAccount := &corev1.ServiceAccount{}
    if err := g.GetClient().Get(context.Background(), types.NamespacedName{Name: instance.Spec.ServiceAccountName, Namespace: instance.Namespace}, serviceAccount); err != nil {
        // Return validation error
    }
    return nil
}
```

## Python SDK Changes

### 1. Enhanced Client Methods

Updated the `tune` and `create_experiment` methods to accept an optional `service_account` parameter:

```python
def tune(
    self,
    name: str,
    # ... other parameters ...
    service_account: Optional[str] = None,
):
    # ... implementation ...
    if service_account is not None:
        experiment.spec.service_account_name = service_account

def create_experiment(
    self,
    experiment: models.V1beta1Experiment,
    namespace: Optional[str] = None,
    service_account: Optional[str] = None,
):
    # ... implementation ...
    if service_account is not None:
        experiment.spec.service_account_name = service_account
```

### 2. Service Account Injection Utilities

Added utility functions to inject service accounts into trial templates:

```python
def inject_service_account_to_trial_template(
    trial_template: Dict[str, Any], 
    service_account: str
) -> Dict[str, Any]:
    """
    Inject service account into trial template for different Kubernetes resource types.
    """
    kind = trial_template.get('kind', '')
    
    if kind == 'Job':
        return _inject_service_account_to_job(trial_template, service_account)
    elif kind in ['PyTorchJob', 'TFJob', 'MPIJob']:
        return _inject_service_account_to_training_job(trial_template, service_account)
    # ... handle other resource types
```

## Usage Examples

### 1. Using the Python SDK

#### Basic Usage with `tune` method:

```python
from kubeflow.katib import KatibClient

client = KatibClient()

# Create experiment with custom service account
client.tune(
    name="my-experiment",
    objective=my_objective_function,
    parameters={"lr": katib.search.double(min=0.001, max=0.1)},
    service_account="my-custom-sa"  # New parameter
)
```

#### Using `create_experiment` method:

```python
from kubeflow.katib import models

experiment = models.V1beta1Experiment(
    metadata=models.V1ObjectMeta(name="my-exp"),
    spec=models.V1beta1ExperimentSpec(
        # ... other spec fields ...
    )
)

client.create_experiment(
    experiment=experiment,
    service_account="my-custom-sa"  # New parameter
)
```

### 2. Direct YAML Creation

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: my-experiment
spec:
  serviceAccountName: my-custom-sa  # New field
  parameters:
    - name: lr
      parameterType: double
      feasibleSpace:
        min: "0.001"
        max: "0.1"
  objective:
    type: minimize
    objectiveMetricName: loss
  trialTemplate:
    # ... trial template configuration
```

## Benefits

1. **Security**: Trials run with specific permissions instead of default service account
2. **Access Control**: Fine-grained control over what resources trials can access
3. **Multi-tenancy**: Better isolation between different users/teams
4. **Cloud Integration**: Easier access to cloud resources with proper service account bindings
5. **Compliance**: Meet security requirements that mandate specific service accounts

## Implementation Details

### Service Account Injection Targets

The feature automatically injects service accounts into:

- **Kubernetes Jobs**: `spec.template.spec.serviceAccountName`
- **Training Jobs**: Each replica type (master, worker, chief, ps, evaluator, launcher)
- **Workloads**: Deployments, StatefulSets, DaemonSets, ReplicaSets
- **Pods**: Direct pod specifications
- **Generic Resources**: Resources with common pod template patterns

### API Flow

1. User specifies service account name in SDK call
2. SDK validates service account exists and has proper permissions
3. SDK injects service account into trial template before sending to backend
4. Backend controller creates trials with the specified service account
5. Trial pods run with the permissions of the specified service account

## Migration and Compatibility

- **Backward Compatible**: Existing experiments without service accounts continue to work
- **Optional Field**: Service account specification is completely optional
- **Default Behavior**: If no service account is specified, the default namespace service account is used

## Testing

The feature includes comprehensive testing:

- **Unit Tests**: Service account injection logic for different resource types
- **Integration Tests**: End-to-end service account injection in experiments
- **Validation Tests**: Webhook validation of service account existence

## Future Enhancements

Potential future improvements:

1. **Service Account Templates**: Pre-defined service account configurations
2. **Dynamic Service Account Creation**: Automatic creation of service accounts with required permissions
3. **Service Account Validation**: Enhanced validation of service account permissions and roles
4. **Multi-Service Account Support**: Different service accounts for different trial types
