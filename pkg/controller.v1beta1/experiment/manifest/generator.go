/*
Copyright 2022 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package manifest

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"

	configv1beta1 "github.com/kubeflow/katib/pkg/apis/config/v1beta1"
	commonapiv1beta1 "github.com/kubeflow/katib/pkg/apis/controller/common/v1beta1"
	experimentsv1beta1 "github.com/kubeflow/katib/pkg/apis/controller/experiments/v1beta1"
	"github.com/kubeflow/katib/pkg/controller.v1beta1/consts"
	"github.com/kubeflow/katib/pkg/controller.v1beta1/util"
	"github.com/kubeflow/katib/pkg/util/v1beta1/katibclient"
	"github.com/kubeflow/katib/pkg/util/v1beta1/katibconfig"
	"log"
)

var (
	errConfigMapNotFound                  = errors.New("configMap not found")
	errConvertStringToUnstructuredFailed  = errors.New("failed to convert string to unstructured")
	errConvertUnstructuredToStringFailed  = errors.New("failed to convert unstructured to string")
	errParamNotFoundInParameterAssignment = errors.New("unable to find non-meta parameter from TrialParameters in ParameterAssignment")
	errParamNotFoundInTrialParameters     = errors.New("unable to find parameter from ParameterAssignment in TrialParameters")
	errTrialTemplateNotFound              = errors.New("unable to find trial template in ConfigMap")
)

// Generator is the type for manifests Generator.
type Generator interface {
	InjectClient(c client.Client)
	GetTrialTemplate(instance *experimentsv1beta1.Experiment) (string, error)
	GetRunSpecWithHyperParameters(experiment *experimentsv1beta1.Experiment, trialName, trialNamespace string, assignments []commonapiv1beta1.ParameterAssignment) (*unstructured.Unstructured, error)
	GetSuggestionConfigData(algorithmName string) (configv1beta1.SuggestionConfig, error)
	GetEarlyStoppingConfigData(algorithmName string) (configv1beta1.EarlyStoppingConfig, error)
	GetMetricsCollectorConfigData(cKind commonapiv1beta1.CollectorKind) (configv1beta1.MetricsCollectorConfig, error)
}

// DefaultGenerator is the default implementation of Generator.
type DefaultGenerator struct {
	client katibclient.Client
}

// New creates a new Generator.
func New(c client.Client) Generator {
	katibClient := katibclient.NewWithGivenClient(c)
	return &DefaultGenerator{
		client: katibClient,
	}
}

func (g *DefaultGenerator) InjectClient(c client.Client) {
	g.client.InjectClient(c)
}

// GetMetricsCollectorConfigData returns metrics collector configuration for a given collector kind.
func (g *DefaultGenerator) GetMetricsCollectorConfigData(cKind commonapiv1beta1.CollectorKind) (configv1beta1.MetricsCollectorConfig, error) {
	return katibconfig.GetMetricsCollectorConfigData(cKind, g.client.GetClient())
}

// GetSuggestionConfigData returns suggestion configuration for a given algorithm name.
func (g *DefaultGenerator) GetSuggestionConfigData(algorithmName string) (configv1beta1.SuggestionConfig, error) {
	return katibconfig.GetSuggestionConfigData(algorithmName, g.client.GetClient())
}

// GetEarlyStoppingConfigData returns early stopping configuration for a given algorithm.
func (g *DefaultGenerator) GetEarlyStoppingConfigData(algorithmName string) (configv1beta1.EarlyStoppingConfig, error) {
	return katibconfig.GetEarlyStoppingConfigData(algorithmName, g.client.GetClient())
}

// GetRunSpecWithHyperParameters returns the specification for trial with hyperparameters.
func (g *DefaultGenerator) GetRunSpecWithHyperParameters(experiment *experimentsv1beta1.Experiment, trialName, trialNamespace string, assignments []commonapiv1beta1.ParameterAssignment) (*unstructured.Unstructured, error) {

	// Apply parameters to Trial Template from assignment
	replacedTemplate, err := g.applyParameters(experiment, trialName, trialNamespace, assignments)
	if err != nil {
		return nil, err
	}
	// Convert Trial template to unstructured
	runSpec, err := util.ConvertStringToUnstructured(replacedTemplate)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", errConvertStringToUnstructuredFailed, err)
	}

	// Set name and namespace for Run Spec
	runSpec.SetName(trialName)
	runSpec.SetNamespace(trialNamespace)

	// Inject service account if specified in experiment
	if experiment.Spec.ServiceAccountName != "" {
		if err := g.injectServiceAccount(runSpec, experiment.Spec.ServiceAccountName); err != nil {
			return nil, fmt.Errorf("failed to inject service account: %w", err)
		}
	}

	return runSpec, nil
}

func (g *DefaultGenerator) applyParameters(experiment *experimentsv1beta1.Experiment, trialName, trialNamespace string, assignments []commonapiv1beta1.ParameterAssignment) (string, error) {
	// Get string Trial template from Experiment spec
	trialTemplate, err := g.GetTrialTemplate(experiment)
	if err != nil {
		return "", err
	}

	trialSpec := experiment.Spec.TrialTemplate.TrialSpec
	// If trialSpec is not defined in TrialTemplate, deserialize templateString to fetch it
	if trialSpec == nil {
		trialSpec, err = util.ConvertStringToUnstructured(trialTemplate)
		if err != nil {
			return "", fmt.Errorf("%w: %w", errConvertStringToUnstructuredFailed, err)
		}
	}

	// Convert parameter assignment to map key = parameter name, value = parameter value
	assignmentsMap := make(map[string]string)
	for _, assignment := range assignments {
		assignmentsMap[assignment.Name] = assignment.Value
	}

	placeHolderToValueMap := make(map[string]string)
	var metaRefKey, metaRefIndex string
	nonMetaParamCount := 0
	for _, param := range experiment.Spec.TrialTemplate.TrialParameters {
		metaMatchRegex := regexp.MustCompile(consts.TrialTemplateMetaReplaceFormatRegex)
		sub := metaMatchRegex.FindStringSubmatch(param.Reference)
		// handle trial parameters which consume trial assignments
		if len(sub) == 0 {
			if value, ok := assignmentsMap[param.Reference]; ok {
				placeHolderToValueMap[param.Name] = value
				nonMetaParamCount += 1
				continue
			} else {
				return "", fmt.Errorf("%w: parameter: %v, parameter assignment: %v", errParamNotFoundInParameterAssignment, param.Reference, assignmentsMap)
			}
		}
		metaRefKey = sub[1]

		// handle trial parameters which consume trial meta data
		// extract index (key) of Labels and Annotations if exists
		if sub := regexp.MustCompile(consts.TrialTemplateMetaParseFormatRegex).FindStringSubmatch(metaRefKey); len(sub) > 0 {
			if len(sub) != 3 {
				return "", fmt.Errorf("illegal reference of trial metadata: %v", param.Reference)
			}
			metaRefKey = sub[1]
			metaRefIndex = sub[2]
		}
		// fetch metadata value
		switch metaRefKey {
		case consts.TrialTemplateMetaKeyOfName:
			placeHolderToValueMap[param.Name] = trialName
		case consts.TrialTemplateMetaKeyOfNamespace:
			placeHolderToValueMap[param.Name] = trialNamespace
		case consts.TrialTemplateMetaKeyOfKind:
			placeHolderToValueMap[param.Name] = trialSpec.GetKind()
		case consts.TrialTemplateMetaKeyOfAPIVersion:
			placeHolderToValueMap[param.Name] = trialSpec.GetAPIVersion()
		case consts.TrialTemplateMetaKeyOfAnnotations:
			if value, ok := trialSpec.GetAnnotations()[metaRefIndex]; !ok {
				return "", fmt.Errorf("illegal reference of trial metadata: %v; failed to fetch Annotation: %v", param.Reference, metaRefIndex)
			} else {
				placeHolderToValueMap[param.Name] = value
			}
		case consts.TrialTemplateMetaKeyOfLabels:
			if value, ok := trialSpec.GetLabels()[metaRefIndex]; !ok {
				return "", fmt.Errorf("illegal reference of trial metadata: %v; failed to fetch Label: %v", param.Reference, metaRefIndex)
			} else {
				placeHolderToValueMap[param.Name] = value
			}
		default:
			return "", fmt.Errorf("illegal reference of trial metadata: %v", param.Reference)
		}
	}

	// Number of assignment parameters must be equal to the number of non-meta trial parameters
	// i.e. all parameters in ParameterAssignment must be in TrialParameters
	if len(assignments) != nonMetaParamCount {
		return "", fmt.Errorf("%w: parameter assignments: %v, non-meta trial parameter count: %v", errParamNotFoundInTrialParameters, assignments, nonMetaParamCount)
	}

	// Replacing placeholders with parameter values
	for placeHolder, paramValue := range placeHolderToValueMap {
		trialTemplate = strings.Replace(trialTemplate, fmt.Sprintf(consts.TrialTemplateParamReplaceFormat, placeHolder), paramValue, -1)
	}

	return trialTemplate, nil
}

// GetTrialTemplate returns string Trial template from experiment
func (g *DefaultGenerator) GetTrialTemplate(instance *experimentsv1beta1.Experiment) (string, error) {
	var trialTemplateString string
	var err error

	trialSource := instance.Spec.TrialTemplate.TrialSource
	if trialSource.TrialSpec != nil {
		trialTemplateString, err = util.ConvertUnstructuredToString(trialSource.TrialSpec)
		if err != nil {
			return "", fmt.Errorf("%w: %w", errConvertUnstructuredToStringFailed, err)
		}
	} else {
		configMapNS := trialSource.ConfigMap.ConfigMapNamespace
		configMapName := trialSource.ConfigMap.ConfigMapName
		templatePath := trialSource.ConfigMap.TemplatePath
		configMap, err := g.client.GetConfigMap(configMapName, configMapNS)
		if err != nil {
			return "", fmt.Errorf("%w: %w", errConfigMapNotFound, err)
		}
		var ok bool
		trialTemplateString, ok = configMap[templatePath]
		if !ok {
			return "", fmt.Errorf("%w: TemplatePath: %v, ConfigMap: %v", errTrialTemplateNotFound, templatePath, configMap)
		}
	}

	return trialTemplateString, nil
}

// injectServiceAccount injects the specified service account name into the trial template.
// It handles different types of Kubernetes resources that may contain pod templates.
func (g *DefaultGenerator) injectServiceAccount(runSpec *unstructured.Unstructured, serviceAccountName string) error {
	kind := runSpec.GetKind()
	
	switch kind {
	case "Job":
		return g.injectServiceAccountToJob(runSpec, serviceAccountName)
	case "PyTorchJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "TFJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "MPIJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "MXJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "XGBoostJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "PaddleJob":
		return g.injectServiceAccountToTrainingJob(runSpec, serviceAccountName)
	case "Deployment":
		return g.injectServiceAccountToDeployment(runSpec, serviceAccountName)
	case "StatefulSet":
		return g.injectServiceAccountToStatefulSet(runSpec, serviceAccountName)
	case "DaemonSet":
		return g.injectServiceAccountToDaemonSet(runSpec, serviceAccountName)
	case "ReplicaSet":
		return g.injectServiceAccountToReplicaSet(runSpec, serviceAccountName)
	case "Pod":
		return g.injectServiceAccountToPod(runSpec, serviceAccountName)
	default:
		// For other resource types, try to inject into pod template if it exists
		return g.injectServiceAccountToGenericResource(runSpec, serviceAccountName)
	}
}

// injectServiceAccountToJob injects service account into a Kubernetes Job
func (g *DefaultGenerator) injectServiceAccountToJob(job *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(job.Object, "spec", "template", "spec")
	if err != nil {
		return fmt.Errorf("failed to get job spec: %w", err)
	}
	if !found {
		return fmt.Errorf("job spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(job.Object, spec, "spec", "template", "spec"); err != nil {
		return fmt.Errorf("failed to set job spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToTrainingJob injects service account into training job types (PyTorchJob, TFJob, etc.)
func (g *DefaultGenerator) injectServiceAccountToTrainingJob(job *unstructured.Unstructured, serviceAccountName string) error {
	// Training jobs have multiple replica types, each with their own pod template
	replicaTypes := []string{"master", "worker", "chief", "ps", "evaluator", "launcher"}
	
	for _, replicaType := range replicaTypes {
		spec, found, err := unstructured.NestedMap(job.Object, "spec", replicaType, "template", "spec")
		if err != nil {
			continue // Skip if this replica type doesn't exist
		}
		if found {
			spec["serviceAccountName"] = serviceAccountName
			if err := unstructured.SetNestedMap(job.Object, spec, "spec", replicaType, "template", "spec"); err != nil {
				return fmt.Errorf("failed to set %s spec: %w", replicaType, err)
			}
		}
	}
	
	return nil
}

// injectServiceAccountToDeployment injects service account into a Kubernetes Deployment
func (g *DefaultGenerator) injectServiceAccountToDeployment(deployment *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(deployment.Object, "spec", "template", "spec")
	if err != nil {
		return fmt.Errorf("failed to get deployment spec: %w", err)
	}
	if !found {
		return fmt.Errorf("deployment spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(deployment.Object, spec, "spec", "template", "spec"); err != nil {
		return fmt.Errorf("failed to set deployment spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToStatefulSet injects service account into a Kubernetes StatefulSet
func (g *DefaultGenerator) injectServiceAccountToStatefulSet(statefulSet *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(statefulSet.Object, "spec", "template", "spec")
	if err != nil {
		return fmt.Errorf("failed to get statefulset spec: %w", err)
	}
	if !found {
		return fmt.Errorf("statefulset spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(statefulSet.Object, spec, "spec", "template", "spec"); err != nil {
		return fmt.Errorf("failed to set statefulset spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToDaemonSet injects service account into a Kubernetes DaemonSet
func (g *DefaultGenerator) injectServiceAccountToDaemonSet(daemonSet *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(daemonSet.Object, "spec", "template", "spec")
	if err != nil {
		return fmt.Errorf("failed to get daemonset spec: %w", err)
	}
	if !found {
		return fmt.Errorf("daemonset spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(daemonSet.Object, spec, "spec", "template", "spec"); err != nil {
		return fmt.Errorf("failed to set daemonset spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToReplicaSet injects service account into a Kubernetes ReplicaSet
func (g *DefaultGenerator) injectServiceAccountToReplicaSet(replicaSet *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(replicaSet.Object, "spec", "template", "spec")
	if err != nil {
		return fmt.Errorf("failed to get replicaset spec: %w", err)
	}
	if !found {
		return fmt.Errorf("replicaset spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(replicaSet.Object, spec, "spec", "template", "spec"); err != nil {
		return fmt.Errorf("failed to set replicaset spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToPod injects service account into a Kubernetes Pod
func (g *DefaultGenerator) injectServiceAccountToPod(pod *unstructured.Unstructured, serviceAccountName string) error {
	spec, found, err := unstructured.NestedMap(pod.Object, "spec")
	if err != nil {
		return fmt.Errorf("failed to get pod spec: %w", err)
	}
	if !found {
		return fmt.Errorf("pod spec not found")
	}
	
	spec["serviceAccountName"] = serviceAccountName
	
	if err := unstructured.SetNestedMap(pod.Object, spec, "spec"); err != nil {
		return fmt.Errorf("failed to set pod spec: %w", err)
	}
	
	return nil
}

// injectServiceAccountToGenericResource attempts to inject service account into any resource that might have a pod template
func (g *DefaultGenerator) injectServiceAccountToGenericResource(resource *unstructured.Unstructured, serviceAccountName string) error {
	// Try to find pod template in common locations
	podTemplatePaths := [][]string{
		{"spec", "template", "spec"},
		{"spec", "jobTemplate", "spec", "template", "spec"},
		{"spec", "workflowTemplate", "spec", "template", "spec"},
	}
	
	for _, path := range podTemplatePaths {
		spec, found, err := unstructured.NestedMap(resource.Object, path...)
		if err != nil {
			continue
		}
		if found {
			spec["serviceAccountName"] = serviceAccountName
			if err := unstructured.SetNestedMap(resource.Object, spec, path...); err != nil {
				continue // Try next path
			}
			return nil // Successfully injected
		}
	}
	
	// If no pod template found, log a warning but don't fail
	log.Printf("Warning: Could not inject service account into resource of kind %s", resource.GetKind())
	return nil
}
