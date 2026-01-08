import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    Model,
    CodeConfiguration,
)

def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    v = os.environ.get(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v)

def main():
    # Workspace
    sub_id = _get_env("AZURE_SUBSCRIPTION_ID", required=True)
    rg = _get_env("AZURE_RESOURCE_GROUP", required=True)
    ws = _get_env("AZUREML_WORKSPACE_NAME", required=True)

    # Deployment names
    endpoint_name = _get_env("ENDPOINT_NAME", "cst-demand-llm-endpoint")
    deployment_name = _get_env("DEPLOYMENT_NAME", "blue")

    # Code + env
    conda_file = _get_env("CONDA_FILE", "infra/conda.yaml")
    code_dir = _get_env("CODE_DIR", "endpoint")
    scoring_script = _get_env("SCORING_SCRIPT", "score.py")

    # Compute
    instance_type = _get_env("INSTANCE_TYPE", "Standard_DS3_v2")
    instance_count = int(_get_env("INSTANCE_COUNT", "1"))

    # Azure OpenAI (required)
    aoai_endpoint = _get_env("AZURE_OPENAI_ENDPOINT", required=True)
    aoai_key = _get_env("AZURE_OPENAI_API_KEY", required=True)
    aoai_api_version = _get_env("AZURE_OPENAI_API_VERSION", "2024-10-21")
    aoai_deployment = _get_env("AZURE_OPENAI_CHAT_DEPLOYMENT", required=True)

    # Demand file shipped with code directory
    demands_path = _get_env("DEMANDS_PATH", "demands.xlsx")
    demands_sheet = _get_env("DEMANDS_SHEET", "Sheet1")

    # Thresholds
    min_prob = _get_env("MIN_PROBABILITY", "0.60")
    rel_thresh = _get_env("RELEVANCE_THRESHOLD", "0.40")
    duplicate = _get_env("DUPLICATE_PREDICTIONS", "true")

    # LLM params
    temperature = _get_env("TEMPERATURE", "0.0")
    top_p = _get_env("TOP_P", "1.0")

    ml_client = MLClient(DefaultAzureCredential(), sub_id, rg, ws)

    # Create/update endpoint
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Register placeholder model (required by managed online deployment)
    model_name = _get_env("MODEL_NAME", "cst-demand-llm-placeholder-model")
    model_path = _get_env("MODEL_PATH", "infra/model_placeholder")
    os.makedirs(model_path, exist_ok=True)
    readme = os.path.join(model_path, "README.txt")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write("Placeholder model. Inference logic is implemented in score.py and uses Azure OpenAI.\n")

    model = Model(path=model_path, name=model_name, type="custom_model")
    model = ml_client.models.create_or_update(model)

    # Create/update environment
    env_name = _get_env("AML_ENV_NAME", "cst-demand-llm-inference-env")
    env = Environment(
        name=env_name,
        description="CST LLM inference env (Azure OpenAI)",
        conda_file=conda_file,
        image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py310-cpu-inference:latest",
    )
    env = ml_client.environments.create_or_update(env)

    # Create/update deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(code=code_dir, scoring_script=scoring_script),
        instance_type=instance_type,
        instance_count=instance_count,
        environment_variables={
            "AZURE_OPENAI_ENDPOINT": aoai_endpoint,
            "AZURE_OPENAI_API_KEY": aoai_key,
            "AZURE_OPENAI_API_VERSION": aoai_api_version,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": aoai_deployment,
            "DEMANDS_PATH": demands_path,
            "DEMANDS_SHEET": demands_sheet,
            "MIN_PROBABILITY": min_prob,
            "RELEVANCE_THRESHOLD": rel_thresh,
            "DUPLICATE_PREDICTIONS": duplicate,
            "TEMPERATURE": temperature,
            "TOP_P": top_p,
        },
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Route traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Endpoint deployed: {endpoint_name} / deployment: {deployment_name}")
    print("Get keys: az ml online-endpoint get-credentials -n", endpoint_name, "-g", rg, "-w", ws)
    print("Invoke:   az ml online-endpoint invoke -n", endpoint_name, "-g", rg, "-w", ws, "--request-file request.json")

if __name__ == "__main__":
    main()
