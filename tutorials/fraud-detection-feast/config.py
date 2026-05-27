from dotenv import load_dotenv
import flyte

load_dotenv()

base_env = flyte.TaskEnvironment(
    name="fraud-detection-env",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "feast==0.63.0", "scikit-learn==1.8.0", "xgboost==3.2.0", "joblib",
        "pandas", "pyarrow", "python-dotenv",
        "kagglehub==0.3.12",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)