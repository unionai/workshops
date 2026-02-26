from dotenv import load_dotenv
import flyte

load_dotenv()

base_env = flyte.TaskEnvironment(
    name="fraud-detection-env",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "feast", "scikit-learn", "xgboost", "joblib",
        "pandas", "pyarrow", "python-dotenv",
        "kagglehub",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)