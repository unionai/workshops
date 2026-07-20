import flyte

# One image for the whole pipeline. Default torch/torchvision wheels ship CUDA,
# so the same image runs the data tasks on CPU nodes and the detector on GPU.
base_image = (
    flyte.Image.from_debian_base(name="lance-streaming-vision")
    # git is needed to pip-install flyteplugins-lance from its GitHub PR until it's
    # published to PyPI (see requirements.txt).
    .with_apt_packages("git").with_requirements("requirements.txt")
)

# GPU environment — detector training, evaluation, and inference.
# T4 (16 GB) is plenty for a MobileNet-backboned Faster R-CNN. Typed "T4:1" so
# the pod lands on a T4 node; if your cluster labels GPUs differently, switch to
# untyped gpu=1 or the right "<TYPE>:1" (a typed request that matches no node
# label silently stays pending).
gpu_env = flyte.TaskEnvironment(
    name="lance-streaming-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
)

# CPU environment — data prep, Lance conversion, the streaming benchmark, and
# the top-level pipeline. Because the pipeline task (on this env) calls tasks in
# gpu_env, we declare that dependency so gpu_env's image is in the cache.
cpu_env = flyte.TaskEnvironment(
    name="lance-streaming-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    depends_on=[gpu_env],
)
