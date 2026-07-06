import asyncio

import flyte
import flyte.errors

# The task environment starts with a deliberately small memory budget so the
# first attempt runs out of memory. Self-healing then retries with more.
env = flyte.TaskEnvironment(
    name="oom-self-healing",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)


@env.task
async def oomer(size: int) -> int:
    """Allocate a large list that blows past the 250Mi limit and triggers an OOM."""
    large_list = [0] * size
    return len(large_list)


@env.task
async def always_succeeds() -> int:
    """A lightweight task used to show the pipeline keeps making progress."""
    await asyncio.sleep(1)
    return 42


@env.task
async def failure_recovery() -> int:
    """Catch the OOM error and self-heal by retrying with more memory.

    Flyte surfaces out-of-memory kills as a typed ``flyte.errors.OOMError``,
    so you can catch it and react — here we re-run the same task with a bigger
    memory allocation via ``.override(...)`` instead of failing the whole run.
    """
    try:
        # First attempt runs under the env default (250Mi) and gets OOM-killed.
        await oomer(100_000_000)
    except flyte.errors.OOMError as e:
        print(f"OOM at 250Mi ({e.code}); retrying with more memory...")
        try:
            # Self-heal: same task, more memory.
            await oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(100_000_000)
        except flyte.errors.OOMError as e:
            print(f"OOM again at 1Gi ({e.code}); giving up.")
            raise
    finally:
        # Runs regardless of OOM outcome — the rest of the pipeline still moves.
        await always_succeeds()

    return await always_succeeds()


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(failure_recovery)
    print(run.url)
    run.wait()

# uv run flyte run oom_self_healing.py failure_recovery
