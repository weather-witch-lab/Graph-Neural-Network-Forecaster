import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
import os


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

# @title Plotting functions

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    output_dir: str = "output",
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> None:

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # For each timestep, create a separate plot
    for frame in range(max_steps):
        cols = min(cols, len(data))
        rows = math.ceil(len(data) / cols)
        figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
        
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
            
        figure.subplots_adjust(wspace=0, hspace=0)
        figure.tight_layout()

        for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
            ax = figure.add_subplot(rows, cols, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            im = ax.imshow(
                plot_data.isel(time=frame, missing_dims="ignore"), 
                norm=norm,
                origin="lower", 
                cmap=cmap
            )
            plt.colorbar(
                mappable=im,
                ax=ax,
                orientation="vertical",
                pad=0.02,
                aspect=16,
                shrink=0.75,
                cmap=cmap,
                extend=("both" if robust else "neither"))

        # Save the figure
        output_file = os.path.join(output_dir, f"{fig_title.replace(' ', '_')}_{frame:03d}.png")
        figure.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(figure)

# @title Load the model

checkpoint_file = "params/model.npz"
with open(checkpoint_file, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")
model_config

# @title Load weather data
dataset_file = 'dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc'
example_batch = xarray.open_dataset(dataset_file).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))

example_batch

# @title Plot 2m_temperature data

plot_size = 7

data = {
    " ": scale(select(example_batch, '2m_temperature', 500, 20),
              robust=True),
}
fig_title = '2m_temperature'
if "level" in example_batch['2m_temperature'].coords:
  fig_title += f"_at_500_hPa"

plot_data(data, fig_title, "output/temperature", plot_size, True)

# @title Plot 10m_v_component_of_wind data

plot_size = 7

data = {
    " ": scale(select(example_batch, '10m_v_component_of_wind', 500, 20),
              robust=True),
}
fig_title = '10m_v_component_of_wind'
if "level" in example_batch['10m_v_component_of_wind'].coords:
  fig_title += f"_at_500_hPa"

plot_data(data, fig_title, "output/wind", plot_size, True)

# @title Plot total_precipitation_6hr data

plot_size = 7

data = {
    " ": scale(select(example_batch, 'total_precipitation_6hr', 500, 20),
              robust=True),
}
fig_title = 'total_precipitation_6hr'
if "level" in example_batch['total_precipitation_6hr'].coords:
  fig_title += f"_at_500_hPa"

plot_data(data, fig_title, "output/precipitation", plot_size, True)

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{1*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{40*6}h"),
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# @title Load normalization data

diffs_stddev_by_level = xarray.open_dataset("stats/diffs_stddev_by_level.nc").compute()
mean_by_level = xarray.open_dataset("stats/mean_by_level.nc").compute()
stddev_by_level = xarray.open_dataset("stats/stddev_by_level.nc").compute()

  # @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
predictions

# @title Plot 2m_temperature predictions

plot_size = 5
plot_max_steps = min(predictions.dims["time"], 40)

data = {
    "Targets": scale(select(eval_targets, '2m_temperature', 500, plot_max_steps), robust=True),
    "Predictions": scale(select(predictions, '2m_temperature', 500, plot_max_steps), robust=True),
    "Diff": scale((select(eval_targets, '2m_temperature', 500, plot_max_steps) -
                        select(predictions, '2m_temperature', 500, plot_max_steps)),
                       robust=True, center=0),
}
fig_title = '2m_temperature'
if "level" in predictions['2m_temperature'].coords:
  fig_title += f" at {500} hPa"

plot_data(data, fig_title, "prediction/temperature", plot_size, True)

# @title Plot 10m_v_component_of_wind predictions

plot_size = 5
plot_max_steps = min(predictions.dims["time"], 40)

data = {
    "Targets": scale(select(eval_targets, '10m_v_component_of_wind', 500, plot_max_steps), robust=True),
    "Predictions": scale(select(predictions, '10m_v_component_of_wind', 500, plot_max_steps), robust=True),
    "Diff": scale((select(eval_targets, '10m_v_component_of_wind', 500, plot_max_steps) -
                        select(predictions, '10m_v_component_of_wind', 500, plot_max_steps)),
                       robust=True, center=0),
}
fig_title = '10m_v_component_of_wind'
if "level" in predictions['10m_v_component_of_wind'].coords:
  fig_title += f" at {500} hPa"

plot_data(data, fig_title, "prediction/wind", plot_size, True)

# @title Plot total_precipitation_6hr predictions

plot_size = 5
plot_max_steps = min(predictions.dims["time"], 40)

data = {
    "Targets": scale(select(eval_targets, 'total_precipitation_6hr', 500, plot_max_steps), robust=True),
    "Predictions": scale(select(predictions, 'total_precipitation_6hr', 500, plot_max_steps), robust=True),
    "Diff": scale((select(eval_targets, 'total_precipitation_6hr', 500, plot_max_steps) -
                        select(predictions, 'total_precipitation_6hr', 500, plot_max_steps)),
                       robust=True, center=0),
}
fig_title = 'total_precipitation_6hr'
if "level" in predictions['total_precipitation_6hr'].coords:
  fig_title += f" at {500} hPa"

plot_data(data, fig_title, "prediction/precipitation", plot_size, True)