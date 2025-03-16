# Scripts

This folder contains scripts to run on HSE HPC.

!IMPORTANT!

Run all the scripts from the repository root directory.

Scripts that end with `.sbatch` need to be run with [SLURM](https://hpc.hse.ru/instructions/run/).
However to test them you can also run them with a normal shell.
For this the commands have a `--dry-run` argument, which outputs the resulting command.

## SLURM parameters

The scripts have a header with some parameters for SLURM in a format `#SBATCH --gpus-per-node=1`.
Read more about the arguments [here](https://hpc.hse.ru/instructions/run/).
You can also use a [generator](https://lk.hpc.hse.ru/sbatch/) to configure the parameters.

The ones you need to specify:

`#SBATCH --mail-user=germanarutyunov@gmail.com`

Change this to your email to receive notifications about jobs.
Jobs might fail for multiple reasons. You will need to fix them and run them again.
This is very important to do fast, because for expensive jobs you might need to wait in a queue for a long time.

## Tips

The jobs write output to the log files specified in SLURM parameters.
Remember the job ID to be able to see the logs and errors.

You can check for jobs that are running using `mj` command.
You can also check job statistics at the [dashbaord](https://lk.hpc.hse.ru/).
However, they update after some time, the job might have already crushed before it appears there.

If the job is still in queue you can check when it will be run with `mj --start`.

You can also check the size of the queue with `squeue`.

You need to to preload the necessary modules and save them as default. More [here](https://hpc.hse.ru/instructions/base#module).

## Data Processing

First you will need to preprocess the data for the model.

For this the `data-process.sbatch` should be used.

The default argument values should suffice. 
To know more about the argument you can use the `--help` argument.

```bash
./scripts/data-process.sbatch --help
```

A very important parameter is `--max-tokens`. 
It configures the maximum number of token (sum of nodes and edges) in a graph.
If a graph surpasses the maximum number of token it is ignored.
Some graphs might have too many tokens, which will result in a OOM during training.
First, you will need to tweak the model parameters (more about them in the next section).
However, if the model is too small the result will be poor.
In this case, you will need to filter out some of the outlier huge graphs.

Also check the log for the processed-dir variable. 
You will need it at the training stage to specify the directory from where the files should come from.

Another important parameter is `--split`. It's the data split that will be processed.
You will need to process `train`, `valid` and `test` splits. You can do it in seperate SLURM jobs to accelerate.

## Training

After you have processed the data you can start training the models.
The training is done using the `train.sbatch` script.

You can also check the `--help` output and use `--dry-run` argument to verify the resulting command.

Very important parameteters are the `--model-name` and the `--model-arch`.
You can test the training first with a model arch that has a `_mini` suffix, it can even be run locally.
However, it is very small and useless. 
The model name parameter controls the folder for the checkpoints. 
By default it is the dataset and the arch separated by hyphen.

The arch is just a set of default arguments that define model architecture.
You can specify all the arguments from the command line to overwrite them.

For example, there are several arguments that you might want to tweak to battle the OOM problem.
First, there is the `--batch-size` which is the size of the batch of data that is fed to the model on each step.
A very small value results in model overfitting, which is very bad. You may notice it in the output.
When the training loss decreases, but the validation loss doesn't, this is a sign of model overfitting.

Second, there are a couple of parameters that affect the size of the model:

- `encoder_embed_dim` - it's the dimension of the embedding, i.e. the dimension of the hidden layers
- `encoder_layers` - number of layers of the Transformer encoder
- `encoder_attention_heads` - number of Transformer encoder heads
- `encoder_ffn_embed_dim` - the dimension of the encoder feed forward network embedding
- `lap_node_id_k` - size of the Laplacian embedding, doesn't affect much the size of the model, but should keep it in mind

The number of parameters is present in the output of the training job. 
If you get an OOM you might try to infer how much you need to decrease the size of the model to fit in memory,
since the output usually includes how much torch tried to allocate.
However it is difficult since sometimes it depends on the data.
Therefore, you could apply a binary search approach: set the value in the middle between the lowest and the highest boundary. If it fits in memory, try the next middle point, etc.

There are some constraints on the model parameters. 
For example, the embedding dimension must be divisible by the number of heads.
Also, the dropout cannot be used with Performer.
These constraints will result in a failed asserts in runtime, so you should keep this in mind.

!IMPORTANT!

For some reason if you launch a distributed training on multiple GPUs even if the training results in an error, the script doesn't fail. In this case you should check the logs ocasionally.

You will need to train the model with three losses modifications:

- `--class-loss=True --space-loss=True` for both losses from the [Typilus](https://arxiv.org/pdf/2004.10657) paper
- `--class-loss=True --space-loss=False` for only the classification loss
- `--class-loss=False --space-loss=True` for only the triplet loss

!IMPORTANT!

Since the dataset and the model architecture will be the same, you need to specify the `--model-name` parameter to save checkpoints in separate folders.