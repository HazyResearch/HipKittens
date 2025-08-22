# ThunderKittens

### AMD Specific Setup

Need to install special PyTorch that targets AMD

From the [PyTorch website](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
```


### Tile primitives for speedy kernels

<div align="center" >
    <img src="assets/thunderkittens.png" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

ThunderKittens is a framework to make it easy to write fast deep learning kernels.

ThunderKittens is built around three key principles:
1. Simplicity. ThunderKittens is stupidly simple to write.
2. Extensibility. ThunderKittens embeds itself natively, so that if you need more than ThunderKittens can offer, it won’t get in your way of building it yourself.
3. Speed. Kernels written in ThunderKittens should be at least as fast as those written from scratch -- especially because ThunderKittens can do things the “right” way under the hood. We think our Flash Attention 3 implementation speaks for this point.

Join us on Discord to get involved: [ThunderKittens channel @ GPU Mode Discord](https://discord.com/channels/1189498204333543425/1300872762163728550)!!!! Here is the invite link to GPU mode: https://discord.gg/gpumode

ThunderKittens is built from the hardware up -- we do what the silicon tells us. And modern GPUs tell us that they want to work with fairly small tiles of data. A GPU is not really a 1000x1000 matrix multiply machine (even if it is often used as such); it’s a manycore processor where each core can efficiently run ~16x16 matrix multiplies. Consequently, ThunderKittens is built around manipulating tiles of data no smaller than 16x16 values.

ThunderKittens makes a few tricky things easy that enable high utilization on modern hardware.
1. Tensor cores. ThunderKittens can call fast tensor core functions, including asynchronous WGMMA calls on H100 GPUs.
2. Shared Memory. I got ninety-nine problems but a bank conflict ain’t one.
3. Loads and stores. Hide latencies with asynchronous copies and address generation with TMA.
4. Distributed Shared Memory. L2 is _so_ last year.
5. Worker overlapping. Use our Load-Store-Compute-Finish template to overlap work and I/O.


## Installation

To use Thunderkittens, there's not all that much you need to do with TK itself. It's a header only library, so just clone the repo, and include kittens.cuh. Easy money.

### Library requirements

But ThunderKittens does use a bunch of modern stuff, so it has fairly aggressive requirements.

```bash
sudo apt update
sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo apt update
sudo apt install clang-11
```

## Tests

To validate your install, and run TK's fairly comprehensive unit testing suite, simply run `make -j` in the tests folder. Be warned: this may nuke your computer for a minute or two while it compiles thousands of kernels.

## ThunderKittens Manual

[In-progress onboarding document](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing). Please contribute to this if you've run into issues and feel the broader community can benefit from explanations. Please leave comments if any aspect of this is unclear. 

ThunderKittens is actually a pretty small library, in terms of what it gives you.

 - Data types: (Register + shared) * (tiles + vectors), all parameterized by layout, type, and size.
 - Operations for manipulating these objects.

Despite its simplicity, there are still a few sharp edges that you might encounter if you don’t know what’s going on under the hood. So, we do recommend giving this manual a good read before sitting down to write a kernel -- it’s not too long, we promise!

### Typing

ThunderKittens tries hard to protect you from yourself. In particular, ThunderKittens wants to know layouts of objects at compile-time and will make sure they’re compatible before letting you do operations. This is important because there are subtleties to the allowable layouts for certain operations, and without static checks it is very easy to get painful silent failures. For example, a normal matrix multiply requires the B operand to be in a column layout, whereas an outer dot product requires the B operand to be in a row layout.

If you are being told an operation that you think exists doesn't exist, double-check your layouts -- this is the most common error. Only then report a bug :)

### Scopes

By default, ThunderKittens operations exist at the warp-level. In other words, each function expects to be called by only a single warp, and that single warp will do all of the work of the function. If multiple warps are assigned to the same work, undefined behavior will result. (And if the operation involves memory movement, it is likely to be completely catastrophic.) In general, you should expect your programming pattern to involve instantiating a `warpid` at the beginning of the kernel with `kittens::warpid()`, and assigning tasks to data based on that id.

However, not all ThunderKittens functions operate at the warp level. Many important operations, particularly WGMMA instructions, require collaborative groups of warps. These operations exist in the templated `kittens::group<collaborative size>`. For example, wgmma instructions are available through `kittens::group<4>::mma_AB` (or `kittens::warpgroup::mma_AB`, which is an alias.) Groups of warps can also collaboratively load shared memory or do reductions in shared memory

### Other Restrictions

Most operations in ThunderKittens are pure functional. However, some operations _do_ have special restrictions; ThunderKittens tries to warn you by giving them names that stand out. For example, a register tile transpose needs separable arguments: if it is given the same underlying registers as both source and destination, it will silently fail. Consequently, it is named `transpose_sep`.


## Learn more and get involved!

This repository is maintained by:

- William Hu [willhu@stanford.edu](willhu@stanford.edu)
- Simran Arora [simran@cs.stanford.edu](simran@cs.stanford.edu)


The Nvidia ThunderKittens repository is at: https://github.com/HazyResearch/ThunderKittens

Learn more about the ThunderKittens project and how GPUs work by checking out our prior blogs:
- [Easier, Better, Faster, Cuter Blogpost, Oct. 2024](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)
- [GPUs Go Brrr Blogpost, May 2024](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [ThunderKittens: Bringing fp8 to theaters near you, Nov 2024](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)
- [ThunderMittens For Your ThunderKittens, Nov 2024](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)

Please check out our paper for even more details: [arxiv](https://arxiv.org/abs/2410.20399).

