"""What wraps the kernel: the module boundary.

Split from the translation because the kernel body -- what hk-translate
writes from the traced module -- is the same artifact in every deployment and
only its wrapper changes. A pybind module is what `hk.compile()`
loads; a TORCH_LIBRARY registration is what vLLM and SGLang want (Phase 6); an
AOT build wants neither, just the object file. Keeping this in one place means
those three share a single emitter and cannot drift.

The module name is a macro rather than a literal because the compiled artifact
is named after its content hash, and pybind requires PYBIND11_MODULE's name to
match the .so's basename exactly -- get that wrong and the import fails with
"dynamic module does not define module export function", which says nothing
about the real cause.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..types import GlobalType

if TYPE_CHECKING:
    from ..dsl.kernel import TracedKernel

MODULE_NAME_MACRO = "HK_MODULE_NAME"

#: hk dtype -> the at:: scalar type a tensor must have to be read as it.
_AT_DTYPE = {
    "bf16": "at::kBFloat16",
    "fp16": "at::kHalf",
    "fp32": "at::kFloat",
    "i32": "at::kInt",
    "i8": "at::kChar",
    "fp8e4m3": "at::kFloat8_e4m3fn",
    "fp8e5m2": "at::kFloat8_e5m2",
}


def _members(ir: "TracedKernel") -> str:
    return ", ".join(f"&globals::{p.name}" for p in ir.tensors())


def pybind_module(ir: "TracedKernel") -> str:
    """A self-contained pybind11 extension exporting one function, `ir.name`,
    that takes the tensor parameters positionally in declaration order."""
    src = ir.cpp(('"kittens.cuh"', '"pyutils/hk_bind.cuh"'))
    lines = [
        src,
        f"#ifndef {MODULE_NAME_MACRO}",
        f"#define {MODULE_NAME_MACRO} {ir.name}",
        "#endif",
        "",
        f"PYBIND11_MODULE({MODULE_NAME_MACRO}, m) {{",
        # bind_function rather than bind_kernel: the latter re-derives the
        # launch inline, which would duplicate the dynamic-LDS attribute logic
        # that `launch` already has and that the torch path also needs.
        #
        # `py::fast` rather than `py` -- pyutils.cuh's converter spends eleven
        # Python operations and three std::strings per tensor, which is free
        # behind a millisecond kernel and is not free behind a sixty-
        # microsecond one. See include/pyutils/hk_bind.cuh for what it drops
        # and what it keeps.
        f'    kittens::py::fast::bind_function<launch>(m, "{ir.name}", '
        f'{_members(ir)});',
        "}",
    ]
    return "\n".join(lines) + "\n"


def bare(ir: "TracedKernel") -> str:
    """Kernel and `launch`, no module boundary. For AOT objects and for reading
    the generated code when debugging a schedule."""
    return ir.cpp(('"kittens.cuh"',))


def written_tensors(ir: "TracedKernel") -> List[str]:
    """The tensor parameters the kernel stores into, in declaration order.

    A torch schema has to say which arguments are mutated -- get it wrong and
    the functionalization pass in torch.compile will happily reorder a read
    before the write that fed it. The module already knows: a global that is
    the destination of a store is an output and nothing else is. Deriving it
    beats asking the kernel author, who would have to repeat themselves and
    could disagree with their own kernel.
    """
    return ir.written_tensors()


def _alias_letters(n: int) -> List[str]:
    # a, b, c ... one per mutated argument. Two outputs that shared a letter
    # would be declared as aliases of each other, which they are not.
    return [chr(ord("a") + i) for i in range(n)]


def torch_schema(ir: "TracedKernel", op_name: str = "") -> str:
    """The op's torch schema: tensors in declaration order, outputs marked
    mutable, returning nothing. Everything the kernel needs besides tensors is
    a constexpr baked into the source, so there are no other arguments."""
    name = op_name or ir.name
    written = written_tensors(ir)
    letters = dict(zip(written, _alias_letters(len(written))))
    args = []
    for p in ir.tensors():
        ann = f"({letters[p.name]}!)" if p.name in letters else ""
        args.append(f"Tensor{ann} {p.name}")
    return f"{name}({', '.join(args)}) -> ()"


def torch_library(ir: "TracedKernel", ns: str = "hk", op_name: str = "") -> str:
    """A shared object registering one `torch.ops.<ns>.<op>`.

    Out-parameter form on purpose. The kernel cannot say which shape an output
    should have -- the grid is an expression over the globals, and the kernel
    stores wherever it was told to -- so an op that allocated its own result
    would have to guess. Taking the destination as an argument also happens to
    be what a serving framework wants: the buffer is usually already there, and
    an op that does not allocate is one torch can capture in a CUDA graph.

    Three details that are each a silent bug if dropped:

    * `TORCH_LIBRARY_FRAGMENT`, not `TORCH_LIBRARY`. Every kernel compiles to
      its own .so and they all register into one namespace; the non-fragment
      form claims the namespace exclusively and the second .so to load throws.
    * The stream is torch's current stream, not the default stream. On the
      default stream the kernel is ordered against nothing torch did.
    * A `Meta` implementation, so the op can be traced by torch.compile. It is
      a no-op because the op allocates nothing: with the outputs passed in,
      there is no shape for a fake tensor to carry back.
    """
    name = op_name or ir.name
    written = set(written_tensors(ir))
    if not written:
        raise ValueError(
            f"{ir.name} stores into no global, so a torch op over it would "
            f"mutate nothing and return nothing. Store into an output tensor."
        )
    tensors = ir.tensors()
    for p in ir.params:
        if not isinstance(p.type, GlobalType) and not p.is_const:
            raise ValueError(
                f"{ir.name} takes {p.name} as a runtime non-tensor argument, "
                f"which the generated globals struct has no member for. Make "
                f"it a constexpr (it is baked into the source, and each value "
                f"gets its own cached build) or pass it as a tensor."
            )

    def argdecl(p):
        ref = "at::Tensor &" if p.name in written else "const at::Tensor &"
        return f"{ref}{p.name}"

    args = ", ".join(argdecl(p) for p in tensors)
    first = tensors[0].name
    body = []
    for p in tensors:
        dt = _AT_DTYPE.get(p.type.dtype.name)
        if dt is None:
            raise ValueError(
                f"no at:: scalar type for {p.type.dtype}; add it to _AT_DTYPE"
            )
        body.append(f'    hk_check({p.name}, "{p.name}", {dt}, "{ns}::{name}");')
    for p in tensors[1:]:
        body.append(
            f"    TORCH_CHECK({p.name}.get_device() == {first}.get_device(), "
            f'"{ns}::{name}: every tensor must be on one device");'
        )

    init = ",\n".join(
        f"        .{p.name} = hk_gl<decltype(globals::{p.name})>({p.name})"
        for p in tensors
    )

    src = ir.cpp(('"kittens.cuh"',))
    return "\n".join([
        src,
        "#include <torch/library.h>",
        "#include <torch/types.h>",
        "#include <ATen/ATen.h>",
        "#include <c10/hip/HIPStream.h>",
        "#include <c10/hip/HIPGuard.h>",
        "",
        "namespace {",
        "",
        "// Checked rather than fixed: a gl is a base pointer, four extents and",
        "// one row stride, so a tensor it cannot describe is read wrong and not",
        "// read slowly. A slice of the last axis -- contiguous along it, rows",
        "// evenly strided, outer axes compact over the rows -- it can, and an",
        "// engine hands those over (the two halves of a fused gate/up output).",
        "void hk_check(const at::Tensor &t, const char *name,",
        "              at::ScalarType want, const char *op) {",
        '    TORCH_CHECK(t.is_cuda(), op, ": ", name, " must be on the GPU");',
        '    TORCH_CHECK(t.scalar_type() == want, op, ": ", name, " must be ",',
        '                want, ", got ", t.scalar_type());',
        '    TORCH_CHECK(t.dim() >= 1 && t.dim() <= 4, op, ": ", name,',
        '                " must have 1 to 4 dimensions, got ", t.dim());',
        "    if (t.is_contiguous()) return;",
        "    const int nd = (int)t.dim();",
        "    const int64_t row = nd >= 2 ? t.stride(nd - 2) : t.size(nd - 1);",
        "    bool ok = (t.size(nd - 1) == 1 || t.stride(nd - 1) == 1) &&",
        "              row >= t.size(nd - 1);",
        "    if (nd >= 3) ok = ok && (t.size(nd - 3) == 1 ||",
        "                             t.stride(nd - 3) == t.size(nd - 2) * row);",
        "    if (nd >= 4) ok = ok && (t.size(0) == 1 ||",
        "                             t.stride(0) == t.size(1) * t.stride(1));",
        '    TORCH_CHECK(ok, op, ": ", name, " is taken non-contiguous only as a "',
        '                "slice of its last axis; call .contiguous()");',
        "}",
        "",
        "// (B, D, R, C) with the leading dimensions padded, which is the same",
        "// rule pyutils/hk_bind.cuh applies to a pybind call: a 2-D tensor is",
        "// one batch of one depth, so a kernel written against 4-D globals",
        "// takes a matrix without the caller reshaping it.",
        "template <typename GL> GL hk_gl(const at::Tensor &t) {",
        "    int s[4] = {1, 1, 1, 1};",
        "    const int nd = (int)t.dim();",
        "    for (int i = 0; i < nd; ++i) s[4 - nd + i] = (int)t.size(i);",
        "    const int row = t.is_contiguous() || nd < 2 ? -1 : (int)t.stride(nd - 2);",
        "    return kittens::make_gl<GL>((uint64_t)t.data_ptr(), s[0], s[1],",
        "                               s[2], s[3], row);",
        "}",
        "",
        f"void {name}_impl({args}) {{",
        *body,
        "    // DeviceIndex, not DeviceType::CUDA: HIPGuardImpl rejects a CUDA",
        "    // device type outright.",
        f"    const c10::hip::HIPGuard guard((c10::DeviceIndex){first}.get_device());",
        "    globals g{",
        init,
        "    };",
        "    launch_on(g, c10::hip::getCurrentHIPStream());",
        "}",
        "",
        f"void {name}_meta({args}) {{",
        "    // Nothing to do: the op allocates nothing, so tracing it under",
        "    // fake tensors has no result to describe.",
        *[f"    (void){p.name};" for p in tensors],
        "}",
        "",
        "}  // namespace",
        "",
        f"TORCH_LIBRARY_FRAGMENT({ns}, m) {{",
        f'    m.def("{torch_schema(ir, name)}");',
        "}",
        "",
        f"TORCH_LIBRARY_IMPL({ns}, CUDA, m) {{",
        f'    m.impl("{name}", TORCH_FN({name}_impl));',
        "}",
        "",
        f"TORCH_LIBRARY_IMPL({ns}, Meta, m) {{",
        f'    m.impl("{name}", TORCH_FN({name}_meta));',
        "}",
    ]) + "\n"


SCAFFOLDS = {"pybind": pybind_module, "bare": bare, "torch": torch_library}


def render(ir: "TracedKernel", scaffold: str = "pybind") -> str:
    """The whole translation unit: the translated kernel and its wrapper."""
    try:
        fn = SCAFFOLDS[scaffold]
    except KeyError:
        raise ValueError(
            f"unknown scaffold {scaffold!r}; have {sorted(SCAFFOLDS)}"
        ) from None
    return fn(ir)
