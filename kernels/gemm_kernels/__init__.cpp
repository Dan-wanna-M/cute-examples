#include <torch/extension.h>
#include <torch/torch.h>
namespace test {
torch::Tensor gemm_tn_test(const torch::Tensor A, const torch::Tensor B, torch::Tensor C);
// PyBind11 module definition
PYBIND11_MODULE(_C, m) {
  m.def("gemm_tn_test", &gemm_tn_test, "GEMM transpose-normal test function",
        pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
}
} // namespace test