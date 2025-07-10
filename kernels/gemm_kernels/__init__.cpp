#include <torch/extension.h>
#include <torch/torch.h>
void debug_mbarrier();
// PyBind11 module definition
PYBIND11_MODULE(_C, m) {
  m.def("debug_mbarrier", &debug_mbarrier, "Debug mbarrier in SM90 architecture");
}