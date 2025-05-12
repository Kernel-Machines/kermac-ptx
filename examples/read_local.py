from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, ObjectCode

dev = Device()
dev.set_current()
s = dev.create_stream()

mod = ObjectCode.from_ptx('test.ptx')

print(mod.get_kernel('test_kernel').attributes.num_regs())
print(mod.get_kernel('test_kernel').attributes.ptx_version())
print(mod.get_kernel('test_kernel').attributes.binary_version())

