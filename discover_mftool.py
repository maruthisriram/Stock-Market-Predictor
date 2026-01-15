from mftool import Mftool
import inspect

mf = Mftool()
print("Methods in Mftool:")
methods = [method_name for method_name in dir(mf) if callable(getattr(mf, method_name)) and not method_name.startswith("__")]
for m in methods:
    print(m)

# Check specifically for "performance" or "top" related methods
print("\nChecking for performance methods:")
potential = [m for m in methods if "perform" in m or "top" in m or "gain" in m]
print(potential)
