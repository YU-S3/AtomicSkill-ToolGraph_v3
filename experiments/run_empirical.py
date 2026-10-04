"""Repository convenience entry; implementation is included in the wheel."""
import sys
from atomic_skillgraph.experiments import run_empirical as implementation
if __name__ == '__main__':
    implementation.main()
else:
    sys.modules[__name__] = implementation
