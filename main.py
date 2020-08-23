import sys

sys.path.append("lib/MagicCube/code")

from matplotlib.pyplot import show

from cube_interactive import Cube as UICube # pylint: disable=import-error

### Load UI

c = UICube(N=3)
c.draw_interactive()
show()

